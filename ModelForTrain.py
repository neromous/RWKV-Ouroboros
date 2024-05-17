import types
import functools
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch._lowrank import svd_lowrank
from lion_pytorch import Lion
from prefix_tokenizer import prefix_tokenizer
from torch.cuda.amp import autocast,GradScaler
import deepspeed
#from apex import amp

os.environ["RWKV_TORCH_COMPILE"] = ""
os.environ["RWKV_MY_TESTING"] = ""


HEAD_SIZE = 64

LORA_CONFIG = {
    "r": 64,
    "alpha": 128,
    "dropout": 0,
    "parts": {"att", "ln", "time", "ffn"},
}

class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG["alpha"], LORA_CONFIG["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.r = r
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.pissa = False

    def pissa_init(self, svd_niter):
        self.pissa = True
        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A

    def forward(self, x):
        if self.pissa:
            return (F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B))
        return (F.linear(x, self.weight) + self.scaling * F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)

def __nop(ob):
    return ob
MyModule = nn.Module
MyFunction = __nop

from torch.utils.cpp_extension import load

wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", "cuda/wkv6_cuda.cu"],
                 verbose=True,
                 extra_cuda_cflags=["-res-usage",
                                    "--use_fast_math", "-O3", "-Xptxas -O3",
                                    "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                    f"-D_T_={512}"])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                            memory_format=torch.contiguous_format)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False,
                             dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False,
                             dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False,
                             dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False,
                             dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False,
                             dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA(B, T, C, H, r, k, v, w, u):
    return WKV.apply(B, T, C, H, r, k, v, w, u)


class RWKV_Tmix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            if '7b' in args.load_model:
                TIME_MIX_EXTRA_DIM = TIME_MIX_EXTRA_DIM*2
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            if '7b' in args.load_model:
                TIME_DECAY_EXTRA_DIM = TIME_DECAY_EXTRA_DIM*2
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-1e-4, 1e-4))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.key = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.value = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.output = make_linear_att(args.dim_att, args.n_embd, bias=False)
        self.gate = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        r, k, v, g, w = self.jit_func(x)

        x = RUN_CUDA(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

class RWKV_CMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False)
        self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))



class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix(args, layer_id)
        self.ffn = RWKV_CMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()

        if self.layer_id == 0:
            x = self.ln0(x)
            # if args.my_pos_emb > 0:
            #     pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
            #     x = x + pos_emb

        if self.args.dropout == 0:
            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def forward(self, batch, batch_idx):
        args = self.args
        idx, targets, mask = batch
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.lora:
                    x = torch_checkpoint(block, x, x_emb ,use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_out(x)
        x = self.head(x)
        logits = x
        if sum_mask == mask.shape[0]:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = torch.sum(loss * mask) / sum_mask

        return L2Wrap.apply(loss, logits)

args = types.SimpleNamespace()

args.lora= 1
args.dropout = 0
args.dim_ffn = 0
args.dim_att = 0
args.load_model = "./resources/weights/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
#args.load_model = "./resources/weights/RWKV-x060-World-3B-v2-20240228-ctx4096.pth"
args.n_layer = 32
args.n_embd = 2048
args.head_size_a = 64
args.ctx_len = 2048
args.grad_cp = 1
args.vocab_size = 65536
args.lora_r = 32
args.lora_dropout=0.01
args.lora_alpha = 64
args.lora_parts= "att,ln,time,ffn"
args.PISSA = 1
args.svd_niter=4
args.head_size_divisor=8
args.emb = "store_true"
if args.dim_att == 0:
    args.dim_att = 2048
if args.dim_ffn <= 0:
    args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
args.proj_dir = "./resources"

model = RWKV(args).to('cuda')

# lora 配置
if args.lora:
    assert args.lora_r > 0, "LoRA should have its `r` > 0"
    LORA_CONFIG["r"] = args.lora_r
    LORA_CONFIG["alpha"] = args.lora_alpha
    LORA_CONFIG["dropout"] = args.lora_dropout
    LORA_CONFIG["parts"] = set(str(args.lora_parts).split(','))
    enable_time_finetune = 'time' in LORA_CONFIG["parts"]
    enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]

# 配置模型的加载
if args.lora or args.LISA:
    model.requires_grad_(False)
    for name, module in model.named_modules():
        if any(n.startswith("emb.") for n, _ in module.named_parameters()):
            for pname, param in module.named_parameters():
                if args.emb and 'emb.weight' == pname:
                    print(f'  EMB additionally training module {pname}')
                    param.requires_grad = True
        if any(n.startswith("head.") for n, _ in module.named_parameters()):
            for pname, param in module.named_parameters():
                if args.emb and 'head.weight'==pname:
                    print(f'  head additionally training module {pname}')
                    param.requires_grad = True
        if any(n.startswith("lora_") for n, _ in module.named_parameters()):
            print(f'  LoRA additionally training module {name}')
            for pname, param in module.named_parameters():
                param.requires_grad = 'lora_' in pname
        elif enable_ln_finetune and '.ln' in name:
            print(f'  LoRA additionally training module {name}')
            for param in module.parameters():
                param.requires_grad = True
        elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
            for pname, param in module.named_parameters():
                if pname.startswith("time"):
                    print(f'  LoRA additionally training parameter {pname}')
                    param.requires_grad = True

try:
    load_dict = torch.load(args.load_model, map_location="cuda")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            load_dict[k.replace('_forward_module.','')] = load_dict[k]
            del load_dict[k]
except:
    pass

load_dict = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in load_dict.items()}

model.load_state_dict(load_dict, strict=(not args.lora))

if args.PISSA:
      init_dict = {}
      for name, m in model.named_modules():
          if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
              m.pissa_init(args.svd_niter)
              init_dict[f'{name}.init_lora_A'] = m.lora_A.data
              init_dict[f'{name}.init_lora_B'] = m.lora_B.data
      torch.save(init_dict, f'{args.proj_dir}/init_lora.pth')
model.to(dtype=torch.bfloat16)
lr = 0.0001 # learning rate

#optimizer = Lion(model.parameters(), lr=1e-4,weight_decay=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=8,
                                            gamma=0.5,
                                            last_epoch=-1)
parameters = model.parameters()

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config="./resources/ds_config/ds_config.config",
)

tokenizer = prefix_tokenizer()

idx_origin = tokenizer.encode("""第一章 游动的暗礁
1866年，发生了一起非同寻常的事件，那真是一种神秘而又无法解释的现象。无疑，人们对当时的情景至今仍不能忘怀。且不说当时那些沿海地区的居民对此感到兴奋异常，到处散布各种传闻，即使是内陆地区的居民也被各种消息搞得心神不宁，情绪极为激动。尤其是那些从事航海工作的海员，他们对这件事尤其充满了兴趣。欧洲、美洲两个大陆的商人、普通海员、船主、船长、各国的海军军官以及两大洲的各国政府等等，大家都密切地关注着事态的进一步发展。
事情的原委是这样的：不久前，有几艘轮船在海上遇到一头“庞然大物”，那是一个长长的东西，呈纺锤形，有时全身散发着磷光，而且它的体积超过了一头鲸鱼，动作也比鲸鱼迅速得多。
有关这个离奇怪物的出现，各种航海日志都留下相关记载。这些日志大都记载了这个物体或者说可疑生物的形状、它在运动时一直保持的高速，以及它令人惊异的运动能量。它那奇特的生命力似乎是与生俱来的本能。如果它是一种鲸类动物，可是它的身体尺寸却超过了迄今为止生物学界研究过的各类鲸鱼。居维叶〔1〕、拉塞佩德〔2〕、迪梅里、德·卡特法日这些博物学家是不会承认这种怪物的存在的，除非他们看到过它，也就是说除非这些科学家亲眼目睹了这头怪物的存在。
综合考虑人们的多次观察结果——我们排除了那些最保守的估计，他们认为这头怪物只有200英尺〔3〕长，同时我们也不能接受那些过于夸张的观点，认为这个怪物足有1英里宽、3英里〔4〕长——最后，我们可以比较公正地得出结论说，如果这个神秘的物体果真存在，那么这个存在物的体积，大大地超过了当前所有鱼类学家所认可的尺寸
事情的原委是这样的：不久前，有几艘轮船在海上遇到一头“庞然大物”，那是一个长长的东西，呈纺锤形，有时全身散发着磷光，而且它的体积超过了一头鲸鱼，动作也比鲸鱼迅速得多。
有关这个离奇怪物的出现，各种航海日志都留下相关记载。这些日志大都记载了这个物体或者说可疑生物的形状、它在运动时一直保持的高速，以及它令人惊异的运动能量。它那奇特的生命力似乎是与生俱来的本能。如果它是一种鲸类动物，可是它的身体尺寸却超过了迄今为止生物学界研究过的各类鲸鱼。居维叶〔1〕、拉塞佩德〔2〕、迪梅里、德·卡特法日这些博物学家是不会承认这种怪物的存在的，除非他们看到过它，也就是说除非这些科学家亲眼目睹了这头怪物的存在。
综合考虑人们的多次观察结果——我们排除了那些最保守的估计，他们认为这头怪物只有200英尺〔3〕长，同时我们也不能接受那些过于夸张的观点，认为这个怪物足有1英里宽、3英里〔4〕长——最后，我们可以比较公正地得出结论说，如果这个神秘的物体果真存在，那么这个存在物的体积，大大地超过了当前所有鱼类学家所认可的尺寸
事情的原委是这样的：不久前，有几艘轮船在海上遇到一头“庞然大物”，那是一个长长的东西，呈纺锤形，有时全身散发着磷光，而且它的体积超过了一头鲸鱼，动作也比鲸鱼迅速得多。
有关这个离奇怪物的出现，各种航海日志都留下相关记载。这些日志大都记载了这个物体或者说可疑生物的形状、它在运动时一直保持的高速，以及它令人惊异的运动能量。它那奇特的生命力似乎是与生俱来的本能。如果它是一种鲸类动物，可是它的身体尺寸却超过了迄今为止生物学界研究过的各类鲸鱼。居维叶〔1〕、拉塞佩德〔2〕、迪梅里、德·卡特法日这些博物学家是不会承认这种怪物的存在的，除非他们看到过它，也就是说除非这些科学家亲眼目睹了这头怪物的存在。
综合考虑人们的多次观察结果——我们排除了那些最保守的估计，他们认为这头怪物只有200英尺〔3〕长，同时我们也不能接受那些过于夸张的观点，认为这个怪物足有1英里宽、3英里〔4〕长——最后，我们可以比较公正地得出结论说，如果这个神秘的物体果真存在，那么这个存在物的体积，大大地超过了当前所有鱼类学家所认可的尺寸""")
print("======",len(idx_origin))
idx_origin = idx_origin[:512]
idx = torch.tensor([idx_origin[:-1]],dtype=torch.long).to('cuda')
target = torch.tensor([idx_origin[1:]],dtype=torch.long).to('cuda')
mask = torch.tensor([[1 for x in target]],dtype=torch.bfloat16).to('cuda')
for x in range(0,100):
    loss = model_engine((idx,target,mask), 0)
    model_engine.backward(loss)
    model_engine.step()
    print(loss.item())
