########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
from config import config


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop



########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load


HEAD_SIZE = int(config['model']["head_size_a"])



if config['model']["dtype"] == "bf16":
    wkv5_cuda = load(name="wkv5",
                     sources=["cuda/v5/wkv5_op.cpp",
                              f"cuda/v5/wkv5_cuda.cu"],
                     verbose=True,
                     extra_cuda_cflags=["-res-usage",
                                        "--use_fast_math", "-O3", "-Xptxas -O3",
                                        "--extra-device-vectorization",
                                        f"-D_N_={HEAD_SIZE}"])

    class WKV_5(torch.autograd.Function):
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
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C),
                                device=r.device,
                                dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
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
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)
elif config['model']["dtype"] == "fp32":
    wkv5_cuda = load(name="wkv5_fp32",
                     sources=["cuda/v5/wkv5_op_fp.cpp",
                              f"cuda/v5/wkv5_cuda_fp.cu"],
                     verbose=True,
                     extra_cuda_cflags=["-res-usage",
                                        "--use_fast_math", "-O3", "-Xptxas -O3",
                                        "--extra-device-vectorization",
                                        f"-D_N_={HEAD_SIZE}"])

    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.float
                assert k.dtype == torch.float
                assert v.dtype == torch.float
                assert w.dtype == torch.float
                assert u.dtype == torch.float
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
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C),
                                device=r.device,
                                dtype=torch.float,
                                memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.float
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

elif config['model']["dtype"] == "fp16":
    wkv5_cuda = load(name="wkv5_fp32",
                     sources=["cuda/v5/wkv5_op_fp.cpp",
                              f"cuda/v5/wkv5_cuda_fp.cu"],
                     verbose=True,
                     extra_cuda_cflags=["-res-usage",
                                        "--use_fast_math", "-O3", "-Xptxas -O3",
                                        "--extra-device-vectorization",
                                        f"-D_N_={HEAD_SIZE}"])

    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.half
                assert k.dtype == torch.half
                assert v.dtype == torch.half
                assert w.dtype == torch.half
                assert u.dtype == torch.half
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
                eew = (torch.exp(ew)).contiguous()

                r = r.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
                u = u.float().contiguous()

                ctx.save_for_backward(r,
                                      k,
                                      v,
                                      eew,
                                      ew,
                                      u)
                y = torch.empty((B, T, C),
                                device=r.device,
                                dtype=torch.float,
                                memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y.half()

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.half
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy.float().contiguous(), gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr.half(), gk.half(), gv.half(), gw.half(), gu.half())


def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_ChannelMix(MyModule):
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
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
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

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        if 'g' in config['environ']["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.args.dropout == 0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
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


class RWKV(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.dtype == "fp32":
            self.args.dtype=torch.float32
        elif self.args.dtype == "fp16":
            self.args.dtype=torch.float16
        elif self.args.dtype== "bf16":
            self.args.dtype = torch.bfloat16

        model_weights = torch.load(self.args.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())

        if self.args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            self.args.n_layer = max_block_id + 1

        if self.args.n_embd < 0:
            self.args.n_embd = model_weights['head.weight'].shape[1]

        if self.args.vocab_size < 0:
            self.args.vocab_size = model_weights['head.weight'].shape[0]


        self.args.dim_att = self.args.n_embd
        self.args.dim_ffn = int((self.args.n_embd * 3.5) // 32 * 32)
        if not hasattr(args, 'tiny_att_layer'):
            self.args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            self.args.tiny_att_dim = -1


        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

        if model_weights != None:
            # Print the loading event
            print(f"[RWKV.model]: Loading model weights ( L{args.n_layer}-D{args.n_embd}-V{args.vocab_size} )")
            model_weights = {k:v.to(dtype=self.args.dtype) for k,v
                             in model_weights.items()}
            self.load_state_dict(model_weights)
            del model_weights
            gc.collect()

        print(f"[RWKV.model]: Finished initial model load")


    def get_optimizers(self):
        args = self.args
        lr_init= args.lr_init
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if "time_mix" in n:
                lr_1x.add(n)
            elif "time_decay" in n:
                lr_2x.add(n)
            elif "time_first" in n:
                lr_3x.add(n)
            elif "time_faaaa" in n:
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}

        if args.dtype == torch.float32:
            optim_groups = [
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": args.weight_decay,
                    "lr": 3.0 * lr_init
                },
            ]
            if args.weight_decay > 0:
                print("===load weight_decay fp32 ===")
                optim_groups += [{"params": [param_dict[n] for n in lr_decay],
                                  "weight_decay": args.weight_decay,
                                  "my_lr_scale": 1.0}]
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(args.beta1, args.beta2),
                                         eps=args.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=args.adamw_mode,
                                         amsgrad=False,
                                         fp32_optimizer_states=True)
        else:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.00,
                    "lr": 3.0 * lr_init
                },
            ]
            if args.weight_decay > 0:
                print("===load weight_decay===")
                optim_groups += [{"params": [param_dict[n] for n in lr_decay],
                                  "weight_decay": args.weight_decay,
                                  "my_lr_scale": 1.0}]
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(args.beta1, args.beta2),
                                         eps=args.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=args.adamw_mode,
                                         #fp32_optimizer_states=False,
                                         amsgrad=False)
        lr_scheduler = None
        if args.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * args.lr_init,
                warmup_max_lr=args.lr_init,
                warmup_num_steps=args.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler


    @property
    def deepspeed_offload(self) -> bool:
        return True

    def forward(self, batch, states=None):
        args = self.args
        # ================
        seq = batch['input_ids']
        mask = batch.get('attention_mask',None)
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x!=0) for x in idx]
        idx = torch.tensor([idx],dtype=torch.long).to('cuda')
        targets = torch.tensor([targets],dtype=torch.long).to('cuda')
        mask = torch.tensor([mask],dtype=torch.float32).to('cuda')

        # 前向 获得logits

        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)

        # 计算loss
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()

        if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # loss_raw = loss
            loss = torch.sum(loss * mask) / sum_mask

        # if torch.any(torch.isnan(loss)):
        #     print("\n=====error=======\n")
        #     loss = torch.where(torch.isnan(loss), torch.full_like(loss,1.0e-7), loss)

        # loss_raw = loss
        loss = L2Wrap.apply(loss, logits)
        return loss, states

    # def training_step(self, batch):
    #     args = self.args
    #     seq = batch['input_ids']
    #     mask = batch.get('attention_mask',None)
    #     idx = seq[:-1]
    #     targets = seq[1:]
    #     if mask == None:
    #         mask = [int(x!=0) for x in idx]
    #     idx = torch.tensor([idx],dtype=torch.long).to('cuda')
    #     targets = torch.tensor([targets],dtype=torch.long).to('cuda')
    #     mask = torch.tensor([mask],dtype=torch.float32).to('cuda')
    #     mask = mask.view(-1)
    #     sum_mask = torch.sum(mask).item()
    #     # 前向 获得ligts
    #     logits = self(idx)
    #     # 计算loss
    #     if sum_mask == mask.shape[0]:
    #             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    #             # print('rank', self.global_rank, 'loss', loss.item())
    #     else:
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    #         # loss_raw = loss
    #         loss = torch.sum(loss * mask) / sum_mask
    #     # if torch.any(torch.isnan(loss)):
    #     #     print("\n=====error=======\n")
    #     #     loss = torch.where(torch.isnan(loss), torch.full_like(loss,1.0e-7), loss)

    #     # loss_raw = loss

    #     loss = L2Wrap.apply(loss, logits)
    #     return loss
