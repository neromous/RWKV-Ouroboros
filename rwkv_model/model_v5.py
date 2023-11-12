########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import sys
import os, math, gc, importlib
import torch

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# import pytorch_lightning as pl
# from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
# from pytorch_lightning.strategies import DeepSpeedStrategy
import time
import types
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam



def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

local_path = os.path.dirname(__file__)
########################################################################################################
# CUDA Kernel
########################################################################################################

# T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
T_MAX = 2048
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp",
                                       f"cuda/wkv5_cuda.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                                                 "-O3", "-Xptxas -O3",
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
            y = torch.empty((B, T, C), device=r.device,
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
            gr = torch.empty((B, T, C), device=gy.device,
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

def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)


################################################################
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

# class RWKV_ChannelMix(MyModule):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args

#         self.layer_id = layer_id
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

#         with torch.no_grad():  # fancy init of time_mix
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#             self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
#             self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

#         self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
#         self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
#         self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x)
#         xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
#         xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
#         k = self.key(xk)
#         k = torch.square(torch.relu(k))
#         kv = self.value(k)
#         return torch.sigmoid(self.receptance(xr)) * kv


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

        self.att = RWKV_TimeMix_RWKV5(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
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
    def __init__(self,
                 load_model:str,
                 n_layer = -1,
                 n_embd = -1,
                 vocab_size = -1,
                 ctx_len = 2048,
                 grad_cp = 1,
                 weight_decay = 0,
                 pre_ffn = 0,
                 lr_init = 1.0e-5,
                 lr_final = 1.0e-6,
                 adam_eps = 1.0e-7,
                 beta1=0.9,
                 beta2=0.999,
                 warmup_steps = 8,
                 adamw_mode=False,
                 dtype="fp32",
                 dropout=0,
                 lora=False,
                 loaded=True):
        super().__init__()
        if dtype == "fp32":
            self.my_datatype=torch.float
        elif dtype == "fp16":
            self.my_datatype=torch.half
        elif dtype== "bf16":
            self.my_datatype = torch.bfloat16
        self.load_model = load_model
        #加载model_weights
        model_weights = torch.load(self.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())

        # Lets compute the model various sizes, if they are not provided
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            n_layer = max_block_id + 1

        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]

        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]

        self.adamw_mode =adamw_mode
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.lr_init = lr_init
        self.beta1= beta1
        self.beta2=beta2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        self.warmup_steps= warmup_steps
        self.grad_cp = grad_cp
        self.pre_ffn = pre_ffn
        self.dim_att = self.n_embd
        self.dim_ffn = self.n_embd * 4 - 64 * 20

        args = types.SimpleNamespace()
        args.n_layer = self.n_layer
        args.n_embd = self.n_embd
        args.vocab_size = self.vocab_size
        args.ctx_len = self.ctx_len
        args.grad_cp = self.grad_cp
        args.weight_decay = self.weight_decay
        args.pre_ffn = self.pre_ffn
        args.lr_init = self.lr_init
        args.adam_eps = self.adam_eps
        args.beta1=self.beta1
        args.beta2=self.beta2
        args.warmup_steps = self.warmup_steps
        args.dim_att = self.dim_att
        args.dim_ffn = self.dim_ffn
        args.dropout = dropout
        args.head_size_a = 64
        args.head_size_divisor = 8

        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        model_weights = {k:v.to('cuda',dtype=self.my_datatype) for k,v
                         in model_weights.items()}
        # 加载至系统
        self.load_state_dict(model_weights)
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()


    def get_optimizers(self):
        lr_init= self.lr_init
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
            else:
                lr_1x.add(n)
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}

        if self.my_datatype == torch.float32:
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
                    "weight_decay": 0.00,
                    "lr": 3.0 * lr_init
                },
            ]
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=self.adamw_mode,
                                         weight_decay=self.weight_decay,
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

            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=self.adamw_mode,
                                         weight_decay=self.weight_decay,
                                         fp32_optimizer_states=False,
                                         amsgrad=False)
        lr_scheduler = None
        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler

    def forward(self, idx):
        # -------- 计算 idx 到logits --------
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x
        for block in self.blocks:
            if self.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)

        # -------- 计算loss  --------
        return logits

    def inference(self, tokens):
        with torch.no_grad():
            idx = [x for x in tokens]
            #print(f'=======\n{idx}')
            #idx = [0 for x in range(0,self.ctx_len)]
            #idx[:len(tokens)] =  tokens
            idx = idx[:self.ctx_len - 1]

            idx = torch.tensor([idx],dtype=torch.long).to('cuda')
            # -------- 计算 idx 到logits --------
            B, T = idx.size()
            assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."
            x = self.emb(idx)

            for block in self.blocks:
                x = block(x)

            x = self.ln_out(x)

            x = self.head(x)

            x = x.view(-1, x.size(-1))

            # -------- 计算loss  --------
            #gc.collect()
            #torch.cuda.empty_cache()
            return x


    def training_step(self, batch:dict, mask=None,**kwargs):
        seq = batch['input_ids']
        mask = batch.get('attention_mask',None)
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x!=0) for x in idx]
        idx = torch.tensor([idx],dtype=torch.long).to('cuda')
        targets = torch.tensor([targets],dtype=torch.long).to('cuda')
        mask = torch.tensor([mask],dtype=torch.float32).to('cuda')
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()
        # 前向 获得ligts
        logits = self(idx)
        # 计算loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction="none")
        # if torch.any(torch.isnan(loss)):
        #     print("\n=====error=======\n")
        #     loss = torch.where(torch.isnan(loss), torch.full_like(loss,1.0e-7), loss)

        # loss_raw = loss
        loss = torch.sum(loss * mask) / sum_mask
        loss = L2Wrap.apply(loss, logits)
        return loss
