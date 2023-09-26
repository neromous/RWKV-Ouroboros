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

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ""


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

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(
        name=f"wkv_{T_MAX}_bf16",
        sources=[f"cuda_origin/wkv_op_bf16.cpp",
                 f"cuda_origin/wkv_cuda_bf16.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-t 4",
            "-std=c++17",
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DTmax={T_MAX}",
        ],
    )

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = torch.empty((B, T, C),
                            device=w.device,
                            memory_format=torch.contiguous_format,
                            dtype=torch.bfloat16)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gu = torch.empty((B, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gk = torch.empty((B, T, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gv = torch.empty((B, T, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y,
                              gy.contiguous(),
                              gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

else:
    wkv_cuda = load(
        name=f"wkv_{T_MAX}",
        sources=[f"cuda_origin/wkv_op.cpp", f"cuda_origin/wkv_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DTmax={T_MAX}",
        ],
    )

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16()

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


########################################################################################################


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if "a" in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)

    if "a" not in os.environ["RWKV_MY_TESTING"]:

        @MyFunction
        def jit_func(self, x):
            xx = self.time_shift(x)  # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            return sr, k, v

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v = self.jit_func(x)
            rwkv = sr * RUN_CUDA(B,
                                 T,
                                 self.args.dim_att,
                                 self.time_decay,
                                 self.time_first,
                                 k,
                                 v)
            return self.output(rwkv)

    if "a" in os.environ["RWKV_MY_TESTING"]:

        @MyFunction
        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            x = att @ v
            return x

        @MyFunction
        def jit_funcQKV(self, x):
            xx = self.time_shift(x)  # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)
            return sr, k, v, qq, kk, vv

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
            return rwkv


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
        k = torch.square(torch.relu(k))
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

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
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
                 n_layer = 32,
                 n_embd = 2560,
                 vocab_size = 65536,
                 ctx_len = 2048,
                 grad_cp = 1,
                 weight_decay = 0,
                 pre_ffn = 0,
                 lr_init = 1.0e-5,
                 adam_eps = 1.0e-7,
                 beta1=0.9,
                 beta2=0.999,
                 warmup_steps = 8,
                 adamw_mode=False,
                 dtype="fp16"):
        super().__init__()
        if dtype == "fp32":
            self.dtype=torch.float32
        elif dtype == "fp16":
            self.dtype=torch.float16
        elif dtype== "bf16":
            self.dtype = torch.bfloat16
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
        self.load_model = load_model
        self.warmup_steps= warmup_steps
        self.grad_cp = grad_cp
        self.pre_ffn = pre_ffn
        self.dim_att = self.n_embd
        self.dim_ffn = self.n_embd * 4

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

        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        model_weights = torch.load(self.load_model, map_location='cpu')
        model_weights = {k:v.to(dtype=self.dtype) for k,v
                         in model_weights.items()}
        self.load_state_dict(model_weights)
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()


    def configure_optimizers(self):
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

        if self.dtype == torch.float32:
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
        print("===dtype in forward==", x.dtype)
        for block in self.blocks:
            if self.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)

        # -------- 计算loss  --------
        return logits

    def training_step(self, data,mask=None):
        idx = data[:-1]
        targets = data[1:]
        if mask == None:
            mask = [int(x!=0) for x in targets]
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

    def inference(self, tokens):
        with torch.no_grad():
            idx = [x for x in tokens]
            #idx = [0 for x in range(0,self.ctx_len)]
            #idx[:len(tokens)] =  tokens
            idx = torch.tensor([idx],dtype=torch.long).to('cuda')
            # -------- 计算 idx 到logits --------
            B, T = idx.size()
            assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."
            x = self.emb(idx)
            x_emb = x
            for block in self.blocks:
                x = block(x)
            x = self.ln_out(x)
            logits = self.head(x)
            output = logits.view(-1, logits.size(-1))
        # -------- 计算loss  --------
        #gc.collect()
        #torch.cuda.empty_cache()
        return output


    @classmethod
    def sample_logits(cls, logits:torch.tensor,
                      temperature=0.1,
                      top_p=0.1, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
