########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
from torch import Tensor

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ""


def __nop(ob):
    return ob


# NOTE: These are main function to make fp16 training useable, but noone knnows it can learn efficienctly.

# here are some hp for clipping.
FP16_LIMIT = 65504
GRAD_LIMIT = 4
LARGE_GRAD_LIMIT = 1
RELAX_GRAD_LIMIT = 128


class sym_protector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_in):
        tensor_out = torch.nan_to_num(
            tensor_in, nan=0.0, posinf=FP16_LIMIT, neginf=-FP16_LIMIT
        )
        return tensor_out

    @staticmethod
    def backward(ctx, grad_tensor_in):
        grad_tensor_out = torch.nan_to_num(
            grad_tensor_in, nan=0.0, posinf=FP16_LIMIT, neginf=-FP16_LIMIT
        )
        grad_tensor_out = torch.clip(grad_tensor_out, -GRAD_LIMIT, GRAD_LIMIT)
        return grad_tensor_out


class large_tensor_sym_protector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_in):
        tensor_out = torch.nan_to_num(
            tensor_in, nan=0.0, posinf=FP16_LIMIT, neginf=-FP16_LIMIT
        )
        return tensor_out

    @staticmethod
    def backward(ctx, grad_tensor_in):
        grad_tensor_out = torch.nan_to_num(
            grad_tensor_in, nan=0.0, posinf=FP16_LIMIT, neginf=-FP16_LIMIT
        )
        grad_tensor_out = torch.clip(
            grad_tensor_out, -LARGE_GRAD_LIMIT, LARGE_GRAD_LIMIT
        )
        return grad_tensor_out


class relax_protector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_in):
        return tensor_in

    @staticmethod
    def backward(ctx, grad_tensor_in):
        grad_tensor_out = torch.nan_to_num(
            grad_tensor_in, nan=0.0, posinf=FP16_LIMIT, neginf=-FP16_LIMIT
        )
        grad_tensor_out = torch.clip(
            grad_tensor_out, -RELAX_GRAD_LIMIT, RELAX_GRAD_LIMIT
        )
        return grad_tensor_out


class FP16_LINEAR(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        weight = sym_protector.apply(self.weight)
        input = sym_protector.apply(input)
        return sym_protector.apply(F.linear(input, weight, self.bias))


class FP16_LARGETENSOR_LINEAR(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        weight = large_tensor_sym_protector.apply(self.weight)
        input = sym_protector.apply(input)
        return sym_protector.apply(F.linear(input, weight, self.bias))


class FP16_EMBEDDING(nn.Embedding):
    def forward(self, input: Tensor) -> Tensor:
        weight = large_tensor_sym_protector.apply(self.weight)
        input = sym_protector.apply(input)
        return sym_protector.apply(
            F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        )


class FP16_LAYERNORM(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        weight = sym_protector.apply(self.weight)
        input = sym_protector.apply(input)
        return sym_protector.apply(
            F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)
        )


class FP16_LINEAR_RELAXED(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        weight = relax_protector.apply(self.weight)
        input = relax_protector.apply(input)
        return relax_protector.apply(F.linear(input, weight, self.bias))


class FP16_LAYERNORM_RELAXED(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        weight = relax_protector.apply(self.weight)
        input = relax_protector.apply(input)
        return relax_protector.apply(
            F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)
        )


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(
        name=f"wkv_{T_MAX}_bf16",
        sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"],
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
            y = torch.empty(
                (B, T, C),
                device=w.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
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
            gw = torch.empty(
                (B, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gu = torch.empty(
                (B, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

else:
    wkv_cuda = load(
        name=f"wkv_{T_MAX}",
        sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
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
            y = torch.empty(
                (B, T, C), device=w.device, memory_format=torch.contiguous_format
            )
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
            gw = torch.empty(
                (B, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gu = torch.empty(
                (B, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gk = torch.empty(
                (B, T, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gv = torch.empty(
                (B, T, C), device=gy.device, memory_format=torch.contiguous_format
            )
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(
                    B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv
                )
            else:
                wkv_cuda.backward(
                    B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv
                )
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (
                    None,
                    None,
                    None,
                    gw.bfloat16(),
                    gu.bfloat16(),
                    gk.bfloat16(),
                    gv.bfloat16(),
                )


def RUN_CUDA(B, T, C, w, u, k, v):
    return sym_protector.apply(WKV.apply(B, T, C, w, u, k, v))


def RUN_CUDA_RELAXED(B, T, C, w, u, k, v):
    return relax_protector.apply(WKV.apply(B, T, C, w, u, k, v))


########################################################################################################

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
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
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(args.dim_att) * math.log(0.3) + zigzag
            )

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            )
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if layer_id == 0:
            self.key = FP16_LINEAR(args.n_embd, args.dim_att, bias=False)
            self.value = FP16_LINEAR(args.n_embd, args.dim_att, bias=False)
            self.receptance = FP16_LINEAR(args.n_embd, args.dim_att, bias=False)
            self.output = FP16_LINEAR(args.dim_att, args.n_embd, bias=False)

        else:
            self.key = FP16_LINEAR_RELAXED(args.n_embd, args.dim_att, bias=False)
            self.value = FP16_LINEAR_RELAXED(args.n_embd, args.dim_att, bias=False)
            self.receptance = FP16_LINEAR_RELAXED(args.n_embd, args.dim_att, bias=False)
            self.output = FP16_LINEAR_RELAXED(args.dim_att, args.n_embd, bias=False)

    if "a" not in os.environ["RWKV_MY_TESTING"]:

        @MyFunction
        def jit_func(self, x):
            xx = self.time_shift(
                x
            )  # Mix x with the previous timestep to produce xk, xv, xr
            if self.layer_id == 0:
                tmk, tmv, tmr = (
                    sym_protector.apply(self.time_mix_k),
                    sym_protector.apply(self.time_mix_v),
                    sym_protector.apply(self.time_mix_r),
                )
            else:
                tmk, tmv, tmr = (
                    relax_protector.apply(self.time_mix_k),
                    relax_protector.apply(self.time_mix_v),
                    relax_protector.apply(self.time_mix_r),
                )
            xk = x * tmk + xx * (1 - tmk)
            xv = x * tmv + xx * (1 - tmv)
            xr = x * tmr + xx * (1 - tmr)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            return sr, k, v

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v = self.jit_func(x)
            if self.layer_id == 0:
                rwkv = sr * RUN_CUDA(
                    B, T, self.args.dim_att, self.time_decay, self.time_first, k, v
                )
            else:
                rwkv = sr * RUN_CUDA_RELAXED(
                    B, T, self.args.dim_att, self.time_decay, self.time_first, k, v
                )
            return self.output(rwkv)


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
        if layer_id == 0:
            self.key = FP16_LINEAR(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = FP16_LINEAR(args.n_embd, args.n_embd, bias=False)
            self.value = FP16_LINEAR(args.dim_ffn, args.n_embd, bias=False)
        else:
            self.key = FP16_LINEAR_RELAXED(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = FP16_LINEAR_RELAXED(args.n_embd, args.n_embd, bias=False)
            self.value = FP16_LINEAR_RELAXED(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        tmk, tmr = sym_protector.apply(self.time_mix_k), sym_protector.apply(
            self.time_mix_r
        )
        xk = x * tmk + xx * (1 - tmk)
        xr = x * tmr + xx * (1 - tmr)
        k = self.key(xk)
        if self.layer_id == 0:
            k = torch.clip(k, -255.0, 255.0)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return sym_protector.apply(torch.sigmoid(self.receptance(xr)) * kv)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        #if layer_id == 0:
        self.ln1 = FP16_LAYERNORM(args.n_embd)
        self.ln2 = FP16_LAYERNORM(args.n_embd)
        # else:
        #     self.ln1 = FP16_LINEAR_RELAXED(args.n_embd)
        #     self.ln2 = FP16_LINEAR_RELAXED(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = FP16_LAYERNORM(args.n_embd)
            if args.my_pos_emb > 0:
                NotImplementedError
        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if "r" in os.environ["RWKV_MY_TESTING"]:
                NotImplementedError
            else:
                self.att = RWKV_TimeMix(args, layer_id)

        if "g" in os.environ["RWKV_MY_TESTING"]:
            NotImplementedError
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            NotImplementedError

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            NotImplementedError
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


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = FP16_EMBEDDING(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = FP16_LAYERNORM(args.n_embd)
        self.head = FP16_LARGETENSOR_LINEAR(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            NotImplementedError

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if ("time_mix" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_decay" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [{"params": [param_dict[n] for n in lr_1x],
                         #"weight_decay": 0.0,
                         #"my_lr_scale": 1.0
                         }]
        return optim_groups


    def forward(self, batch):
        args = self.args
        layer_logits = []
        #idx, targets, mask = x,y,z
        idx, targets, mask = batch
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()
        # logits = self(idx)
        #-------- 计算 idx 到logits --------
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
            #with torch.no_grad():
            #   layer_logits.append(self.head(self.ln_out(x.detach().cpu().float())))
        x = self.ln_out(x)

        logits = self.head(x)

        #-------- 计算loss  --------
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), reduction='none')
        # loss_raw = loss
        loss = torch.sum(loss * mask) / sum_mask
        loss = L2Wrap.apply(loss, logits)
        return loss


        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x
