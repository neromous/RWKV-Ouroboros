########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import functools
import sys
import os, math, gc, importlib
from config import config_args

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import numpy as np
import time
import types
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

os.environ['RWKV_MY_TESTING'] = config_args['environ']['RWKV_MY_TESTING']
LORA_CONFIG = config_args['lora_config']
LORA_CONFIG['parts'] = set(LORA_CONFIG['parts'])

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop

def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)


local_path = os.path.dirname(__file__)
########################################################################################################


class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, device, dtype):
        wkv_states = torch.empty((N, B, C, 3),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])

# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16",
                    sources=["cuda_state/wkv_op_state_bf16.cpp",
                             "cuda_state/wkv_cuda_state_bf16.cu"],
                    verbose=True,
                    extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage",
                                       "--maxrregcount 60", "--use_fast_math", "-O3",
                                       "-Xptxas -O3", "--extra-device-vectorization",
                                       f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        # @staticmethod
        # def init_state(B, C):
        #     state = torch.zeros((B, C, 3), device='cuda')
        #     state[:,:,2] = -1e38
        #     return state.cuda()
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v,last_state):
            # global_args.forward_wkv_count += 1
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            last_state = last_state.contiguous()
            y = torch.empty((B, T, C), device=w.device,
                            memory_format=torch.contiguous_format,
                            dtype=torch.bfloat16)
            new_state = torch.empty((B, C, 3),
                                    device=w.device,
                                    memory_format=torch.contiguous_format,
                                    dtype=torch.float32)
            wkv_cuda.forward(B, T, C, w, u, k, v, last_state, y,new_state)
            ctx.save_for_backward(w, u, k, v, y,last_state)
            return y,new_state

        @staticmethod
        def backward(ctx, gy,gnew_state):
            # global_args.backward_wkv_count += 1
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y, last_state = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format,
                             dtype=torch.bfloat16)
            glast_state = torch.empty((B, C, 3), device=w.device,
                                      memory_format=torch.contiguous_format,
                                      dtype=torch.float32)
            wkv_cuda.backward(B, T, C, w, u, k, v, last_state, y,
                              gy.contiguous(), gnew_state.contiguous(),
                              gw, gu, gk, gv, glast_state)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv, glast_state)

else:
    wkv_cuda = load(name=f"wkv_{T_MAX}",
                    sources=["cuda_state/wkv_op_state.cpp",
                             "cuda_state/wkv_cuda_state.cu"],
                    verbose=True,
                    extra_cuda_cflags=["-res-usage",
                                       "--maxrregcount 60",
                                       "--use_fast_math",
                                       "-O3", "-Xptxas -O3",
                                       "--extra-device-vectorization",
                                       f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v,last_state):
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
            last_state = last_state.contiguous()
            y = torch.empty((B, T, C),
                            device=w.device,
                            memory_format=torch.contiguous_format)
            new_state = torch.empty((B, C, 3),
                                    device=w.device,
                                    memory_format=torch.contiguous_format,
                                    dtype=torch.float32)
            wkv_cuda.forward(B, T, C, w, u, k, v, last_state, y,new_state)
            ctx.save_for_backward(w, u, k, v, y,last_state)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y,new_state
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half(),new_state
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16(),new_state
        @staticmethod
        def backward(ctx, gy,gnew_state):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y, last_state = ctx.saved_tensors
            gw = torch.empty((B, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format)
            gu = torch.empty((B, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C),
                             device=gy.device,
                             memory_format=torch.contiguous_format)
            glast_state = torch.empty((B, C, 3),
                                      device=w.device,
                                      memory_format=torch.contiguous_format,
                                      dtype=torch.float32)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v,last_state, y,
                                  gy.contiguous(),gnew_state.contiguous(),
                                  gw, gu, gk, gv, glast_state)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, last_state,y,
                                  gy.float().contiguous(),gnew_state.contiguous(),
                                  gw, gu, gk, gv, glast_state)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv, glast_state)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half(), glast_state)
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16(), glast_state)

def RUN_CUDA(B, T, C, w, u, k, v, last_state):
    return WKV.apply(B, T, C, w, u, k, v, last_state)


########################################################################################################
# LoRA
########################################################################################################
class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
            "alpha"], LORA_CONFIG["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


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

########################################################################################################

class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.ctx_len = args.ctx_len
        self.ctx_len = T_MAX
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
        if LORA_CONFIG['layers'] is None or layer_id in LORA_CONFIG["layers"]:
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False)
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False)
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: TimeMixState):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        y, new_wkv_state = RUN_CUDA(B, T, C, self.time_decay, self.time_first,
                                    k, v, last_state.wkv_state)
        return self.output(sr * y), TimeMixState(x[:, -1], new_wkv_state)

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        if LORA_CONFIG["layers"] is None or layer_id in LORA_CONFIG["layers"]:
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False)
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])



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

        self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, last_state: BlockState):
        args = self.args
        B, T, C = x.size()
        x_emb = x
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        if self.args.dropout == 0:
            x = x + att_out
        else:
            x = self.drop0(x + att_out)

        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )

        if self.args.dropout == 0:
            x = x + ffn_out
        else:
            x = self.drop1(x + ffn_out)

        return x, BlockState(att_state, ffn_state)

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
        if os.environ.get("WN_FIX_L2WRAP"):
            maxx[maxx<3.]=0.
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dtype == "fp32":
            args.dtype = torch.float32
        elif args.dtype == "fp16":
            args.dtype = torch.float16
        elif args.dtype == "bf16":
            args.dtype = torch.bfloat16
        # 加载model_weights
        model_weights = torch.load(args.load_model, map_location='cpu')
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

        args.n_layer = n_layer
        args.n_embd = n_embd
        args.vocab_size = vocab_size

        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4

        args = types.SimpleNamespace()


        self.args = args
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        model_weights = {k: v.to(dtype=args.dtype) for k, v
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
                    "weight_decay": 0.00,
                    "lr": 3.0 * lr_init
                },
            ]
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(args.beta1, args.beta2),
                                         eps=args.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=args.adamw_mode,
                                         weight_decay=args.weight_decay,
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
                                         betas=(args.beta1, args.beta2),
                                         eps=args.adam_eps,
                                         adamw_mode=args.adamw_mode,
                                         weight_decay=args.weight_decay,
                                         amsgrad=False,
                                         bias_correction=True)
        lr_scheduler = None
        if args.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * args.lr_init,
                warmup_max_lr=args.lr_init,
                warmup_num_steps=args.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler


    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):

        args = self.args
        B, T = idx.size()
        assert T <= T_MAX, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer,
                                          B,
                                          args.n_embd,
                                          x.device,
                                          x.dtype)

        if last_shift_states is None:
            cur_bs_list = BlockStateList.create(
                self.n_layer, B,
                self.n_embd,
                x.device,
                x.dtype
            )
        else:
            cur_bs_list = BlockStateList(last_shift_states, last_wkv_states)


        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if args.grad_cp:
                x, new_state = deepspeed_checkpoint(block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state


        x = self.ln_out(x)

        x = self.head(x)
        return x, new_states.shift_states, new_states.wkv_states


    def training_step(self, batch:dict,states = None, **kwargs):
        args = self.args
        seq = batch['input_ids']
        mask = batch.get('attention_mask',None)

        # data process
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x!=0) for x in idx]

        # data into tensor
        idx = torch.tensor([idx],dtype=torch.long).cuda()
        targets = torch.tensor([targets],dtype=torch.long).cuda()

        # process mask
        mask = torch.tensor([mask],dtype=torch.float32).to('cuda')
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()

        # idx, targets, *others = batch
        B, T = idx.shape
        C = args.n_embd

        if states is None:
            states = BlockStateList.create(args.n_layer,
                                           B,
                                           C,
                                           idx.device,
                                           self.emb.weight.dtype)

        prv_shift_states = states.shift_states
        prv_wkv_states = states.wkv_states
        logits, new_shift_states, new_wkv_states = self(idx, prv_shift_states, prv_wkv_states)
        states = BlockStateList(new_shift_states, new_wkv_states)

        if sum_mask == mask.shape[0]:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # loss_raw = loss
            loss = torch.sum(loss * mask)
            if sum_mask > 0:
                loss = loss/sum_mask
        return L2Wrap.apply(loss, logits), states
