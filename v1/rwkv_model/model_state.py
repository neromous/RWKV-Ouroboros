########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import functools
import sys
import os, math, gc, importlib
from utils import log, load_config
config = load_config()
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import numpy as np
import time
import types
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


os.environ['RWKV_MY_TESTING'] = config['environ']['RWKV_MY_TESTING']

LORA_CONFIG = config['lora_config']
LORA_CONFIG['parts'] = set(LORA_CONFIG['parts'])

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop


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

T_MAX =  int(os.environ["RWKV_T_MAX"])

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
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # 这个函数会不会影响batch和grad_accu的一致性？
        # 感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        if ctx.token_amount == 0:
            return (grad_output, None, None)
        factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
            # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)


class RWKV(nn.Module):
    def __init__(self,
                 load_model:str,
                 n_layer=-1,
                 n_embd=-1,
                 vocab_size=-1,
                 ctx_len=2048,
                 grad_cp=1,
                 weight_decay=0,
                 pre_ffn=0,
                 lr_init=1.0e-5,
                 lr_final=1e-5,
                 adam_eps=1.0e-7,
                 beta1=0.9,
                 beta2=0.999,
                 warmup_steps=8,
                 adamw_mode=False,
                 dtype="fp32",
                 head_qk=0,
                 vacab_size=-1,
                 accelerator="gpu",
                 devices=1,
                 precision="bf16",
                 accumulate_grad_batches=1,
                 strategy="",
                 lora=False,
                 # lora_r=16,
                 # lora_alpha=32,
                 # lora_dropout=0.01,
                 # lora_parts="ffn",
                 ctx_parts=None,
                 my_pos_emb=0,
                 tiny_att_dim=0,
                 tiny_att_layer=-999):
        super().__init__()
        if dtype == "fp32":
            self.dtype=torch.float32
        elif dtype == "fp16":
            self.dtype=torch.float16
        elif dtype== "bf16":
            self.dtype = torch.bfloat16
        self.load_model = load_model
        # 加载model_weights
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.lora = lora
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
        args.lora = lora
        args.warmup_steps = self.warmup_steps
        args.dim_att = self.dim_att
        args.dim_ffn = self.dim_ffn
        self.args = args
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        model_weights = {k:v.to(dtype=self.dtype) for k,v
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
                                         asbias_correction=True)
        lr_scheduler = None
        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler





    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):

        args = self.args
        B, T = idx.size()
        assert T <= T_MAX, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd,
            x.device, x.dtype)
        for i,(block,block_state) in enumerate(zip(self.blocks,
            BlockStateList(last_shift_states, last_wkv_states))):
            # x = x.to(block.device)
            if args.grad_cp == 1 and i>0 : #and i < len(self.blocks)-1
                if args.lora:
                    x, new_block_state = torch_checkpoint(block, x, block_state,use_reentrant=False)
                else:
                    x, new_block_state = torch_checkpoint(block, x, block_state,use_reentrant=False)
            else:
                x, new_block_state = block(x, block_state)
            new_states[i] = new_block_state
        # x = x.to(self.ln_out.device)

        x = self.ln_out(x)

        x = self.head(x)
        return x,new_states.shift_states, new_states.wkv_states


    def training_step(self, batch:dict, **kwargs):
        args = self.args
        seq = batch['input_ids']
        masks = batch.get('attention_mask',None)
        idx = seq[:-1]
        targets = seq[1:]
        idx = torch.tensor([idx],dtype=torch.long).cuda()
        targets = torch.tensor([targets],dtype=torch.long).cuda()
        # idx, targets, *others = batch
        B, T = idx.shape
        C = args.n_embd

        states = BlockStateList.create(args.n_layer, B, C, idx.device,
            self.emb.weight.dtype)
        # init_states = states
        # init_states.shift_states.requires_grad_()
        # init_states.wkv_states.requires_grad_()
        def checkpointed_step(idx, targets, prev_loss, last_shift_states,
                              last_wkv_states, prev_token_amount):
            logits, new_shift_states, new_wkv_states = self(idx, last_shift_states, last_wkv_states)
            current_token_amount = (targets!=-100).sum() #这样是不是更合适？
            # current_token_amount = idx.shape[1]
            if current_token_amount == 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1),reduction='sum')
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1))
                loss = L2Wrap.apply(loss, logits, current_token_amount)
            new_token_amount = prev_token_amount+current_token_amount
            if new_token_amount>0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                    current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss
            return new_loss, new_shift_states, new_wkv_states, new_token_amount

        total_loss = torch.tensor(0.,dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        # i = 0
        #Blealtan的做法是ctx_len定义为cuda核的大小（对应我们这里的T_max），然后引入ctx_len_cutoff作为控制状态重置的长度
        #然后T都是样本长度
        #我感觉类似ctx_len_cutoff以后还是用额外的输入来标记每个序列的重置点，而不是模型内部规定一个重置点。
        #所以这里就不改成Blealtan的思路了，不过稍后可以在他的基础上rebase。他的代码更简洁一些
        i = 0
        for i in range(math.ceil(T / T_MAX)-1):
            # pdb.set_trace()
            # total_loss, states, token_amount = deepspeed.checkpointing.checkpoint(
            total_loss,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_MAX:(i + 1) * T_MAX],
                targets[:, i * T_MAX:(i + 1) * T_MAX],
                total_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                # use_reentrant=False
            )
            states = BlockStateList(new_shift_states, new_wkv_states)
            # if total_loss.isnan().all():
            #     import transformers
            #     tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
            #     pdb.set_trace()
        # pdb.set_trace()
        total_loss, new_shift_states, new_wkv_states, token_amount = checkpointed_step(
            idx[:, i * T_MAX:(i + 1) * T_MAX],
            targets[:, i * T_MAX:(i + 1) * T_MAX],
            total_loss,
            states.shift_states,
            states.wkv_states,
            token_amount
        )
        # pdb.set_trace()
        return total_loss


    # def inference_step(self, token):
    #     args = self.args

    #     idx, targets, *others = batch
    #     B, T = idx.shape
    #     C = args.n_embd

    #     states = BlockStateList.create(args.n_layer, B, C, idx.device,
    #         self.emb.weight.dtype)
    #     def checkpointed_step(idx, last_shift_states,
    #                           last_wkv_states,
    #                           prev_token_amount):
    #         logits, new_shift_states, new_wkv_states = self(idx,
    #                                                         last_shift_states,
    #                                                         last_wkv_states)
    #         return logits, new_shift_states, new_wkv_states


        token_amount = 0
        i = 0
        for i in range(math.ceil(T / T_MAX)-1):
            total_loss,new_shift_states, new_wkv_states = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_MAX:(i + 1) * T_MAX],
                states.shift_states,
                states.wkv_states)
            states = BlockStateList(new_shift_states, new_wkv_states)



        return logits




    @classmethod
    def sample_logits(cls,
                      logits:torch.tensor,
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
