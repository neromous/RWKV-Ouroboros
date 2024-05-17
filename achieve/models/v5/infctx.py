########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import functools
import sys
import os, math, gc, importlib
from config import config
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

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)


local_path = os.path.dirname(__file__)
########################################################################################################

######state
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
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:] = 0
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.empty((N, B, H, C//H, C//H),
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

HEAD_SIZE = int(config['model']['head_size_a'])

# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


if  config['model']["dtype"] == "bf16":
    wkv5_cuda = load(name="wkv5_infctx",
                     sources=["cuda/v5/wkv5_op_state.cpp",
                              f"cuda/v5/wkv5_cuda_state.cu"],
                     verbose=True,
                     extra_cuda_cflags=["-res-usage", "--use_fast_math",
                                        "-O3", "-Xptxas -O3",
                                        "--extra-device-vectorization",
                                        f"-D_N_={HEAD_SIZE}"])

    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u, last_state):
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
                new_state = torch.zeros((B, H, HEAD_SIZE, HEAD_SIZE),
                                        dtype=torch.float,
                                        requires_grad=False, device=r.device).contiguous()
                y = torch.empty((B, T, C), device=r.device,
                                dtype=torch.bfloat16,
                                memory_format=torch.contiguous_format)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y, last_state, new_state)
                return y, new_state

        @staticmethod
        def backward(ctx, gy, _):
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
                                 memory_format=torch.contiguous_format)
                gk = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format)
                gv = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format)
                gw = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format)
                gu = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.bfloat16,
                                 memory_format=torch.contiguous_format)
                wkv5_cuda.backward(B, T, C, H, r, k, v,
                                   eew, ew, u,
                                   gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu, None)

elif config['model']["dtype"] == "fp32":
    wkv5_cuda = load(name="wkv5_infctx_fp",
                     sources=["cuda/v5/wkv5_op_state_fp.cpp",
                              f"cuda/v5/wkv5_cuda_state_fp.cu"],
                     verbose=True,
                     extra_cuda_cflags=["-res-usage", "--use_fast_math",
                                        "-O3", "-Xptxas -O3",
                                        "--extra-device-vectorization",
                                        f"-D_N_={HEAD_SIZE}"])

    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u, last_state):
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
                new_state = torch.zeros((B, H, HEAD_SIZE, HEAD_SIZE),
                                        dtype=torch.float,
                                        requires_grad=False, device=r.device).contiguous()
                y = torch.empty((B, T, C), device=r.device,
                                dtype=torch.float,
                                memory_format=torch.contiguous_format)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y, last_state, new_state)
                return y, new_state

        @staticmethod
        def backward(ctx, gy, _):
            with torch.no_grad():
                assert gy.dtype == torch.float
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format)
                gk = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format)
                gv = torch.empty((B, T, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format)
                gw = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format)
                gu = torch.empty((B, C),
                                 device=gy.device,
                                 requires_grad=False,
                                 dtype=torch.float,
                                 memory_format=torch.contiguous_format)
                wkv5_cuda.backward(B, T, C, H, r, k, v,
                                   eew, ew, u,
                                   gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu, None)





def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u, last_state):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u, last_state)



########################################################################################################
class RWKV_TimeMix(MyModule):
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
    def jit_func(self, x, shift_state):
        B, T, C = x.size()

        #xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g, x[:, -1]

    @MyFunction
    def jit_func_2(self, x, g, timemixstate:TimeMixState):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x, timemixstate

    def forward(self, x, last_state: TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        shift_state = last_state.shift_state
        r, k, v, g, lx = self.jit_func(x, shift_state)

        x, new_wkv_state = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa, last_state=last_state.wkv_state)

        return self.jit_func_2(x, g, TimeMixState(lx, new_wkv_state))


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
        if config['environ']["WN_FIX_L2WRAP"]:
            maxx[maxx<3.]=0.
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args
        if self.args.dtype == "fp32":
            self.args.dtype=torch.float
        elif self.args.dtype == "fp16":
            self.args.dtype=torch.half
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

        self.emb = nn.Embedding(self.args.vocab_size, self.args.n_embd)

        self.blocks = nn.ModuleList([Block(self.args, i) for i in range(self.args.n_layer)])

        self.ln_out = nn.LayerNorm(self.args.n_embd)
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
        model_weights = {k:v.to(dtype=self.args.dtype) for k,v
                         in model_weights.items()}
        # 加载至系统
        self.load_state_dict(model_weights)
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()


    def get_optimizers(self):
        lr_init= self.args.lr_init
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if "time_mix" in n:
                lr_1x.add(n)
            elif "time_decay" in n:
                lr_2x.add(n)
            elif "time_faaaa" in n:
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

        if self.args.dtype == torch.float32:
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
                                         betas=(self.args.beta1, self.args.beta2),
                                         eps=self.args.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=self.args.adamw_mode,
                                         weight_decay=self.args.weight_decay,
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
                                         betas=(self.args.beta1, self.args.beta2),
                                         eps=self.args.adam_eps,
                                         adamw_mode=self.args.adamw_mode,
                                         weight_decay=self.args.weight_decay,
                                         amsgrad=False,
                                         bias_correction=True)
        lr_scheduler = None
        if self.args.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.args.lr_init,
                warmup_max_lr=self.args.lr_init,
                warmup_num_steps=self.args.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler





    # def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
    #             last_wkv_states: torch.Tensor):

    def forward(self, batch:dict, states:BlockStateList=None):
        args = self.args
        # pre calc
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

        # 计算logits
        args = self.args

        B, T = idx.size()
        C = args.n_embd
        H =  args.dim_att // args.head_size_a

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C== H * args.head_size_a

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer,
                                          B,
                                          args.n_embd,
                                          H,
                                          x.device,
                                          x.dtype)

        if states is None:
            cur_bs_list = BlockStateList.create(
                self.args.n_layer,
                B,
                self.args.n_embd,
                H,
                x.device,
                x.dtype)
        else:
            cur_bs_list = BlockStateList(states.shift_states, states.wkv_states)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.args.grad_cp:
                x, new_state = deepspeed_checkpoint(block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        logits = self.head(x)

        #logits 计算完毕
        # states = BlockStateList(new_shift_states, new_wkv_states)

        if sum_mask == mask.shape[0]:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # loss_raw = loss
            loss = torch.sum(loss * mask)
            if sum_mask > 0:
                loss = loss/sum_mask
        return L2Wrap.apply(loss, logits), new_states
