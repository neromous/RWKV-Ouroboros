from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import gc
import deepspeed
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import copy
from .rwkv_inner import rwkv_inner

def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

MyModule = torch.nn.Module
def __nop(ob):
    return ob
MyFunction = __nop
MyStatic = __nop



def rwkv_recurrent(r_in, k_in, v_in, w_in, u, state):
    L = r_in.size(-2)
    out = []
    for t in range(L):
        r, k, v, w = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        kv = k.mT @ v # KV
        out.append( r @ (state + u.mT * kv) ) # 1K @ (KV + 1)
        state = (w.mT * state) + kv # KV
    out = torch.cat(out, dim=-2)
    return out, state

class TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        # init
        self.n_kv_head = args.n_kv_head
        self.r_head_size = args.dim_rk // args.n_head
        self.k_head_size = args.dim_rk // args.n_head
        self.v_head_size = args.dim_v // args.n_head
        #
        assert args.dim_rk % self.n_head == 0
        assert args.dim_rk % self.n_kv_head == 0
        assert args.dim_v % self.n_kv_head == 0

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


    def forward(self, xq: Tensor, state: Optional[Tensor] = None, for_infer=False):
        x = xq # FIXME - support encoder-decoder models

        H = self.n_head
        KVH = self.n_kv_head
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce kx, vx, rx, gx
        kx = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        vx = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        rx = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        gx = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        # Mix kx, vx with the previous timestep in a learned manner to produce new time-mixed xk, xv
        #kx = self.time_mixer_k(kx)
        #vx = self.time_mixer_v(vx)
        #rx = gx = x

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(gx))

        #r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH,K)
        time_faaaa = self.time_faaaa.float() # (KVH,K)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
            time_faaaa = time_faaaa.expand(reps, KVH, K).contiguous().view(H, K)

        state = state
        if state is None:
            state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if state.dtype != r.dtype:
            state = state.contiguous()

        w = torch.exp(-torch.exp(time_decay)).view(1,H,1,K).expand(1,H,T,K)
        u = time_faaaa.float().view(1,H,1,K)

        r = r.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        w = w.float().contiguous()
        u = u.float().contiguous()
        state = state.float().contiguous()

        if for_infer:
            out, state = rwkv_recurrent(r, k, v, w, u, state)
        else:
            out, state = rwkv_inner(r, k, v, w, u, state)
        # r = r.to(dtype=self.args.dtype)
        # k = k.to(dtype=self.args.dtype)
        # v = v.to(dtype=self.args.dtype)
        # w = w.to(dtype=self.args.dtype)
        # u = u.to(dtype=self.args.dtype)
        #state = state.to(dtype=self.args.dtype)
        out = out.to(dtype=self.args.dtype)
        out = out.transpose(1,2).reshape(B*T, H*V)
        out = self.ln_x(out / self.args.head_size_divisor).view(B, T, H*V)
        out = self.output(out * g)
        return out, state


class ChannelMix(nn.Module):
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

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)
        self.ffn = ChannelMix(args, layer_id)

    def forward(self, x, state, for_infer):
        if self.layer_id == 0:
            x = self.ln0(x)
        out, state = self.att(self.ln1(x), state=state, for_infer = for_infer)
        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x, state



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
        if True:
            maxx[maxx<3.]=0.
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)

        # gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dtype == "fp32":
            args.dtype=torch.float
        elif args.dtype == "fp16":
            args.dtype=torch.half
        elif args.dtype== "bf16":
            args.dtype = torch.bfloat16
        else:
            args.dtype = torch.bfloat16
        # load model
        model_weights = torch.load(args.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())
        # init layer num
        max_block_id = 0
        for x in model_keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        args.n_layer = max_block_id + 1
        # init n_embd number
        args.n_embd = model_weights['head.weight'].shape[1]
        # init vocab number
        args.vocab_size = model_weights['head.weight'].shape[0]
        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        #
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
        args.tiny_att_layer = -1
        args.tiny_att_dim = -1
        self.args = args
        # init
        # args.n_kv_head = args.n_head
        # args.dim_rk = args.n_embd
        # args.dim_v = args.n_embd
        # init

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i)
                                     for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        model_weights = {k: v.to(dtype=args.dtype) for k, v
                         in model_weights.items()}

        self.load_state_dict(model_weights)
        self.to("cuda")
        print("=====", args.dtype)
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, batch, states=None):
        args = self.args
        if states is None:
            pass
        else:
            state = states.to('cuda')
        # ================
        seq = batch['input_ids']
        mask = batch.get('attention_mask',None)
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x!=0) for x in idx]
        idx = torch.tensor([idx],dtype=torch.long).to('cuda')
        targets = torch.tensor([targets],dtype=torch.long).to('cuda')
        mask = torch.tensor([mask],dtype=torch.float).to('cuda')

        # 前向 获得logits

        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        for block in self.blocks:
            if self.args.grad_cp:
                x, states = deepspeed_checkpoint(block, x, states, False)
            else:
                x, states = block(x, states, False)
        x = self.ln_out(x)
        logits = self.head(x)

        states = states.detach().to('cpu')
        return loss, states
