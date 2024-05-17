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
import types
from .rwkv_inner import rwkv_inner


def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

MyModule = torch.nn.Module
def __nop(ob):
    return ob
MyFunction = __nop
MyStatic = __nop

class BlockState:

    def __init__(self, time_mix_state: tuple[torch.Tensor,torch.Tensor],
                 channel_mix_state: torch.Tensor):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states


    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        wkv_states = torch.empty((N, B, n_head, head_size, head_size),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            (self.shift_states[layer, 0], self.wkv_states[layer]),
            (self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state


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

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 24, precision:int = 64, max_ctx_len:int = 4096):
        super().__init__()

        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32
            self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, n_embd))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, n_embd))

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)

        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att, eps=(1e-5)*(self.head_size_divisor**2))

        # Preload the CUDA kernel if needed
        self.max_ctx_len = max_ctx_len

        self.chunk_len = chunk_len
        self.precision = precision

    def forward(self, x, last_state: tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor]]:

        shift_state_out = x[:,-1]

        # print("x.size(-2)", x.size(-2))
        # print("self.chunk_len", self.chunk_len)
        assert x.size(-2) % self.chunk_len == 0 or x.size(-2) == 1, "optimized nocuda rwkv requires data len supplied to be an exact multiple of the chunk len"

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        self.n_kv_head = self.n_head
        KVH = self.n_kv_head
        K = self.head_size
        V = K

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # Get the xk, xv, xr, xg, xw, and rkvg
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)
        xw = x + dxprev * (self.time_maa_w + mw)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(xg))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        # time_decay = self.time_decay.float() # (KVH,K)
        # time_first = self.time_faaaa.float() # (KVH,K)
        # if KVH < H:
        #     reps = H // KVH
        #     k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
        #     v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
        #     time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
        #     time_first = time_first.expand(reps, KVH, K).contiguous().view(H, K)

        w = self.time_decay.float().view(1,H,1,K)
        w = w + (torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).view(B, T, H, K).transpose(1, 2) # BHTK
        w = torch.exp(-torch.exp(w))

        u = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # Logits and state
        wkv_state = last_state[1].to(r.dtype)

        r = r.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        w = w.float().contiguous()
        u = u.float().contiguous()
        wkv_state = wkv_state.float().contiguous()
        x_logits, wkv_state = rwkv_inner(r, k, v, w, u, wkv_state, self.chunk_len)
        x_logits = x_logits.to(dtype=torch.bfloat16)
        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, (shift_state_out,wkv_state))
# Dependencies


class ChannelMix(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    # forwarding channel mix given the model weights and the input tokens and states.
    #
    # Given:
    # - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
    # - Last shift states of the various batches [batch_size, state_size]
    #
    # Returns a pair
    # - of output embedding of shape [batch_size, seq_len, embedding_size]
    # - and the last output state of shape [batch_size, state_size]
    def forward(self,
                x: torch.Tensor,
                last_state: torch.Tensor):
        xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        dxx = xx - x
        xk = x + dxx * self.time_maa_k
        xr = x + dxx * self.time_maa_r
        kv = self.value(torch.relu(self.key(xk)) ** 2)
        return torch.sigmoid(self.receptance(xr)) * kv, x[:, -1]


class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn, chunk_len=24):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level
        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p = dropout)
            self.drop1 = nn.Dropout(p = dropout)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )

        if self.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = x + ffn_out

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
        if True:
            maxx[maxx<3.]=0.
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)

        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self, args_in):
        super().__init__()

        args = types.SimpleNamespace()
        args.n_embd = args_in.model.n_embd
        args.n_layer = args_in.model.n_layer
        args.vocab_size = args_in.model.vocab_size
        args.dropout = args_in.trainer.dropout
        args.grad_cp = args_in.trainer.grad_cp
        args.lora_on = args_in.lora.lora_on
        args.ctx_len = args_in.model.ctx_len
        args.head_size = args_in.model.head_size
        args.head_size_divisor = args_in.model.head_size_divisor
        args.load_model = args_in.model.load_model
        args.lora = args_in.lora
        args.trainer = args_in.trainer
        args.ctx_len = args_in.model.ctx_len
        args.model = args_in.model
        args.chunk_len = args.model.chunk_len
        self.args = args
        if self.args.model.dtype == "fp32":
            self.args.model.dtype = torch.float
        elif self.args.model.dtype == "fp16":
            self.args.model.dtype = torch.half
        elif self.args.model.dtype == "bf16":
            self.args.model.dtype = torch.bfloat16


        # load weight
        model_weights = torch.load(args.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())

        # calc init layer
        if args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            args.n_layer = max_block_id + 1

        # calc n_embd
        if args.n_embd < 0:
            args.n_embd = model_weights['head.weight'].shape[1]

        # clac vocab_size
        if args.vocab_size < 0:
            args.vocab_size = model_weights['head.weight'].shape[0]

        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(i,
                                           args.n_layer,
                                           args.n_embd,
                                           args.n_head,
                                           args.head_size,
                                           args.dropout,
                                           args.dim_att,
                                           args.dim_ffn,
                                           args.chunk_len
                                           ) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # init dropout
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
        self.load_state_dict(model_weights, strict=False)
        del model_weights

        # if args.trainer.train_type == "state-tuning":
        #     # self.requires_grad_(False)
        #     for name, module in self.named_modules():
        #         for pname, param in module.named_parameters():
        #             if pname.endswith('.time_state') and pname.startswith('blocks.'):
        #                 print(pname)
        #                 param.requires_grad = True

        for p in self.parameters():
            p.data = p.data.to(dtype=self.args.model.dtype)

        gc.collect()
        torch.cuda.empty_cache()


    def forward(self, idx, states=None):
        args = self.args
        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)
        # ================
        # 前向 获得logits
        B, T = idx.size()
        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        new_states = BlockStateList.create(args.n_layer,
                                           B,
                                           args.n_embd,
                                           args.n_head,
                                           args.head_size,
                                           idx.device,
                                           self.emb.weight.dtype)
        if states is None:
            states = BlockStateList.create(args.n_layer,
                                           B,
                                           args.n_embd,
                                           args.n_head,
                                           args.head_size,
                                           idx.device,
                                           self.emb.weight.dtype)
        else:
            states = BlockStateList(states.shift_states, states.wkv_states)

        x = self.emb(idx)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]
            if args.grad_cp > 0:
                x, state = deepspeed_checkpoint(block, x, state)
            else:
                x, state = block(x, state)
            new_states[i] = state
        x = self.ln_out(x)
        logits = self.head(x)
        # states = states.detach().to('cpu')
        return logits, new_states

    def get_optim_groups(self):
        lr_init = self.args.trainer.lr_init
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if "time_mix" in n:
                lr_1x.add(n)
            elif "time_decay" in n:
                lr_2x.add(n)
            elif "time_faaaa" in n:
                lr_2x.add(n)
            else:
                lr_1x.add(n)
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}
        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "lr": 1.0 * lr_init,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "lr": 2.0 * lr_init,
            },
            {
                "params": [param_dict[n] for n in lr_3x],
                "weight_decay": 0.00,
                "lr": 3.0 * lr_init,
            },
        ]
        return optim_groups
