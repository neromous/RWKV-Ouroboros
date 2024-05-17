########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import sys
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import numpy as np
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import copy
from types import SimpleNamespace

from .module.CoreDependencies import *
from .module.ChannelMix import RWKV_ChannelMix
from .module.TimeMix import RWKV_TimeMix


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop


def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)


local_path = os.path.dirname(__file__)
########################################################################################################
### ---
# RWKV: State Blocks
### ---


class BlockState:
    def __init__(
        self,
        time_mix_state: tuple[torch.Tensor, torch.Tensor],
        channel_mix_state: torch.Tensor,
    ):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    # @ TCompileMax (no difference)
    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        # @TODO: confirm if dtype can be changed from .flaot to dtype=dtype (when bf16)
        wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size),
            # wkv_states = torch.empty((N, B, 1, n_head, head_size, head_size),
            device=device,
            #  dtype=dtype)
            dtype=torch.float,
        )
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            (self.shift_states[layer, 0], self.wkv_states[layer]),
            (self.shift_states[layer, 1]),
        )

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

    # def get_clone_copy(self):
    #     cur_bs_list = {}
    #     if self.shift_states.is_leaf or self.wkv_states.is_leaf:
    #         print("===copy===")
    #         shift_states = self.shift_states.clone().detach()
    #         wkv_states = self.wkv_states.clone().detach()
    #         return BlockStateList(shift_states, wkv_states)
    #     else:
    #         self



### ---
# The RWKV Model blocks
### ---


class Block(nn.Module):
    def __init__(
        self, layer_id, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level
        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)

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
    def forward(ctx, loss, y, token_amount, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        #
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y, token_amount, currentMask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, token_amount, currentMask = ctx.saved_tensors

        # to encourage the logits to be close to 0
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)

        # We ensure the mask is reshaped accordingly, and apply it against gy
        gy = gy * currentMask.reshape(
            gy.shape[0], gy.shape[1], 1
        )  # currentMask[:, None][None, :]
        return (grad_output, gy, None, None)


########################################################################################################


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = SimpleNamespace()
        self.args.lr_init = args.trainer.lr_init
        self.args.dtype = args.model.dtype
        self.args.dropout= args.trainer.dropout
        self.args.head_size_a = args.model.head_size_a
        self.args.ctx_len =  args.trainer.ctx_len
        self.args.grad_cp = args.trainer.grad_cp
        if self.args.dtype == "fp32":
            self.args.dtype = torch.float
        elif self.args.dtype == "fp16":
            self.args.dtype = torch.half
        elif self.args.dtype == "bf16":
            self.args.dtype = torch.bfloat16

        model_weights = torch.load(args.model.load_model, map_location="cpu")
        model_keys = list(model_weights.keys())

        self.args.n_layer = args.model.n_layer
        if self.args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            self.args.n_layer = max_block_id + 1

        self.args.n_embd = args.model.n_embd
        if self.args.n_embd < 0:
            self.args.n_embd = model_weights["head.weight"].shape[1]

        self.args.vocab_size = args.model.vocab_size
        if self.args.vocab_size < 0:
            self.args.vocab_size = model_weights["head.weight"].shape[0]

        self.args.head_size = args.model.head_size
        self.args.dim_att = self.args.n_embd
        self.args.n_head = self.args.dim_att // self.args.head_size
        self.args.dim_ffn = int((self.args.n_embd * 3.5) // 32 * 32)
        if not hasattr(args, "tiny_att_layer"):
            self.args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            self.args.tiny_att_dim = -1

        self.emb = nn.Embedding(self.args.vocab_size, self.args.n_embd)

        self.blocks = nn.ModuleList(
            [
                Block(
                    i,
                    self.args.n_layer,
                    self.args.n_embd,
                    self.args.n_head,
                    self.args.head_size,
                    self.args.dropout,
                    self.args.dim_att,
                    self.args.dim_ffn,
                )
                for i in range(self.args.n_layer)
            ]
        )

        self.ln_out = nn.LayerNorm(self.args.n_embd)
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)

        # 加载至系统
        self.load_state_dict(model_weights)

        del model_weights

        for p in self.parameters():
            p.data = p.data.to(dtype=self.args.dtype)

        gc.collect()
        torch.cuda.empty_cache()

    def get_optimizers(self):
        lr_init = self.args.lr_init
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


    def forward(
        self,
        idx,
        states: BlockStateList = None,
    ):
        """
        前向传一个序列
        """
        # print(next(self.parameters()).device)

        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)

        # 计算logits
        args = self.args

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size_a

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * args.head_size_a

        x = self.emb(idx)
        new_states = BlockStateList.empty(
            args.n_layer, B, args.n_embd, args.n_head, args.head_size, x.device, x.dtype
        )

        if states is None:
            cur_bs_list = BlockStateList.create(
                args.n_layer,
                B,
                args.n_embd,
                args.n_head,
                args.head_size,
                x.device,
                x.dtype,
            )
        else:
            cur_bs_list = copy.deepcopy(states)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.args.grad_cp:
                x, new_state = deepspeed_checkpoint(block, x, last_state)
                # print(x.shape,x.device)
                # x, new_state = block(x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        logits = self.head(x)

        return logits, new_states
