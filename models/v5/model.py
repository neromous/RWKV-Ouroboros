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
import copy

from .module.CoreDependencies import *
from .module.ChannelMix import RWKV_ChannelMix
from .module.TimeMix import RWKV_TimeMix


os.environ["RWKV_MY_TESTING"] = config["environ"]["RWKV_MY_TESTING"]


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

        self.args = args
        if self.args.dtype == "fp32":
            self.args.dtype = torch.float
        elif self.args.dtype == "fp16":
            self.args.dtype = torch.half
        elif self.args.dtype == "bf16":
            self.args.dtype = torch.bfloat16

        model_weights = torch.load(self.args.load_model, map_location="cpu")
        model_keys = list(model_weights.keys())

        if self.args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            self.args.n_layer = max_block_id + 1

        if self.args.n_embd < 0:
            self.args.n_embd = model_weights["head.weight"].shape[1]

        if self.args.vocab_size < 0:
            self.args.vocab_size = model_weights["head.weight"].shape[0]

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
                    args.n_layer,
                    args.n_embd,
                    args.n_head,
                    args.head_size,
                    args.dropout,
                    args.dim_att,
                    args.dim_ffn,
                )
                for i in range(self.args.n_layer)
            ]
        )

        self.ln_out = nn.LayerNorm(self.args.n_embd)
        self.head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
        model_weights = {
            k: v.to(dtype=self.args.dtype) for k, v in model_weights.items()
        }
        # 加载至系统
        self.load_state_dict(model_weights)
        del model_weights
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
                    "lr": 1.0 * lr_init,
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init,
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.00,
                    "lr": 3.0 * lr_init,
                },
            ]
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=lr_init,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.adam_eps,
                bias_correction=True,
                adamw_mode=self.args.adamw_mode,
                weight_decay=self.args.weight_decay,
                amsgrad=False,
                fp32_optimizer_states=True,
            )
        else:
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

            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=lr_init,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.adam_eps,
                adamw_mode=self.args.adamw_mode,
                weight_decay=self.args.weight_decay,
                amsgrad=False,
                bias_correction=True,
            )
        lr_scheduler = None
        if self.args.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.args.lr_init,
                warmup_max_lr=self.args.lr_init,
                warmup_num_steps=self.args.warmup_steps,
                warmup_type="linear",
            )
        return optimizer, lr_scheduler

    # def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
    #             last_wkv_states: torch.Tensor):

    def forward(self, batch: dict, states: BlockStateList = None):
        args = self.args
        # pre calc
        seq = batch["input_ids"]
        mask = batch.get("attention_mask", None)

        # data process
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x != 0) for x in idx]

        # data into tensor
        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)
        targets = torch.tensor([targets], dtype=torch.long).to(
            next(self.parameters()).device
        )

        # process mask
        mask = torch.tensor([mask], dtype=torch.float32).to(
            next(self.parameters()).device
        )
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()

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
            cur_bs_list = BlockStateList(states.shift_states, states.wkv_states)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.args.grad_cp:
                x, new_state = deepspeed_checkpoint(block, x, last_state)
                # x, new_state = block(x, last_state)

            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        logits = self.head(x)
        # logits 计算完毕
        # states = BlockStateList(new_shift_states, new_wkv_states)

        if sum_mask == mask.shape[0]:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            # loss_raw = loss
            loss = torch.sum(loss * mask)
            if sum_mask > 0:
                loss = loss / sum_mask
        return L2Wrap.apply(loss, logits, torch.sum(mask), mask), new_states

    def forward_no_supervise(
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
                # x, new_state = block(x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        logits = self.head(x)

        return logits, new_states

    def speak(
        self,
        start_with_token: int,
        states,
        temperature=None,
        top_p=None,
        token_stop=[65535],
        max_tokens=100,
        m_postfix_token=[],
        debug_mode=False,
    ):
        with torch.no_grad():
            speak_sequence = []
            next_token = start_with_token
            args = self.args

            temperature = (
                config["inference"]["prompt_config"]["temperature"]
                if temperature is None
                else temperature
            )
            top_p = (
                config["inference"]["prompt_config"]["top_p"]
                if top_p is None
                else top_p
            )

            B, T = 1, 1
            C = args.n_embd
            H = args.dim_att // args.head_size_a
            assert C == H * args.head_size_a

            if states is None:
                new_states = BlockStateList.create(
                    args.n_layer,
                    B,
                    args.n_embd,
                    args.n_head,
                    args.head_size,
                    x.device,
                    x.dtype,
                )
            else:
                new_states = copy.deepcopy(states)

            for i in range(max_tokens):
                idx = torch.tensor([[next_token]], dtype=torch.long).to(
                    next(self.parameters()).device
                )
                # print(idx.size())
                x = self.emb(idx)
                new_states, logits = self.infer_sequence(x, new_states)

                # logits[0] = -9999999999

                next_token = self.sample_logits(
                    logits[:, -1, :].squeeze(), temperature, top_p
                )

                speak_sequence.append(next_token)

                if debug_mode:
                    xxx = config["inference"]["tokenizer"].decode([next_token])
                    print(xxx, end="")
                    sys.stdout.flush()

                if next_token in token_stop:
                    if m_postfix_token:
                        postfix = torch.tensor([[m_postfix_token]], dtype=torch.long).to(
                            next(self.parameters()).device
                        )
                        new_states, logits = self.infer_sequence(
                            self.emb(postfix), new_states
                        )
                    speak_sequence.pop()
                    break
        return speak_sequence, new_states

    def infer_sequence(self, x, states: BlockStateList):
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = states[i]
            x, new_state = block(x, last_state)
            states[i] = new_state

        x = self.ln_out(x)

        logits = self.head(x)
        return states, logits

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top_k = int(top_k)
            # 'privateuseone' is the type of custom devices like `torch_directml.device()`
            if probs.device.type in ["cpu", "privateuseone"]:
                probs = probs.cpu().numpy()
                sorted_ids = np.argsort(probs)
                sorted_probs = probs[sorted_ids][::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
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
                cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
                probs[probs < cutoff] = 0
                if top_k < len(probs) and top_k > 0:
                    probs[sorted_ids[:-top_k]] = 0
                if temperature != 1.0:
                    probs = probs ** (1.0 / temperature)
                out = torch.multinomial(probs, num_samples=1)[0]
                return int(out)
