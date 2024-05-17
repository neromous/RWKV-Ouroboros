#########################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#########################################################################
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import deepspeed
from .block import Block, L2Wrap
# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
# if os.environ["RWKV_JIT_ON"] == "1":
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method


#########################################################################
# The RWKV Model with our blocks
#########################################################################

class RWKV(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.model.dtype == "fp32":
            self.args.model.dtype = torch.float
        elif self.args.model.dtype == "fp16":
            self.args.model.dtype = torch.half
        elif self.args.model.dtype == "bf16":
            self.args.model.dtype = torch.bfloat16

        model_weights = torch.load(self.args.model.load_model, map_location="cpu")
        model_keys = list(model_weights.keys())

        if self.args.model.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            self.args.model.n_layer = max_block_id + 1

        if self.args.model.n_embd < 0:
            self.args.model.n_embd = model_weights["head.weight"].shape[1]

        if self.args.model.vocab_size < 0:
            self.args.model.vocab_size = model_weights["head.weight"].shape[0]

        self.args.model.dim_att = self.args.model.n_embd
        self.args.model.n_head = self.args.model.dim_att // self.args.model.head_size
        self.args.model.dim_ffn = int((self.args.model.n_embd * 3.5) // 32 * 32)

        if not hasattr(self.args.model, 'tiny_att_layer'):
            self.args.model.tiny_att_layer = -1
        if not hasattr(self.args.model, 'tiny_att_dim'):
            self.args.model.tiny_att_dim = -1
        assert self.args.model.n_embd % 32 == 0
        assert self.args.model.dim_att % 32 == 0
        assert self.args.model.dim_ffn % 32 == 0
        # init speacker
        self.args.n_embd = self.args.model.n_embd
        self.args.dim_att = self.args.model.dim_att
        self.args.head_size_a = self.args.model.head_size_a

        self.emb = nn.Embedding(self.args.model.vocab_size,
                                self.args.model.n_embd)

        self.blocks = nn.ModuleList([Block(self.args, i) for i in range(self.args.model.n_layer)])

        self.ln_out = nn.LayerNorm(self.args.model.n_embd)
        self.head = nn.Linear(self.args.model.n_embd,
                              self.args.model.vocab_size,
                              bias=False)

        if self.args.trainer.head_qk > 0:
            self.head_q = nn.Linear(self.args.model.n_embd,
                                    self.args.trainer.head_qk,
                                    bias=False)
            self.head_k = nn.Linear(self.args.model.n_embd,
                                    self.args.trainer.head_qk,
                                    bias=False)
            self.register_buffer("copy_mask",
                                 torch.tril(torch.ones(self.args.model.ctx_len,
                                                       self.args.model.ctx_len)))
        if self.args.trainer.dropout > 0:
            self.drop0 = nn.Dropout(p=args.trainer.dropout)

        # for i in range(self.args.model.n_layer):
        #     n = f"blocks.{i}.att.time_state"
        #     model_weights[n] = torch.zeros(self.args.model.n_head,
        #                                    self.args.model.head_size,
        #                                    self.args.model.head_size)

        self.load_state_dict(model_weights, strict=False)
        del model_weights

        if args.trainer.train_type == "state-tuning":
            # self.requires_grad_(False)
            for name, module in self.named_modules():
                for pname, param in module.named_parameters():
                    if pname.endswith('.time_state') and pname.startswith('blocks.'):
                        print(pname)
                        param.requires_grad = True

        for p in self.parameters():
            p.data = p.data.to(dtype=self.args.model.dtype)

        gc.collect()
        torch.cuda.empty_cache()

    def get_optim_groups(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():

            # if not p.requires_grad:
            #     print("====", n)
                #continue
            if args.trainer.train_type == 'state-tuning':
                if 'time_state' not in n:
                    continue

            if (("_w1" in n) or ("_w2" in n)) and (args.trainer.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.trainer.layerwise_lr > 0):
                if args.trainer.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.trainer.layerwise_lr > 0):
                if args.trainer.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.trainer.layerwise_lr > 0):
                if args.trainer.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.trainer.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.trainer.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}

        if args.trainer.layerwise_lr > 0:
            if args.trainer.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 5.0},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x],
                     "weight_decay": 0.0,
                     "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x],
                             "weight_decay": 0.0,
                             "my_lr_scale": 1.0}]

        if args.trainer.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay],
                              "weight_decay": args.trainer.weight_decay,
                              "my_lr_scale": 1.0}]
        return optim_groups

    def forward(self, idx, states=None):
        """
        前向传一个序列
        """
        # print(next(self.parameters()).device)

        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)

        # 计算logits
        args = self.args

        B, T = idx.size()
        C = args.model.n_embd
        H = args.model.dim_att // args.model.head_size_a

        assert T <= self.args.model.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * args.model.head_size_a

        x = self.emb(idx)

        if torch.is_grad_enabled():
            new_states = states
        else:
            new_states = [torch.empty((self.args.model.n_layer, B,
                                       self.args.model.n_head,
                                       self.args.model.head_size,
                                       self.args.model.head_size),
                                      dtype=torch.float),
                          torch.empty((self.args.model.n_layer,
                                       2,
                                       B,
                                       C),
                                      dtype=torch.float)]


        if states is None:
            states = [torch.empty((self.args.model.n_layer, B,
                                       self.args.model.n_head,
                                       self.args.model.head_size,
                                       self.args.model.head_size),
                                  dtype=torch.float).cuda(),
                      torch.empty((self.args.model.n_layer,
                                   2,
                                   B,
                                   C),
                                  type=torch.float).cuda()]
        else:
            states = [x.cuda() for x in states]

        if args.trainer.dropout > 0:
            x = self.drop0(x)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if not torch.is_grad_enabled():
                block.att.patch_time_state = states[0][i]
                block.att.patch_shift_state = states[1][i, 0]
                block.ffn.patch_shift_state = states[1][i, 1]
            if args.trainer.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)
            if not torch.is_grad_enabled():
                new_states[0][i] = block.att.patch_time_state.detach().cpu()
                new_states[1][i,0] = block.att.patch_shift_state.detach().cpu()
                new_states[1][i,1] = block.ffn.patch_shift_state.detach().cpu()


        x = self.ln_out(x)
        x = self.head(x)
        return x, None
