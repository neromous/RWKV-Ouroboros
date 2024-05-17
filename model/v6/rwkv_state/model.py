##############################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
##############################################
import importlib, gc
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import types
if importlib.util.find_spec('deepspeed'):
    import deepspeed
from .block import Block, BlockStateList
from .lora import LoraLinear

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
        args.model = args_in.model
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
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # init dropout
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

        if args.lora.lora_on:
            if args.lora.train_state:
                pass
            else:
                model_weights = self.load_lora(model_weights)
        

        self.load_state_dict(model_weights, strict=False)
        del model_weights

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
        lr_10x = set()
        for n, p in self.named_parameters():
            if args.lora.lora_on:
                if args.lora.train_state:
                    if "encode" in n:
                        print("======load state===========")
                        lr_1x.add(n)
                    if "decode" in n:
                        print("======load state===========")
                        lr_1x.add(n)
                    else:
                        pass
                else:
                    print("======load lora===========")
                    if 'time_state' in n:
                        lr_10x.add(n)
                    elif 'time_decay' in n:
                        lr_2x.add(n)
                    elif "lora_" in n:
                        lr_1x.add(n)
                    elif (".ln" in n) and ("ln" in args.lora.parts):
                        lr_1x.add(n)
                    elif "time" in n:
                        lr_1x.add(n)
                    elif ("emb.weight" == n) and ("emb" in args.lora.parts):
                        lr_1x.add(n)
                    elif ("head.weight" == n) and ("head" in args.lora.parts):
                        lr_1x.add(n)
                    elif ("gate" == n) and ("gate" in args.lora.parts):
                        lr_1x.add(n)
                    else:
                        pass
            else:
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
        lr_10x = sorted(list(lr_10x))
        param_dict = {n: p for n, p in self.named_parameters()}

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
            {"params": [param_dict[n] for n in lr_10x],
                "weight_decay": 0.0,
                "my_lr_scale": 10.0},
        ]

        if args.trainer.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay],
                              "weight_decay": args.trainer.weight_decay,
                              "my_lr_scale": 1.0}]
        return optim_groups


    def forward(self, idx, states=None):
        args = self.args

        # idx
        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)

        # 计算logits
        args = self.args

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * args.head_size


        new_states = BlockStateList.create(args.n_layer,
                                           B,
                                           C,
                                           args.n_head,
                                           args.head_size,
                                           idx.device,
                                           self.emb.weight.dtype)
        if states is None:
            states = BlockStateList.create(args.n_layer,
                                           B,
                                           C,
                                           args.n_head,
                                           args.head_size,
                                           idx.device,
                                           self.emb.weight.dtype)
        else:
            states = BlockStateList(states.shift_states, states.wkv_states)
        states = states.cuda()

        x = self.emb(idx)

        if args.dropout > 0:
            x = self.drop0(x)

        for i in range(len((self.blocks))):
            block = self.blocks[i]
            state = states[i]

            if int(args.grad_cp) == 1:
                if args.lora.r > 0:
                    x, state = torch_checkpoint(block, x, state, use_reentrant=False)
                else:
                    x, state = deepspeed.checkpointing.checkpoint(block, x, state)
            else:
                x, state = block(x, state)
            new_states[i] = state

        x = self.ln_out(x)
        logits = self.head(x)

        return logits, new_states


    def load_lora(self, weight, debug=True):
        if len(self.args.lora.path) != 0:
            try: 
                print("===========load from path======================")
                lora_weight = torch.load(self.args.lora.path, map_location='cpu')
                if debug:
                    for k, v in lora_weight.items():
                        print(k)
                weight.update(lora_weight)
            except:
                print("======load error====")
        return weight
