#############################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#############################################################################
import torch
from torch import nn


#########################################################################
# initlize the att ffn
#########################################################################
from .channel_mix.channel_mix_v6 import ChannelMix
from .time_mix.time_mix_v6_state import TimeMix

#########################################################################
# initlize the blocks
#########################################################################

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.model.n_embd)
        self.ln2 = nn.LayerNorm(args.model.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.model.n_embd)
            if args.trainer.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,
                                                           args.trainer.my_pos_emb,
                                                           args.model.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.trainer.my_pos_emb,
                                                           1,
                                                           args.model.n_embd)))

        if self.layer_id == 0 and self.args.trainer.pre_ffn > 0:
            self.ffnPre = ChannelMix(args, 0)
        else:
            self.att = TimeMix(args, layer_id)
            self.ffn = ChannelMix(args, layer_id)

        if args.model.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.model.n_embd)
            self.tiny_q = nn.Linear(args.model.n_embd,
                                    args.model.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.model.n_embd,
                                    args.model.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.model.n_embd,
                                    args.model.n_embd, bias=False)
            self.register_buffer("tiny_mask",
                                 torch.tril(torch.ones(args.model.ctx_len,
                                                       args.model.ctx_len)))

        if args.trainer.dropout > 0:
            self.drop0 = nn.Dropout(p=args.trainer.dropout)
            self.drop1 = nn.Dropout(p=args.trainer.dropout)

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()

        if self.layer_id == 0:
            x = self.ln0(x)
            if args.trainer.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.args.trainer.dropout == 0:
            if self.layer_id == 0 and args.trainer.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.trainer.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))
        if args.model.tiny_att_dim > 0 and self.layer_id == args.model.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.model.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
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
