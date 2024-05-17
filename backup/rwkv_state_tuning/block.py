import torch.nn as nn
from .time_mix import TimeMix
from .channel_mix import ChannelMix


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
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x):
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)

        if self.args.dropout > 0:
            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))
        else:
            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))

        return x
