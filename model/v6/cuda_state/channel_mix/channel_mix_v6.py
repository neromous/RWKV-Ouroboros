import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop


class ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        args = args.model
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.patch_shift_state = None

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


    # @MyFunction
    # def forward(self, x):
    #     if torch.is_grad_enabled():
    #         xx = self.time_shift(x) - x
    #         xk = x + xx * self.time_maa_k
    #         xr = x + xx * self.time_maa_r
    #         k = self.key(xk)
    #         k = torch.relu(k) ** 2
    #         kv = self.value(k)
    #         return torch.sigmoid(self.receptance(xr)) * kv
    #     else:
    #         # get state
    #         last_state = self.patch_shift_state
    #         # get state end
    #         xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1).bfloat16()
    #         dxx = xx - x
    #         xk = x + dxx * self.time_maa_k
    #         xr = x + dxx * self.time_maa_r
    #         kv = self.value(torch.relu(self.key(xk)) ** 2)
    #         # set state
    #         self.patch_shift_state = x[:, -1]
    #         # set state end
    #         return torch.sigmoid(self.receptance(xr)) * kv
