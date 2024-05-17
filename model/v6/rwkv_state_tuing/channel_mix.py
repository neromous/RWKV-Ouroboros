import torch
import torch.nn as nn
from .lora import LoraLinear

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
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        # ffn lora
        if "ffn" in args.lora.parts and args.lora.r > 0:
            self.key = LoraLinear(args, args.n_embd, args.dim_ffn, bias=False)
            self.receptance = LoraLinear(args, args.n_embd, args.n_embd, bias=False)
            self.value = LoraLinear(args, args.dim_ffn, args.n_embd, bias=False)
        else:
            print("====do not load lora=======")
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)


    def forward(self, x, last_state):
        xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, x[:, -1]        


    # def forward_no_cuda(self, x, last_state):
    #     xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1) - x
    #     xk = x + xx * self.time_maa_k
    #     xr = x + xx * self.time_maa_r

    #     k = self.key(xk)
    #     k = torch.relu(k) ** 2
    #     kv = self.value(k)
    #     return torch.sigmoid(self.receptance(xr)) * kv, x[:, -1]

    # def forward_cuda(self, x, state):
    #     xx = self.time_shift(x) - x
    #     xk = x + xx * self.time_maa_k
    #     xr = x + xx * self.time_maa_r

    #     k = self.key(xk)
    #     k = torch.relu(k) ** 2
    #     kv = self.value(k)
    #     return torch.sigmoid(self.receptance(xr)) * kv, x[:, -1]

    # def forward(self, x, last_state):
    #     if self.training:
    #         return self.forward_cuda(x, last_state)
    #     else:
    #         return self.forward_infer_cuda(x, last_state)