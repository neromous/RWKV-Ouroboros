import math
import torch
import torch.nn as nn
from torch.nn import functional


# lora
class LoraLinear(nn.Module):
    def __init__(self, args, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r = args.lora.r
        alpha = args.lora.alpha
        dropout = args.trainer.dropout

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (functional.linear(x, self.weight) + self.scaling *
                functional.linear(functional.linear(self.lora_dropout(x), self.lora_A), self.lora_B))

