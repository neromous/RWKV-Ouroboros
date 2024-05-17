import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import functional
from torch._lowrank import svd_lowrank


# lora
class LoraLinear(nn.Module):

    def __init__(self, args, in_features: int, out_features: int, bias: bool, svd_niter=4):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r = args.lora.r
        alpha = args.lora.alpha
        dropout = args.trainer.dropout
        self.r = r
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
        Vhr = Vr.t()
        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.weight.data = self.weight.data - lora_B @ lora_A     

    def forward(self, x):
        return (F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B))


# lora
class LoraLinearOrigin(nn.Module):

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
        return (functional.linear(x, self.weight) + self.scaling * functional.linear(functional.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


# class PissaLinear(nn.Module):

#     def __init__(self, args, in_features: int, out_features: int, bias: bool):
#         super().__init__()

#         self.weight = nn.Parameter(torch.empty((out_features, in_features)))
#         assert bias == False, "Biased LoraLinear not supported"

#         r = args.lora.r
#         alpha = args.lora.alpha
#         dropout = args.trainer.dropout

#         self.lora_A = nn.Parameter(torch.empty(r, in_features))
#         self.lora_B = nn.Parameter(torch.empty(out_features, r))
#         self.lora_dropout = nn.Dropout(dropout)
#         self.scaling = alpha / r

#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)

#     def pissa_init(self, svd_niter):
#         self.pissa = True
#         Ur, Sr, Vr = svd_lowrank(self.weight.data, self.r, niter=svd_niter)
#         Vhr = Vr.t()
#         lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
#         lora_B = Ur @ torch.diag(torch.sqrt(Sr))
#         self.lora_A.data = lora_A
#         self.lora_B.data = lora_B
#         self.weight.data = self.weight.data - lora_B @ lora_A        

#     def forward(self, x):
#         return (F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B))