<import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
import torch.nn as nn
from config import log, load_config
config = load_config()
datatype = config['environ']['RWKV_FLOAT_MODE']


def sample_logits(logits:torch.tensor, temperature=0.1, top_p=0.1, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if probs.device == torch.device('cpu'):
        probs = probs.numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
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
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)



class RWKV_RNN(nn.Module):
    def __init__(self,weight):
        super().__init__()
        if datatype == "fp32":
            self.my_datatype=torch.float
        elif datatype == "fp16":
            self.my_datatype=torch.half
        elif datatype == "bf16":
            self.my_datatype = torch.bfloat16

        self.args = types.SimpleNamespace()
        self.args.n_layer= config['model']['n_layer']
        self.args.n_embd= config['model']['n_embd']
        with torch.no_grad():
            self.eval() # set torch to inference mode
            w = weight
            for k in tqdm(w.keys()):
                if      '.time_' in k: w[k] = w[k].squeeze().to(dtype=self.my_datatype)
                if '.time_decay' in k: w[k] = -torch.exp(w[k].to(dtype=self.my_datatype)) 
                else: w[k] = w[k].to(dtype=self.my_datatype)

            self.w = types.SimpleNamespace() 
            self.w.blocks = {}
            for k in tqdm(w.keys()):
                parts = k.split('.')
                last = parts.pop()
                here = self.w
                for p in parts:
                    if p.isdigit():
                        p = int(p)
                        if p not in here:
                            here[p] = types.SimpleNamespace()
                        here = here[p]
                    else:
                        if not hasattr(here, p):
                            setattr(here, p, types.SimpleNamespace())
                        here = getattr(here, p)
                setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)


    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)


    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv
        
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq
        return ow @ (r * wkv)

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd).to('cuda',dtype=self.my_datatype)
                for i in range(self.args.n_layer):
                    state[5*i+4] = -float('inf') # -infinity
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1),
                                         state, i, 
                                         att.time_mix_k,
                                         att.time_mix_v,
                                         att.time_mix_r,
                                         att.time_first,
                                         att.time_decay, 
                                         att.key.weight,
                                         att.value.weight,
                                         att.receptance.weight,
                                         att.output.weight)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x,self.w.blocks[i].ln2),
                                            state, i, 
                                            ffn.time_mix_k, ffn.time_mix_r, 
                                            ffn.key.weight,
                                            ffn.value.weight,
                                            ffn.receptance.weight)
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state



