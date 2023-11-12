import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
import torch.nn as nn
from utils import log, load_config
config = load_config()
datatype = config['environ']['RWKV_FLOAT_MODE']
# args = types.SimpleNamespace()
# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
# args.n_layer = 24
# args.n_embd = 1024

# context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# NUM_TRIALS = 3
# LENGTH_PER_TRIAL = 100
# TEMPERATURE = 1.0
# TOP_P = 0.85

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
            self.my_datatype = torch.float
        elif datatype == "fp16":
            self.my_datatype = torch.half
        elif datatype == "bf16":
            self.my_datatype = torch.bfloat16

        self.args = types.SimpleNamespace()
        self.args.n_layer= config['model']['n_layer']
        self.args.n_embd= config['model']['n_embd']
        with torch.no_grad():
            self.eval() # set torch to inference mode
            w = weight
            for k in w.keys():
                print(f'====={k}')
                w[k] = w[k].to(dtype=self.my_datatype)
                if '.time_' in k:
                    w[k] = w[k].squeeze().to(dtype=self.my_datatype)
                if '.time_decay' in k:
                    w[k] = torch.exp(-torch.exp(w[k].to(dtype=self.my_datatype))).reshape(-1,1,1)
                if '.time_faaaa' in k:
                    print("=====")
                    w[k.replace('.time_faaaa','.time_first')] = torch.exp(w[k].to(dtype=self.my_datatype)).reshape(-1,1,1)
                    #w[k] =

            #
            self.n_head = w['blocks.0.att.time_decay'].shape[0]
            self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head

            #
            self.w = types.SimpleNamespace()
            self.w.blocks = {}
            for k in tqdm(w.keys()): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
                parts = k.split('.')
                last = parts.pop()
                here = self.w
                for p in parts:
                    if p.isdigit():
                        p = int(p)
                        if p not in here: here[p] = types.SimpleNamespace()
                        here = here[p]
                    else:
                        if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                        here = getattr(here, p)
                setattr(here, last, w[k])
            # for k in tqdm(w.keys()):
            #     parts = k.split('.')
            #     last = parts.pop()
            #     here = self.w
            #     for p in parts:
            #         if p.isdigit():
            #             p = int(p)
            #             if p not in here:
            #                 here[p] = types.SimpleNamespace()
            #             here = here[p]
            #         else:
            #             if not hasattr(here, p):
            #                 setattr(here, p, types.SimpleNamespace())
            #             here = getattr(here, p)
            #     setattr(here, last, w[k])

    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(gemm(rx, rw))
        vx = torch.square(torch.relu(gemm(kx, kw)))
        out = r * gemm(vx, vw)
        return x + out, xx

    def att_one_v5_1(self, x, sx, s, ln_w, ln_b,
                     lx_w, lx_b, k_mix, v_mix,
                     r_mix, g_mix, t_decay, t_first,
                     kw, vw, rw, gw, ow,
                     kmx, krx, kmy, kry,
                     vmx, vrx, vmy, vry,
                     rmx, rrx, rmy, rry,
                     omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        S = x.shape[-1] // H

        r = gemm(rx, rw, output_dtype=torch.float32).view(H, 1, S)
        k = gemm(kx, kw, output_dtype=torch.float32).view(H, S, 1)
        v = gemm(vx, vw, output_dtype=torch.float32).view(H, 1, S)
        g = F.silu(gemm(gx, gw))

        a = gemm(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = gemm(out, ow)

        return x + out, xx, s


    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        i0 = (2+self.head_size)*i+0
        xk = x * time_mix_k + state[i0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[i0] * (1 - time_mix_r)
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    def time_mixing(self, x, state, i:int, time_mix_k,
                    time_mix_v, time_mix_r,
                    time_first, time_decay, kw, vw, rw, ow, ln_w, ln_b):
        H = self.n_head
        S = self.head_size

        i1 = (2+S)*i+1
        xk = x * time_mix_k + state[i1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[i1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[i1] * (1 - time_mix_r)
        state[i1] = x
        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)

        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + time_decay * s

        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b).squeeze(0)
        return ow @ x

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay,
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i,
                    ffn.time_mix_k, ffn.time_mix_r,
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state
