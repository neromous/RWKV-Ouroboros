import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from tqdm import tqdm
import torch.nn as nn

# args = types.SimpleNamespace()
# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
# args.n_layer = 24
# args.n_embd = 1024

# context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# NUM_TRIALS = 3
# LENGTH_PER_TRIAL = 100
# TEMPERATURE = 1.0
# TOP_P = 0.85

class RWKV_RNN(nn.Module):
    def __init__(self,weight, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode

        w = weight
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) 
            else: w[k] = w[k].float()
            
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
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd).cuda()
                for i in range(self.args.n_layer):
                    state[5*i+4] = -1e30 # -infinity
            
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

##########################################################################################################

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

# ########################################################################################################

# print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
# model = RWKV_RNN(args)

# print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
# init_state = None
# for token in tokenizer.encode(context).ids:
#     init_out, init_state = model.forward(token, init_state)

# for TRIAL in range(NUM_TRIALS):
#     print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
#     all_tokens = []
#     out_last = 0
#     out, state = init_out.clone(), init_state.clone()
#     for i in range(LENGTH_PER_TRIAL):
#         token = sample_logits(out, TEMPERATURE, TOP_P)
#         all_tokens += [token]
#         tmp = tokenizer.decode(all_tokens[out_last:])
#         if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
#             print(tmp, end="", flush=True)
#             out_last = i + 1
#         out, state = model.forward(token, state)       
# print('\n')

        # with torch.no_grad():
        #     n_layer=self.args.n_layer
        #     n_embd=self.args.n_embd
        #     if state == None:
        #         state = torch.zeros(n_layer * 5, n_embd).to('cuda').float()
        #         for i in range(n_layer):
        #             state[5*i+4] = -1e30 # -infinity
        #     #=================emb===============
        #     emb = model_dict['emb.weight'].float()
        #     x =  emb[token]
        #     #================ln0================
        #     ln0_weight = model_dict['blocks.0.ln0.weight'].float()
        #     ln0_bias = model_dict['blocks.0.ln0.bias'].float()
        #     x = self.layer_norm(x, ln0_weight, ln0_bias, n_embd)
        #     #============layers====================
        #     for i in range(n_layer):
        #         time_mix_k =  model_dict[f'blocks.{i}.att.time_mix_k'].squeeze().float()
        #         time_mix_v =  model_dict[f'blocks.{i}.att.time_mix_v'].squeeze().float()
        #         time_mix_r =  model_dict[f'blocks.{i}.att.time_mix_r'].squeeze().float()
        #         time_first =  model_dict[f'blocks.{i}.att.time_first'].squeeze().float()
        #         time_decay =  model_dict[f'blocks.{i}.att.time_decay'].squeeze().float()
        #         time_decay = -torch.exp(time_decay)
        #         att_key =  model_dict[f'blocks.{i}.att.key.weight'].float()
        #         att_value =  model_dict[f'blocks.{i}.att.value.weight'].float()
        #         att_receptance =  model_dict[f'blocks.{i}.att.receptance.weight'].float()
        #         att_output =  model_dict[f'blocks.{i}.att.output.weight'].float()
        #         ln1_weight = model_dict[f'blocks.{i}.ln1.weight'].float()
        #         ln1_bias = model_dict[f'blocks.{i}.ln1.bias'].float()
        #         x = self.layer_norm(x,ln1_weight, ln1_bias,n_embd)

        #         tx, state =  self.time_mixing(x, state, i,
        #             time_mix_k, time_mix_v, time_mix_r, time_first, time_decay,
        #             att_key, att_value, att_receptance, att_output)
        #         x = x + tx
        #         time_mix_k =  model_dict[f'blocks.{i}.ffn.time_mix_k'].squeeze().float()
        #         time_mix_r =  model_dict[f'blocks.{i}.ffn.time_mix_r'].squeeze().float()
        #         ffn_key =  model_dict[f'blocks.{i}.ffn.key.weight'].float()
        #         ffn_value =  model_dict[f'blocks.{i}.ffn.value.weight'].float()
        #         ffn_receptance =  model_dict[f'blocks.{i}.ffn.receptance.weight'].float()
        #         ln2_weight = model_dict[f'blocks.{i}.ln2.weight'].float()
        #         ln2_bias = model_dict[f'blocks.{i}.ln2.bias'].float()
        #         x = self.layer_norm(x, ln2_weight, ln2_bias,n_embd)
                
        #         fx,state = self.channel_mixing(x, state, i,
        #             time_mix_k, time_mix_r,
        #             ffn_key, ffn_value, ffn_receptance)
        #         x = x + fx

        #     ln_out_weight = model_dict['ln_out.weight'].float()
        #     ln_out_bias = model_dict['ln_out.bias'].float()
        #     x = self.layer_norm(x , ln_out_weight,ln_out_bias,n_embd)
        #     head_weight =  model_dict['head.weight'].float()
        #     x = head_weight @ x
        #     return x.float(), state
