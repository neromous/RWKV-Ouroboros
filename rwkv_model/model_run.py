import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from tqdm import tqdm
tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

RWKV_RESCALE_LAYER = 6 


class RWKV_RNN(torch.nn.Module):
    def __init__(self, model_name: str,
                 model_weights=False,
                 n_embd=-1,
                 n_layer=-1,
                 vocab_size=-1,
                 device='cuda'):
        super().__init__()
        self.model_name = model_name
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.device = device
        self.eval()
        if model_weights:
            w = model_weights
        else:
            w = torch.load(model_name, map_location='cpu')
        model_keys = list(w.keys())
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            self.n_layer = max_block_id + 1

        if n_embd < 0:
            self.n_embd = w['head.weight'].shape[1]

        if vocab_size < 0:
            self.vocab_size = w['head.weight'].shape[0]

        self.FLOAT_TYPE = w['head.weight'].dtype

        for k in tqdm(w.keys()):
            block_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            if '.time_' in k:
                w[k] = w[k].squeeze().to(self.device, self.FLOAT_TYPE)
            if '.time_decay' in k:
                w[k] = -torch.exp(w[k].to(self.device, dtype=self.FLOAT_TYPE))
            else:
                w[k] = w[k].to(self.device, dtype=self.FLOAT_TYPE)

            if self.FLOAT_TYPE == torch.float16:
                if 'att.output.weight' in k:
                    w[k] = w[k] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in k:
                    w[k] = w[k] / (2 ** int(block_id // RWKV_RESCALE_LAYER))

        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for k in w.keys():
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
        return F.layer_norm(x, (self.n_embd,), weight=w.weight, bias=w.bias)


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

    def forward(self, token, state, preprocess_only=False):
        with torch.no_grad():
            nef = -float('inf')
            nerf = 0
            if state == None:
                state = torch.zeros(self.n_layer * 5, self.n_embd,dtype=self.FLOAT_TYPE).to(self.device)
                for i in range(self.n_layer): state[5*i+4] = nef # -infinity
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            if preprocess_only:
                return state
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state





#         with torch.no_grad(): # precompute embedding
#             try:
#                 x = self.layer_norm(self.w.emb.weight, self.w.blocks[0].ln0)
#             except:
#                 x = F.layer_norm(self.w.emb.weight.float(),
#                                  (self.n_embd,),
#                                  weight=self.w.blocks[0].ln0.weight.float(),
#                                  bias=self.w.blocks[0].ln0.bias.float())
#             self.w.emb.weight = x.to(dtype=self.FLOAT_TYPE)

#     def layer_norm(self, x, w):
#         return F.layer_norm(x, (self.n_embd,), weight=w.weight, bias=w.bias)

#     def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
#         xx = state[5*i+0].to(dtype=self.FLOAT_TYPE)
#         xk = x * time_mix_k + xx * (1 - time_mix_k)
#         xr = x * time_mix_r + xx * (1 - time_mix_r)
#         state[5*i+0] = x.float()

#         r = torch.sigmoid(xr @ rw)
#         k = torch.square(torch.relu(xk @ kw))
#         kv = k @ vw
#         return r * kv

#     def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
#         xx = torch.cat((state[5*i+0].to(dtype=self.FLOAT_TYPE).unsqueeze(0), x[:-1,:]))
#         xk = x * time_mix_k + xx * (1 - time_mix_k)
#         xr = x * time_mix_r + xx * (1 - time_mix_r)
#         state[5*i+0] = x[-1,:].float()

#         r = torch.sigmoid(xr @ rw)
#         k = torch.square(torch.relu(xk @ kw))
#         kv = k @ vw
#         return r * kv

#     def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
#         xx = state[5*i+1].to(dtype=self.FLOAT_TYPE)
#         xk = x * time_mix_k + xx * (1 - time_mix_k)
#         xv = x * time_mix_v + xx * (1 - time_mix_v)
#         xr = x * time_mix_r + xx * (1 - time_mix_r)
#         state[5*i+1] = x.float()

#         r = torch.sigmoid(xr @ rw)
#         k = (xk @ kw).float()
#         v = (xv @ vw).float()

#         aa = state[5*i+2]
#         bb = state[5*i+3]
#         pp = state[5*i+4]
#         ww = time_first + k
#         p = torch.maximum(pp, ww)
#         e1 = torch.exp(pp - p)
#         e2 = torch.exp(ww - p)
#         a = e1 * aa + e2 * v
#         b = e1 * bb + e2
#         ww = pp + time_decay
#         p = torch.maximum(ww, k)
#         e1 = torch.exp(ww - p)
#         e2 = torch.exp(k - p)
#         state[5*i+2] = e1 * aa + e2 * v
#         state[5*i+3] = e1 * bb + e2
#         state[5*i+4] = p
#         wkv = (a / b).to(dtype=self.FLOAT_TYPE)
#         return (r * wkv) @ ow

#     def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
#         xx = torch.cat((state[5*i+1].to(dtype=self.FLOAT_TYPE).unsqueeze(0), x[:-1,:]))
#         xk = x * time_mix_k + xx * (1 - time_mix_k)
#         xv = x * time_mix_v + xx * (1 - time_mix_v)
#         xr = x * time_mix_r + xx * (1 - time_mix_r)
#         state[5*i+1] = x[-1,:].float()

#         r = torch.sigmoid(xr @ rw)
#         k = (xk @ kw).float()
#         v = (xv @ vw).float()

#         aa = state[5*i+2]
#         bb = state[5*i+3]
#         pp = state[5*i+4]
#         T = x.shape[0]
#         for t in range(T):
#             ww = time_first + k[t]
#             p = torch.maximum(pp, ww)
#             e1 = torch.exp(pp - p)
#             e2 = torch.exp(ww - p)
#             a = e1 * aa + e2 * v[t]
#             b = e1 * bb + e2
#             ww = pp + time_decay
#             p = torch.maximum(ww, k[t])
#             e1 = torch.exp(ww - p)
#             e2 = torch.exp(k[t] - p)
#             if t != T - 1:
#                 aa = e1 * aa + e2 * v[t]
#                 bb = e1 * bb + e2
#                 pp = p
#             else:
#                 state[5*i+2] = e1 * aa + e2 * v[t]
#                 state[5*i+3] = e1 * bb + e2
#                 state[5*i+4] = p
#             xx[t] = (a / b).to(dtype=self.FLOAT_TYPE)
#         return (r * xx) @ ow




    
    # def channel_mixing(self, x, state, i:int,
    #                    time_mix_k, time_mix_r, kw, vw, rw):
    #     xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
    #     xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
    #     state[5*i+0] = x
    #     r = torch.sigmoid(rw @ xr)
    #     k = torch.square(torch.relu(kw @ xk))
    #     return r * (vw @ k)


    # def time_mixing(self, x, state, i:int,
    #                 time_mix_k, time_mix_v,
    #                 time_mix_r, time_first, time_decay, kw, vw, rw, ow):
    #     xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
    #     xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
    #     xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
    #     state[5*i+1] = x
    #     r = torch.sigmoid(rw @ xr)
    #     k = kw @ xk
    #     v = vw @ xv

    #     aa = state[5*i+2]
    #     bb = state[5*i+3]
    #     pp = state[5*i+4]
    #     ww = time_first + k
    #     qq = torch.maximum(pp, ww)
    #     e1 = torch.exp(pp - qq)
    #     e2 = torch.exp(ww - qq)
    #     a = e1 * aa + e2 * v
    #     b = e1 * bb + e2
    #     wkv = a / b
    #     ww = pp + time_decay
    #     qq = torch.maximum(ww, k)
    #     e1 = torch.exp(ww - qq)
    #     e2 = torch.exp(k - qq)
    #     state[5*i+2] = e1 * aa + e2 * v
    #     state[5*i+3] = e1 * bb + e2
    #     state[5*i+4] = qq
    #     return ow @ (r * wkv)

    # def forward(self, tokens, state):
    #     with torch.no_grad():
    #         w = self.w
    #         seq_mode = len(tokens) > 1
    #         x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[0]]
    #         if state == None:
    #             state = torch.zeros(self.n_layer * 5, self.n_embd,device=self.device)
    #             for i in range(self.n_layer):
    #                 state[5*i+4] -= 1e-30 # -infinity

    #         SA = self.SA_seq if seq_mode else self.SA_one
    #         FF = self.FF_seq if seq_mode else self.FF_one


    #         for i in range(self.n_layer):
    #             print("====", i)
    #             att = self.w.blocks[i].att
    #             x = x + SA(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
    #                 att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay,
    #                 att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
    #             ffn = self.w.blocks[i].ffn
    #             x = x + FF(self.layer_norm(x, self.w.blocks[i].ln2), state, i,
    #                 ffn.time_mix_k, ffn.time_mix_r,
    #                 ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

    #             if self.FLOAT_TYPE == torch.float16:
    #                 if (i+1) % RWKV_RESCALE_LAYER == 0:
    #                     x = x / 2
                        
    #         x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
    #         x = w.head.weight @ x

    #         # x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
    #         return x.type(self.FLOAT_TYPE), state

    #     @classmethod
    #     def sample_logits(cls, out, temperature=1.0, top_p=0.8):
    #         probs = F.softmax(out, dim=-1).numpy()
    #         sorted_probs = np.sort(probs)[::-1]
    #         cumulative_probs = np.cumsum(sorted_probs)
    #         cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    #         probs[probs < cutoff] = 0
    #         if temperature != 1.0:
    #             probs = probs.pow(1.0 / temperature)
    #         probs = probs / np.sum(probs)
    #         out = np.random.choice(a=len(probs), p=probs)
    #         return out



    

