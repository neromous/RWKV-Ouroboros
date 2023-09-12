########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, math, os, gc
import torch
from torch.nn import functional as F
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import numpy as np
from .config import get_environ,get_trainer,read_config
from tqdm import tqdm

tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def __nop(ob):
    return ob

if  get_environ("RWKV_JIT_ON")  == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = nn.Module
    MyFunction = __nop


RWKV_RESCALE_LAYER = 6 # set x = x/2 every X layer (to avoid FP16 overflow)

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args, model_state):
        super().__init__()

        self.args = args
        if args.FLOAT_MODE == 'fp32':
            self.FLOAT_MODE = torch.float
        elif args.FLOAT_MODE == 'fp16':
            self.FLOAT_MODE = torch.half
        elif args.FLOAT_MODE == 'bf16':
            self.FLOAT_MODE = torch.bfloat16
        self.RUN_DEVICE = args.RUN_DEVICE
        with torch.no_grad():
            #w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            w = model_state
            gc.collect()
            args.n_embd = w['emb.weight'].shape[1]
            args.n_layer = 0
            keys = list(w.keys()) # refine weights and send to correct device
            print_need_newline = False
            for x in tqdm(keys):
                w[x].requires_grad = False
                if x == 'emb.weight' or 'ln0' in x:
                    continue

                block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, block_id+1)

                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
                    w[x] = w[x].t()

                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].to(dtype=self.FLOAT_MODE)

                if args.FLOAT_MODE == 'fp16':
                    if 'att.output.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                    if 'ffn.value.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))

                if 'cuda' in args.RUN_DEVICE:
                    w[x] = w[x].to(self.RUN_DEVICE)

                if 'ffn.value.weight' in x:
                    gc.collect()
                    if 'cuda' in args.RUN_DEVICE:
                        torch.cuda.empty_cache()

        keys = list(w.keys()) # store weights in self.w
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        with torch.no_grad(): # precompute embedding
            try:
                x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
            except:
                x = F.layer_norm(self.w.emb.weight.float(),
                                 (self.args.n_embd,),
                                 weight=self.w.blocks[0].ln0.weight.float(),
                                 bias=self.w.blocks[0].ln0.bias.float())
            self.w.emb.weight = x.to(dtype=self.FLOAT_MODE)

        self.eval()
        gc.collect()
        if 'cuda' in args.RUN_DEVICE:
            torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5*i+0].to(dtype=self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.float()

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = torch.cat((state[5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x[-1,:].float()

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = state[5*i+1].to(dtype=self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x.float()

        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = (a / b).to(dtype=self.FLOAT_MODE)
        return (r * wkv) @ ow

    @MyFunction
    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = torch.cat((state[5*i+1].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x[-1,:].float()

        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        T = x.shape[0]
        for t in range(T):
            ww = time_first + k[t]
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = torch.maximum(ww, k[t])
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k[t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5*i+2] = e1 * aa + e2 * v[t]
                state[5*i+3] = e1 * bb + e2
                state[5*i+4] = p
            xx[t] = (a / b).to(dtype=self.FLOAT_MODE)
        return (r * xx) @ ow

    def forward(self, tokens, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            seq_mode = len(tokens) > 1

            x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[0]]
            if 'cuda' in self.RUN_DEVICE:
                x = x.to(self.RUN_DEVICE)

            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            SA = self.SA_seq if seq_mode else self.SA_one
            FF = self.FF_seq if seq_mode else self.FF_one

            for i in range(args.n_layer):
                ww = w.blocks[i].att
                x = x + SA(self.LN(x, w.blocks[i].ln1), state, i,
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)

                ww = w.blocks[i].ffn
                x = x + FF(self.LN(x, w.blocks[i].ln2), state, i,
                    ww.time_mix_k, ww.time_mix_r,
                    ww.key.weight, ww.value.weight, ww.receptance.weight)

                if args.FLOAT_MODE == 'fp16':
                    if (i+1) % RWKV_RESCALE_LAYER == 0:
                        x = x / 2

            if preprocess_only:
                return state

            x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state


################################

role_config = read_config()['role']

def prompt2text(prompt:dict):
    over = prompt.get('over',False)
    role_text = prompt['role']
    content = prompt['content']
    role = role_config.get(role_text, False)
    role_prefix = role['prefix']
    role_postfix = role['postfix']
    tokens = tokenizer.encode(content)
    tokens = role_prefix + tokens
    if over:
        tokens = tokens + role_postfix
    prompt['tokens'] = tokens
    return prompt


def rwkv_sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
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

def rwkv_generate(model, message:dict, callback=None, state=None):
    all_tokens = []
    out_last = 0
    out_str = ''
    token_count =  message["token_count"]
    occurrence = {}
    if token_count == 0:
        tokens = prompt2text(message)['tokens']
        while len(tokens) > 0:
            out, state = model.forward(tokens[:message['chunk_len']], state)
            tokens = tokens[message['chunk_len']:]
        response = {"response":"success generate"}
        return response, state
    else:

        for i in range(token_count):
            tokens = prompt2text(message)['tokens']  if i == 0 else [token]
            # forward & adjust prob.
            while len(tokens) > 0:
                out, state = model.forward(tokens[:message['chunk_len']], state)
                tokens = tokens[message['chunk_len']:]

            for n in message['token_ban']:
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (message['alpha_presence'] + occurrence[n] * message['alpha_frequency'])

            # sampler
            token = rwkv_sample_logits(out,
                                           temperature=message["temperature"],
                                           top_p=message['top_p'],
                                           top_k=message["top_k"])
            if token in message["token_stop"]:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= message['alpha_decay']
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            # print(occurrence) # debug

            # output
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
        response = {"prompt": message['content'], "response": out_str}
        return response, state




########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# import numpy as np
# np.set_printoptions(precision=4, suppress=True, linewidth=200)
# import types, torch
# import copy
# from torch.nn import functional as F
# from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
# from tqdm import tqdm
# tokenizer = TRIE_TOKENIZER('/home/neromous/.minicoda3/envs/blackfog/lib/python3.9/site-packages/rwkv/rwkv_vocab_v20230424.txt')
# import gc

#from tokenizers import Tokenizer

########################################################################################################

# class RWKV_RNN(torch.jit.ScriptModule):
#     def __init__(self, args, model_dict):
#         super().__init__()
#         self.args = args
#         # self.eval() # set torch to inference mode
#         # w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
#         with torch.no_grad():
#             w = model_dict
#             for k in tqdm(w.keys()):
#                 w[k].requires_grad = False
#                 block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0


#                 if  '.time_' in k:
#                     w[k] = w[k].squeeze()
#                 if '.time_decay' in k:
#                     w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
#                 else:
#                     w[k] = w[k].to(dtype=torch.half) # convert to f32 type
#                 w[k] = w[k].to('cuda')
#                 if 'ffn.value.weight' in k:
#                     gc.collect()
#                     if 'cuda' in self.args.RUN_DEVICE:
#                         torch.cuda.empty_cache()
#                 if self.args.FLOAT_MODE == 'fp16':
#                     if 'att.output.weight' in x:
#                         w[k] = w[k] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
#                     if 'ffn.value.weight' in x:
#                         w[k] = w[k] / (2 ** int(block_id // RWKV_RESCALE_LAYER))


#             self.w = types.SimpleNamespace() # set self.w from w
#             self.w.blocks = {}
#             for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
#                 parts = k.split('.')
#                 last = parts.pop()
#                 here = self.w
#                 for p in parts:
#                     if p.isdigit():
#                         p = int(p)
#                         if p not in here:
#                             here[p] = types.SimpleNamespace()
#                         here = here[p]
#                     else:
#                         if not hasattr(here, p):
#                             setattr(here, p, types.SimpleNamespace())
#                         here = getattr(here, p)
#                 setattr(here, last, w[k])

#         with torch.no_grad(): # precompute embedding
#             try:
#                 x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
#             except:
#                 x = F.layer_norm(self.w.emb.weight.float(),
#                                  (self.args.n_embd,),
#                                  weight=self.w.blocks[0].ln0.weight.float(),
#                                  bias=self.w.blocks[0].ln0.bias.float())
#             self.w.emb.weight = x.to(dtype=self.FLOAT_MODE)

#         self.eval()
#         gc.collect()
#         if 'cuda' in args.RUN_DEVICE:
#             torch.cuda.empty_cache()

#     def layer_norm(self, x, w):
#         return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

#     @torch.jit.script_method
#     def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
#         xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
#         xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
#         state[5*i+0] = x
#         r = torch.sigmoid(rw @ xr)
#         k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
#         return r * (vw @ k)

#     @torch.jit.script_method
#     def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
#         xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
#         xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
#         xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
#         state[5*i+1] = x
#         r = torch.sigmoid(rw @ xr)
#         k = kw @ xk
#         v = vw @ xv

#         aa = state[5*i+2]
#         bb = state[5*i+3]
#         pp = state[5*i+4]
#         ww = time_first + k
#         qq = torch.maximum(pp, ww)
#         e1 = torch.exp(pp - qq)
#         e2 = torch.exp(ww - qq)
#         a = e1 * aa + e2 * v
#         b = e1 * bb + e2
#         wkv = a / b
#         ww = pp + time_decay
#         qq = torch.maximum(ww, k)
#         e1 = torch.exp(ww - qq)
#         e2 = torch.exp(k - qq)
#         state[5*i+2] = e1 * aa + e2 * v
#         state[5*i+3] = e1 * bb + e2
#         state[5*i+4] = qq
#         return ow @ (r * wkv)

#     def forward(self, token, state):
#         with torch.no_grad():
#             if state == None:
#                 state = torch.zeros(self.args.n_layer * 5, self.args.n_embd,device=self.args.RUN_DEVICE)
#                 for i in range(self.args.n_layer): state[5*i+4] = -1e30 # -infinity

#             x = self.w.emb.weight[token]
#             if 'cuda' in self.args.RUN_DEVICE:
#                 x = x.to('cuda')
#             x = self.layer_norm(x, self.w.blocks[0].ln0)
#             for i in range(self.args.n_layer):
#                 att = self.w.blocks[i].att
#                 x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
#                     att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay,
#                     att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
#                 ffn = self.w.blocks[i].ffn
#                 x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i,
#                     ffn.time_mix_k, ffn.time_mix_r,
#                     ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

#             x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
#             return x.float(), state

##########################################################################################################



########################################################################################################




























# tokenizer = TRIE_TOKENIZER('/home/neromous/.minicoda3/envs/blackfog/lib/python3.9/site-packages/rwkv/rwkv_vocab_v20230424.txt')
# class Inference:
#     @classmethod
#     def new(cls, args,model_dict):
#         m = cls()
#         args.model_dict= {}
#         for k, v in model_dict.items():
#             args.model_dict[k] = v.cpu()
#         m.model = RWKV_RNN(args)
#         return m

#     def unload(self):
#         del self.model
#         return "success"

#     @classmethod
#     def decode(cls,text):
#         return tokenizer.decode(text)

#     @classmethod
#     def encode(cls,tokens):
#         return tokenizer.encode(tokens)

#     @classmethod
#     def sample_logits(cls, logits, temperature=1.0, top_p=0.85, top_k=0):
#         probs = F.softmax(logits.float(), dim=-1)
#         top_k = int(top_k)
#         if probs.device == torch.device('cpu'):
#             probs = probs.numpy()
#             sorted_ids = np.argsort(probs)
#             sorted_probs = probs[sorted_ids][::-1]
#             cumulative_probs = np.cumsum(sorted_probs)
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if top_k < len(probs) and top_k > 0:
#                 probs[sorted_ids[:-top_k]] = 0
#             if temperature != 1.0:
#                 probs = probs ** (1.0 / temperature)
#             probs = probs / np.sum(probs)
#             out = np.random.choice(a=len(probs), p=probs)
#             return int(out)
#         else:
#             sorted_ids = torch.argsort(probs)
#             sorted_probs = probs[sorted_ids]
#             sorted_probs = torch.flip(sorted_probs, dims=(0,))
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if top_k < len(probs) and top_k > 0:
#                 probs[sorted_ids[:-top_k]] = 0
#             if temperature != 1.0:
#                 probs = probs ** (1.0 / temperature)
#             out = torch.multinomial(probs, num_samples=1)[0]
#             return int(out)


#     def generate(self, message, callback=None, state=None):
#         all_tokens = []
#         out_last = 0
#         out_str = ''
#         token_count =  message.token_count
#         occurrence = {}
#         if token_count == 0:
#             tokens = self.encode(message.content)
#             while len(tokens) > 0:
#                 out, state = self.model.forward(tokens[:message.chunk_len], state)
#                 tokens = tokens[message.chunk_len:]
#             return out,state
#         else:
#             for i in range(token_count):
#                 # forward & adjust prob.
#                 tokens = self.encode(message.content)  if i == 0 else [token]
#                 while len(tokens) > 0:
#                     out, state = self.model.forward(tokens[:message.chunk_len], state)
#                     tokens = tokens[message.chunk_len:]

#                 for n in message.token_ban:
#                     out[n] = -float('inf')
#                 for n in occurrence:
#                     out[n] -= (message.alpha_presence + occurrence[n] * message.alpha_frequency)

#                 # sampler
#                 token = self.sample_logits(out,
#                                            temperature=message.temperature,
#                                            top_p=message.top_p,
#                                            top_k=message.top_k)
#                 if token in message.token_stop:
#                     break
#                 all_tokens += [token]
#                 for xxx in occurrence:
#                     occurrence[xxx] *= message.alpha_decay
#                 if token not in occurrence:
#                     occurrence[token] = 1
#                 else:
#                     occurrence[token] += 1
#                 # print(occurrence) # debug

#                 # output
#                 tmp = self.decode(all_tokens[out_last:])
#                 if '\ufffd' not in tmp: # is valid utf-8 string?
#                     if callback:
#                         callback(tmp)
#                     out_str += tmp
#                     out_last = i + 1
#                 # if out_str.endswith(message.str_stop):
#                 #     break
#             # result = { "role" :    message.role,
#             #            "action" :  message.action,
#             #            "prompt" :  message.content,
#             #            "response": out_str}
#             return out_str




# from pydantic import BaseModel
# from typing import List, Union
# from rwkv.model import RWKV
# import gc
# import torch
# import copy
# import os, sys
# import numpy as np
# import torch
# from torch.nn import functional as F
# from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
# import time


# model_register = {"chntune": ["./models/rwkv-16.pth", "cuda:1 fp16"],
#                   "3b-clojure-1.5": ["../../models/RWKV-novel-4-World-7B-20230810-ctx128k.pth",
#                                      "cuda:1 fp16"],
#                   "3b-clojure-19": ["../../models/rwkv-19.pth", "cuda:2 fp16"]}


# class DrawPrompt(BaseModel):
#     file_path: str = "robot"
#     prompt:    str



# class LLM_Model(BaseModel):
#     name: str = "3b-clojure-1.5"


# class Message(BaseModel):
#     role: str = "robot"
#     action: str = "request"
#     # prefix: str = ""
#     # postfix: str = ""
#     content: str = ""
#     temperature: Union[float, None] = 1.1
#     top_p: Union[float, None] = 0.1
#     top_k: Union[float, None] = 0
#     alpha_frequency: Union[float, None] = 0.4
#     alpha_presence: Union[float, None] = 0.4
#     alpha_decay: Union[float, None] = 0.996 # gradually decay the penalty
#     token_ban: List[int] = []
#     token_stop: List[int] = [0,65535]
#     chunk_len: Union[int, None] = 128
#     token_count: Union[int, None] = 0
#     over: Union[bool,None] = True


# class Messages(BaseModel):
#     with_model: str = "3b-clojure-1.5"
#     messages: List[Message] = []

# # class PIPELINE_ARGS():
# #     def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[0,65535], chunk_len=256,str_stop="\n\n"):
# #         self.temperature = temperature
# #         self.top_p = top_p
# #         self.top_k = top_k
# #         self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
# #         self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
# #         self.alpha_decay = alpha_decay # gradually decay the penalty
# #         self.token_ban = token_ban # ban the generation of some tokens
# #         self.token_stop = token_stop # stop generation whenever you see any token here
# #         self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)
# #         self.str_stop = str_stop

# config = {"me":              [65530],    # robot
#           "read-system-doc": [65531],    # robot 阅读系统文档
#           "read-user-req":   [65532],    # robot 阅读user请求
#           "think":           [65533],    # 思考
#           "response":        [65534],    # 从屏幕输出结果
#           "over":            [65535]     # 动作结束
#           }

# me      = config['me']
# system  = config['read-system-doc']
# request = config['read-user-req']
# think   = config['think']
# response = config['response']
# over    = config['over']


# class Service:
#     def __init__(self, model_name):
#         self.name = model_name
#         self.path = model_register[model_name][0]
#         self.strategy = model_register[model_name][1]
#         self.init_role   = "system"
#         self.init_action = "set-system"
#         self.init_prompt = "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can."
#         self.init_state= None
#         self.model = None
#         self.pipeline = None
#         self.all_state = {}
#         self.tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
#         self.history = []
#         self.log_path = f"./history/{self.name}.jsonl"
#         self.srv_name = "completion"
#         self.state = {self.srv_name : None, 'default' : None}

#     def load(self):
#         if self.model is None:
#             self.model = RWKV(model=self.path, strategy=self.strategy)
#         self.load_prompt()
#         gc.collect()
#         torch.cuda.empty_cache()
#         return self

#     def load_prompt(self):
#         message = Message()
#         message.role =  self.init_role
#         message.action = self.init_action
#         message.content =  self.init_prompt
#         message.token_count = 0
#         messages = Messages()
#         messages.messages = [message]
#         self.inference(messages)
#         gc.collect()
#         torch.cuda.empty_cache()
#         return self


#     def reset(self):
#         self.save_state(self.srv_name, self.state['default'])
#         self.history = []
#         return {"response": "reset"}

#     def save_state(self,state_id,state):
#         self.state[state_id] = copy.deepcopy(state)
#         return state


#     def load_state(self,state_id):
#         state = copy.deepcopy(self.state[state_id])
#         return state

#     def refine_context(self, context):
#         context = context.split('\n')
#         for c in range(len(context)):
#             context[c] = context[c].strip().strip('\u3000').strip('\r')
#         context = list(filter(lambda c: c != '', context))
#         context = '\n' + ('\n'.join(context)).strip()
#         if context == '':
#             context = '\n'
#         return context

#     def encode(self, x):
#         return self.tokenizer.encode(x)

#     def decode(self, x):
#         return self.tokenizer.decode(x)

#     def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
#         probs = F.softmax(logits.float(), dim=-1)
#         top_k = int(top_k)
#         if probs.device == torch.device('cpu'):
#             probs = probs.numpy()
#             sorted_ids = np.argsort(probs)
#             sorted_probs = probs[sorted_ids][::-1]
#             cumulative_probs = np.cumsum(sorted_probs)
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if top_k < len(probs) and top_k > 0:
#                 probs[sorted_ids[:-top_k]] = 0
#             if temperature != 1.0:
#                 probs = probs ** (1.0 / temperature)
#             probs = probs / np.sum(probs)
#             out = np.random.choice(a=len(probs), p=probs)
#             return int(out)
#         else:
#             sorted_ids = torch.argsort(probs)
#             sorted_probs = probs[sorted_ids]
#             sorted_probs = torch.flip(sorted_probs, dims=(0,))
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if top_k < len(probs) and top_k > 0:
#                 probs[sorted_ids[:-top_k]] = 0
#             if temperature != 1.0:
#                 probs = probs ** (1.0 / temperature)
#             out = torch.multinomial(probs, num_samples=1)[0]
#             return int(out)

#     def fix_token(self,message):
#         # print("=====",message)
#         action = message.action
#         content = message.content
#         #prefix = message.prefix
#         text = content
#         text = self.refine_context(text)
#         if message.over:
#             over = [65535]
#         else:
#             over = []
#         # text = self.tokenizer.encode(text)
#         if action == "system":
#             output = me + system + self.tokenizer.encode(text) + over
#         elif action == "input":
#             output = me + request + self.tokenizer.encode(text) + over
#         elif action == "output":
#             output = me + response + self.tokenizer.encode(text) +over
#         elif action == "think":
#             output = me + think + self.tokenizer.encode(text) +over
#         else:
#             output = self.tokenizer.encode(action + ': ' +  text + "\n\n")
#         # print("=====",output)
#         return output

#     def my_print(self,s):
#         print(s, end='', flush=True)
#         #pass

#     def generate(self,message, token_count=100, callback=None, state=None):
#         args = message
#         all_tokens = []
#         out_last = 0
#         out_str = ''
#         occurrence = {}
#         if token_count == 0:
#             tokens = self.fix_token(message)
#             while len(tokens) > 0:
#                 out, state = self.model.forward(tokens[:args.chunk_len], state)
#                 tokens = tokens[args.chunk_len:]
#             result = { "role" :    args.role,
#                        "action" :  args.action,
#                        "prompt" :  args.content,
#                        "response": "",}
#             return result,state
#         else:
#             for i in range(token_count):
#                 # forward & adjust prob.
#                 tokens = self.fix_token(message)  if i == 0 else [token]
#                 while len(tokens) > 0:
#                     out, state = self.model.forward(tokens[:args.chunk_len], state)
#                     tokens = tokens[args.chunk_len:]

#                 for n in args.token_ban:
#                     out[n] = -float('inf')
#                 for n in occurrence:
#                     out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

#                 # sampler
#                 token = self.sample_logits(out,
#                                            temperature=args.temperature,
#                                            top_p=args.top_p,
#                                            top_k=args.top_k)
#                 if token in args.token_stop:
#                     break
#                 all_tokens += [token]
#                 for xxx in occurrence:
#                     occurrence[xxx] *= args.alpha_decay
#                 if token not in occurrence:
#                     occurrence[token] = 1
#                 else:
#                     occurrence[token] += 1
#                 # print(occurrence) # debug

#                 # output
#                 tmp = self.decode(all_tokens[out_last:])
#                 if '\ufffd' not in tmp: # is valid utf-8 string?
#                     if callback:
#                         callback(tmp)
#                     out_str += tmp
#                     out_last = i + 1
#                 # if out_str.endswith(args.str_stop):
#                 #     break
#             result = { "role" :    args.role,
#                        "action" :  args.action,
#                        "prompt" :  message.content,
#                        "response": out_str}
#             return result, state


#     def inference(self, messages:Messages):
#         results = []
#         for message in messages.messages:
#             # print(message.__dict__)
#             print(f'{message.role}: {message.content}')
#             #self.history.append(message)
#             if message.role != "system":
#                 result,state = self.generate(message=message,
#                                              token_count=message.token_count,
#                                              callback=self.my_print,
#                                              state = self.load_state(self.srv_name))
#                 results.append(result)
#                 state = self.save_state(self.srv_name, state)
#             else:
#                 action = message.action
#                 content = message.content
#                 if action == "reset":
#                     results.append(self.reset())
#                 else:
#                     #action == "set-system":
#                     result,state = self.generate(message=message,
#                                                  token_count=message.token_count,
#                                                  callback=self.my_print,
#                                                  state = self.load_state(self.srv_name))
#                     results.append(result)
#                     self.save_state('default', state)
#                     self.save_state(self.srv_name, state)

#             #print("---after---",self.state[self.srv_name][0:5])
#             #print("---before---",self.state['default'][0:5])


#         self.history =  self.history +  results
#         return results
