########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from typing import Optional
import types, gc, os, time, re
import torch
from torch.nn import functional as F
import numpy as np
import copy
from tqdm import tqdm
########################################################################################################

MyModule = torch.nn.Module
def __nop(ob):
    return ob
MyFunction = __nop
MyStatic = __nop

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    # 'privateuseone' is the type of custom devices like `torch_directml.device()`
    if probs.device.type in ['cpu', 'privateuseone']:
        probs = probs.cpu().numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
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
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)

########################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, model_weight, args):
        super().__init__()
        self.args = copy.deepcopy(args)
        if args.dtype == "fp32":
            args.dtype = torch.float
        elif args.dtype == "fp16":
            args.dtype = torch.half
        elif args.dtype == "bf16":
            args.dtype = torch.bfloat16

        args = self.args

        with torch.no_grad():
            self.w = model_weight # load model to CPU first
            gc.collect()
            w = self.w
            args.n_embd = w['emb.weight'].shape[1]
            args.n_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix

            args.n_layer = 0
            keys = list(w.keys())
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, layer_id+1)
                if 'att.time_decay' in x:
                    args.n_head = w[x].shape[0]

            for x in tqdm(keys):
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'gate.weight' in x or 'output.weight' in x or 'head.weight' in x:
                        w[x] = w[x].t()
                if '.time_decay' in x: # need fp32 for this
                    w[x] = torch.exp(-torch.exp(w[x].float())).reshape(-1,1,1)
                    w[x] = w[x].reshape(args.n_head, -1, 1)
                elif '.ln_x' in x: # need fp32 for group_norm
                    w[x] = w[x].float()
                elif '.time_faaaa' in x: # need fp32 for this
                    w[x] = w[x].float().reshape(-1,1,1).reshape(args.n_head, -1, 1)
                else:
                    w[x] = w[x].to(dtype=self.args.dtype)
            w['emb.weight'] = F.layer_norm(w['emb.weight'],
                                           (args.n_embd,),
                                           weight=w['blocks.0.ln0.weight'],
                                           bias=w['blocks.0.ln0.bias'])

    ########################################################################################################

    @MyFunction
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(rx @ rw)
        vx = torch.square(torch.relu(kx @ kw))
        out = r * (vx @ vw)
        x = x + out
        return x, xx

    ########################################################################################################

    @MyFunction
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r =  torch.sigmoid(rx @ rw)
        vx = torch.square(torch.relu(kx @ kw))
        out = r * (vx @ vw)
        x = x + out
        return x, xx[-1,:]

    ########################################################################################################

    ########################################################################################################

    @MyFunction
    def att_one(self, x, sx, s, ln_w, ln_b, lx_w, lx_b,
                     k_mix, v_mix, r_mix, g_mix, t_decay, t_first,
                     kw, vw, rw, gw, ow):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        S = x.shape[-1] // H

        r = (rx @ rw).to(dtype=torch.float32).view(H, 1, S)
        k = (kx @ kw).to(dtype=torch.float32).view(H, S, 1)
        v = (vx @ vw).to(dtype=torch.float32).view(H, 1, S)
        g = F.silu(gx @ gw)

        a = k @ v

        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0),
                           num_groups=H,
                           weight=lx_w,
                           bias=lx_b).squeeze(0)
        out = out.to(dtype=self.args.dtype) * g
        out = out @ ow
        x = x + out
        return x, xx, s


    @MyFunction
    def att_seq(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix,
                     v_mix, r_mix, g_mix, t_decay, t_first,
                     kw, vw, rw, gw, ow):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        S = x.shape[-1] // H
        T = x.shape[0]

        r = (rx @ rw).to(dtype=torch.float32).view(T, H, S).transpose(0, 1)
        k = (kx @ kw).to(dtype=torch.float32).view(T, H, S).transpose(0, 1).transpose(-2, -1)
        v = (vx @ vw).to(dtype=torch.float32).view(T, H, S).transpose(0, 1)
        g = F.silu(gx @ gw)

        out = torch.empty((T, H, S),dtype=r.dtype).cuda()

        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = kt @ vt
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H*S)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b)
        out = out.to(dtype=self.args.dtype) * g
        out = out @ ow
        x = x + out
        return x, xx[-1,:], s

    ########################################################################################################



    ########################################################################################################

    def forward(self, tokens, state, full_output=False):
        with torch.no_grad():
            w = self.w
            args = self.args
            if state == None:
                state = [None] * args.n_layer * 3
                for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
                    state[i*3+0] = torch.zeros(args.n_embd,
                                               dtype=self.args.dtype,
                                               requires_grad=False).cuda()
                    state[i*3+1] = torch.zeros((args.n_head,
                                                args.n_att//args.n_head,
                                                args.n_att//args.n_head),
                                               dtype=torch.float32,
                                               requires_grad=False).cuda()
                    state[i*3+2] = torch.zeros(args.n_embd,
                                               dtype=self.args.dtype,
                                               requires_grad=False).cuda()
            seq_mode = len(tokens) > 1

            x = w['emb.weight'][tokens if seq_mode else tokens[0]]

            if seq_mode:
                ATT = self.att_seq
                FFN = self.ffn_seq
            else:
                ATT = self.att_one
                FFN = self.ffn_one

            for i in range(args.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                x = x.to(dtype=self.args.dtype)
                x, state[i*3+0], state[i*3+1] = ATT(
                    x, state[i*3+0], state[i*3+1],
                    w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
                    w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
                    w[f'{att}time_mix_k'], w[f'{att}time_mix_v'],
                    w[f'{att}time_mix_r'], w[f'{att}time_mix_g'],
                    w[f'{att}time_decay'], w[f'{att}time_faaaa'],
                    w[f'{att}key.weight'], w[f'{att}value.weight'],
                    w[f'{att}receptance.weight'], w[f'{att}gate.weight'],
                    w[f'{att}output.weight']
                )

                x, state[i*3+2] = FFN(
                    x, state[i*3+2],
                    w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
                    w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
                    w[f'{ffn}key.weight'], w[f'{ffn}value.weight'],
                    w[f'{ffn}receptance.weight']
                )

            x = x[-1,:] if (seq_mode and (not full_output)) else x

            x = x.to(dtype=self.args.dtype)

            x = F.layer_norm(x, (args.n_embd,), weight=w['ln_out.weight'], bias=w['ln_out.bias'])

            x = x @ w['head.weight']

            return x.float(), state

    def generate(self,
                 tokenizer,
                 message,
                 inference_config,
                 callback=print,
                 state=None):

        tokens, masks = message.tokens()
        token_count = inference_config['token_count']
        token_ban = inference_config['token_ban']
        token_stop = inference_config['token_stop']
        temperature =  inference_config['temperature']
        top_p = inference_config['top_p']
        alpha_presence = inference_config['alpha_presence']
        alpha_frequency = inference_config['alpha_frequency']
        alpha_decay = inference_config['alpha_decay']
        out_str = ""
        occurrence = {}
        all_tokens = []
        out_last = 0
        while len(tokens) > 0:
            do_infer = tokens[:512]
            tokens = tokens[512:]
            logits, state = self.forward(do_infer, state)
        for i in range(0,token_count):
            for n in token_ban:
                logits[n] = -float('inf')
            for n in occurrence:
                logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

            token = sample_logits(logits,
                                  temperature=temperature,
                                  top_p=top_p)
            if token in token_stop:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= alpha_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            text = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in text: # only print when we have a valid utf-8 string
                print(text, end="", flush=True)
                out_str += text
                out_last = i + 1
            logits, state = self.forward([token], state)
        message.generated = True
        message.response = out_str
        return message, state

# 流式输出
    def flow_generate(self,
                 tokenizer,
                 message,
                 inference_config,
                 callback=print,
                 state=None):

        tokens, masks = message.tokens()
        token_count = inference_config['token_count']
        token_ban = inference_config['token_ban']
        token_stop = inference_config['token_stop']
        temperature =  inference_config['temperature']
        top_p = inference_config['top_p']
        alpha_presence = inference_config['alpha_presence']
        alpha_frequency = inference_config['alpha_frequency']
        alpha_decay = inference_config['alpha_decay']
        out_str = ""
        occurrence = {}
        all_tokens = []
        out_last = 0
        while len(tokens) > 0:
            do_infer = tokens[:512]
            tokens = tokens[512:]
            logits, state = self.forward(do_infer, state)
        for i in range(0,token_count):
            for n in token_ban:
                logits[n] = -float('inf')
            for n in occurrence:
                logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

            token = sample_logits(logits,
                                  temperature=temperature,
                                  top_p=top_p)
            if token in token_stop:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= alpha_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            text = tokenizer.decode(all_tokens[out_last:])
            logits, state = self.forward([token], state)
            if '\ufffd' not in text: # only print when we have a valid utf-8 string
                print(text, end="", flush=True)
                if text:
                    # 流式输出
                    yield text
                out_last = i + 1
        return state