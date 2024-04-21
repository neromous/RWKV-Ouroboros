from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import gc
import deepspeed
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import copy
from .rwkv_inner import rwkv_inner

def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

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



def rwkv_recurrent(r_in, k_in, v_in, w_in, u, state):
    L = r_in.size(-2)
    out = []
    for t in range(L):
        r, k, v, w = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        kv = k.mT @ v # KV
        out.append( r @ (state + u.mT * kv) ) # 1K @ (KV + 1)
        state = (w.mT * state) + kv # KV
    out = torch.cat(out, dim=-2)
    return out, state

class TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        # init
        self.n_kv_head = args.n_kv_head
        self.r_head_size = args.dim_rk // args.n_head
        self.k_head_size = args.dim_rk // args.n_head
        self.v_head_size = args.dim_v // args.n_head
        #
        assert args.dim_rk % self.n_head == 0
        assert args.dim_rk % self.n_kv_head == 0
        assert args.dim_v % self.n_kv_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)


    def forward(self, xq: Tensor, state: Optional[Tensor] = None, for_infer=False):
        x = xq # FIXME - support encoder-decoder models

        H = self.n_head
        KVH = self.n_kv_head
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce kx, vx, rx, gx
        kx = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        vx = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        rx = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        gx = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        # Mix kx, vx with the previous timestep in a learned manner to produce new time-mixed xk, xv
        #kx = self.time_mixer_k(kx)
        #vx = self.time_mixer_v(vx)
        #rx = gx = x

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(gx))

        #r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH,K)
        time_faaaa = self.time_faaaa.float() # (KVH,K)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
            time_faaaa = time_faaaa.expand(reps, KVH, K).contiguous().view(H, K)

        state = state
        if state is None:
            state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if state.dtype != r.dtype:
            state = state.contiguous()

        w = torch.exp(-torch.exp(time_decay)).view(1,H,1,K).expand(1,H,T,K)
        u = time_faaaa.float().view(1,H,1,K)

        r = r.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        w = w.float().contiguous()
        u = u.float().contiguous()
        state = state.float().contiguous()

        if for_infer:
            out, state = rwkv_recurrent(r, k, v, w, u, state)
        else:
            out, state = rwkv_inner(r, k, v, w, u, state)
        # r = r.to(dtype=self.args.dtype)
        # k = k.to(dtype=self.args.dtype)
        # v = v.to(dtype=self.args.dtype)
        # w = w.to(dtype=self.args.dtype)
        # u = u.to(dtype=self.args.dtype)
        #state = state.to(dtype=self.args.dtype)

        out = out.to(dtype=self.args.dtype)
        out = out.transpose(1,2).reshape(B*T, H*V)
        out = self.ln_x(out / self.args.head_size_divisor).view(B, T, H*V)

        out = self.output(out * g)
        return out, state


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
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)
        self.ffn = ChannelMix(args, layer_id)

    def forward(self, x, state, for_infer):
        if self.layer_id == 0:
            x = self.ln0(x)
        out, state = self.att(self.ln1(x), state=state, for_infer = for_infer)
        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x, state



class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        if True:
            maxx[maxx<3.]=0.
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)

        # gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dtype == "fp32":
            args.dtype=torch.float
        elif args.dtype == "fp16":
            args.dtype=torch.half
        elif args.dtype== "bf16":
            args.dtype = torch.bfloat16
        else:
            args.dtype = torch.bfloat16
        # load model
        model_weights = torch.load(args.load_model, map_location='cpu')
        model_keys = list(model_weights.keys())
        # init layer num
        max_block_id = 0
        for x in model_keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        args.n_layer = max_block_id + 1
        # init n_embd number
        args.n_embd = model_weights['head.weight'].shape[1]
        # init vocab number
        args.vocab_size = model_weights['head.weight'].shape[0]
        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        #
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
        args.tiny_att_layer = -1
        args.tiny_att_dim = -1
        self.args = args
        # init
        # args.n_kv_head = args.n_head
        # args.dim_rk = args.n_embd
        # args.dim_v = args.n_embd
        # init

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i)
                                     for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        model_weights = {k: v.to(dtype=args.dtype) for k, v
                         in model_weights.items()}

        self.load_state_dict(model_weights)
        self.to("cuda")
        print("=====", args.dtype)
        del model_weights
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, batch, states=None):
        args = self.args
        if states is None:
            pass
        else:
            state = states.to('cuda')
        # ================
        seq = batch['input_ids']
        mask = batch.get('attention_mask',None)
        idx = seq[:-1]
        targets = seq[1:]
        if mask == None:
            mask = [int(x!=0) for x in idx]
        idx = torch.tensor([idx],dtype=torch.long).to('cuda')
        targets = torch.tensor([targets],dtype=torch.long).to('cuda')
        mask = torch.tensor([mask],dtype=torch.float).to('cuda')

        # 前向 获得logits

        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        for block in self.blocks:
            if self.args.grad_cp:
                x, states = deepspeed_checkpoint(block, x, states, False)
            else:
                x, states = block(x, states, False)
        x = self.ln_out(x)
        logits = self.head(x)

        # 计算loss
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()

        if sum_mask == mask.shape[0]:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # loss_raw = loss
            loss = torch.sum(loss * mask) / sum_mask

        loss = L2Wrap.apply(loss, logits)
        states = states.detach().to('cpu')
        return loss, states


    def get_optimizers(self):
        lr_init= self.args.lr_init
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if "time_mix" in n:
                lr_1x.add(n)
            elif "time_decay" in n:
                lr_2x.add(n)
            elif "time_faaaa" in n:
                lr_2x.add(n)
            else:
                lr_1x.add(n)
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}

        if self.args.dtype == torch.float:
            optim_groups = [
                {
                    "fp32_optimizer": True,

                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "fp32_optimizer": True,
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.00,
                    "lr": 1.0 * lr_init
                },
            ]
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.args.beta1, self.args.beta2),
                                         eps=self.args.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=self.args.adamw_mode,
                                         weight_decay=self.args.weight_decay,
                                         amsgrad=False,
                                         fp32_optimizer_states=True)
        else:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.00,
                    "lr": 1.0 * lr_init
                },
            ]

            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.args.beta1, self.args.beta2),
                                         eps=self.args.adam_eps,
                                         adamw_mode=self.args.adamw_mode,
                                         weight_decay=self.args.weight_decay,
                                         amsgrad=False,
                                         bias_correction=True)
        lr_scheduler = None
        if self.args.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.args.lr_init,
                warmup_max_lr=self.args.lr_init,
                warmup_num_steps=self.args.warmup_steps,
                warmup_type='linear')
        return optimizer, lr_scheduler


    def forward_no_grad(self, tokens, states=None):
        with torch.no_grad():
            args = self.args
            if states is None:
                pass
            else:
                state = states.to('cuda')
            # ================
            idx = torch.tensor([tokens],dtype=torch.long).to('cuda')

            # 前向 获得logits

            B, T = idx.size()
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

            x = self.emb(idx)

            for block in self.blocks:
                if self.args.grad_cp:
                    x, states = deepspeed_checkpoint(block, x, states, True)
                else:
                    x, states = block(x, states, True)
            x = self.ln_out(x)
            logits = self.head(x)
            #print("==1====",logits.shape)
            logits = logits[...,-1,:]
            logits = logits.squeeze()
            #logits = logits.squeeze(1)
            #print("===2===",logits.shape)
            states = states.detach().to('cpu')
            return logits.float(), states


    def generate(self,
                 message,
                 inference_config,
                 callback=print,
                 state=None,
                 pos_state=None,
                 neg_state=None,
                 pos_gama=0.4,
                 neg_gama=0.4,
                 pos_logits=None,
                 neg_logits=None):
        tokens, masks = message.tokens()
        if False:
            print("===1==", tokens)
            print("===2==", message.text)
            print("===3==", message.tokenizer().decode(tokens))

        pos, neg = message.cfg_tokens()

        # add pos or neg tag
        if len(pos) > 0:
            pos_tag = True
        else:
            pos_tag = False

        if len(neg) > 0:
            neg_tag = True
            neg_gama = -1 * neg_gama
        else:
            neg_tag = False
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
        for i in range(0,token_count):
            if i == 0:
                while len(tokens) > 0:
                    do_infer = tokens[:512]
                    tokens = tokens[512:]
                    logits, state = self.forward_no_grad(do_infer, state)
                if pos_tag:
                    while len(pos) > 0:
                        do_infer = pos[:512]
                        pos = pos[512:]
                        pos_logits, pos_state = self.forward_no_grad(do_infer, pos_state)
                if neg_tag :
                    while len(neg) > 0:
                        do_infer = neg[:512]
                        neg = neg[512:]
                        neg_logits, neg_state = self.forward_no_grad(do_infer, neg_state)
            else:
                logits, state = self.forward_no_grad([token], state)
                if pos_tag:
                    pos_logits, pos_state = self.forward_no_grad([token], pos_state)
                if neg_tag:
                    neg_logits, neg_state = self.forward_no_grad([token], neg_state)

            if pos_tag:
                logits = pos_logits * pos_gama + logits * (1 - pos_gama)
            if neg_tag:
                logits = neg_logits * neg_gama + logits * (1 - neg_gama)

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
            text = message.tokenizer().decode(all_tokens[out_last:])
            if '\ufffd' not in text: # only print when we have a valid utf-8 string
                print(text, end="", flush=True)
                out_str += text
                out_last = i + 1

        message.generated = True
        message.response = out_str
        return message, state, pos_state, neg_state



    # def generate(self,
    #              message,
    #              inference_config,
    #              callback=print,
    #              state=None,
    #              pos_state=None,
    #              neg_state=None,
    #              pos_gama=0.4,
    #              neg_gama=0.4,
    #              pos_logits=None,
    #              neg_logits=None):

    #     tokens, masks = message.tokens()

    #     if False:
    #         print("===1==", tokens)
    #         print("===2==", message.text)
    #         print("===3==", message.tokenizer().decode(tokens))

    #     pos, neg = message.cfg_tokens()

    #     # add pos or neg tag
    #     if len(pos) > 0:
    #         pos_tag = True
    #     else:
    #         pos_tag = False

    #     if len(neg) > 0:
    #         neg_tag = True
    #         neg_gama = -1 * neg_gama
    #     else:
    #         neg_tag = False
    #     token_count = inference_config['token_count']
    #     token_ban = inference_config['token_ban']
    #     token_stop = inference_config['token_stop']
    #     temperature =  inference_config['temperature']
    #     top_p = inference_config['top_p']
    #     alpha_presence = inference_config['alpha_presence']
    #     alpha_frequency = inference_config['alpha_frequency']
    #     alpha_decay = inference_config['alpha_decay']
    #     out_str = ""
    #     occurrence = {}
    #     all_tokens = [x for x in tokens]
    #     length = len(tokens)
    #     out_last = 0
    #     origin_state = copy.deepcopy(state)
    #     for i in range(0,token_count):
    #         # if i == 0:
    #         #logits, state = self.forward_no_grad(all_tokens, state)
    #         state = copy.deepcopy(origin_state)
    #         while len(all_tokens) > 2:
    #             do_infer = all_tokens[:512]
    #             all_tokens = all_tokens[512:]
    #             logits, state = self.forward_no_grad(do_infer, state)

    #         #     if pos_tag:
    #         #         while len(pos) > 0:
    #         #             do_infer = pos[:1]
    #         #             pos = pos[1:]
    #         #             pos_logits, pos_state = self.forward_no_grad(do_infer, pos_state)
    #         #     if neg_tag :
    #         #         while len(neg) > 0:
    #         #             do_infer = neg[:1]
    #         #             neg = neg[1:]
    #         #             neg_logits, neg_state = self.forward_no_grad(do_infer, neg_state)
    #         # else:
    #         #     logits, state = self.forward_no_grad([token], state)
    #         #     if pos_tag:
    #         #         pos_logits, pos_state = self.forward_no_grad([token], pos_state)
    #         #     if neg_tag:
    #         #         neg_logits, neg_state = self.forward_no_grad([token], neg_state)
    #         # if pos_tag:
    #         #     logits = pos_logits * pos_gama + logits * (1 - pos_gama)
    #         # if neg_tag:
    #         #     logits = neg_logits * neg_gama + logits * (1 - neg_gama)

    #         for n in token_ban:
    #             logits[n] = -float('inf')
    #         for n in occurrence:
    #             logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

    #         token = sample_logits(logits,temperature=temperature,top_p=top_p)
    #         if token in token_stop:
    #             break
    #         tokens.append(token)
    #         all_tokens = [x for x in tokens]
    #         for xxx in occurrence:
    #             occurrence[xxx] *= alpha_decay
    #         if token not in occurrence:
    #             occurrence[token] = 1
    #         else:
    #             occurrence[token] += 1
    #         text = message.tokenizer().decode(tokens[length + out_last:])
    #         if '\ufffd' not in text: # only print when we have a valid utf-8 string
    #             print(text, end="", flush=True)
    #             out_str += text
    #             out_last = i + 1

    #     message.generated = True
    #     message.response = out_str
    #     return message, state, pos_state, neg_state
