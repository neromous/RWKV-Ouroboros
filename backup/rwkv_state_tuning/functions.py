import torch
from torch.nn import functional as F
import numpy as np
from .model import RWKV, L2Wrap
import copy
import sys


def ppl(tokens, logits):
    with torch.no_grad():
        cel = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.tensor([tokens], device=logits.device),
            reduction="none",
        )
        ppl = torch.exp(cel).item()
        return ppl


def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ["cpu", "privateuseone"]:
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


def speak_next_token(
    rwkv,
    last_token,
    states,
    occurrence,
    temperature,
    top_p,
    alpha_presence,
    alpha_frequency,
    use_pos_cfg,
    use_neg_cfg,
    pos_alpha,
    neg_alpha,
    pos_cfg_state=None,
    neg_cfg_state=None,
):

    # idx = torch.tensor([[last_token]], dtype=torch.long).to(
    #     next(rwkv.parameters()).device
    # )
    idx = [last_token]
    # x = rwkv.emb(idx)
    # states, logits = infer_sequence(rwkv, x, states)

    logits, states = rwkv(idx, states)
    if use_pos_cfg:
        if pos_cfg_state is None:
            new_states_pos = []
        else:
            new_states_pos = pos_cfg_state
        # x_pos = copy.deepcopy(x.detach())
        logits_pos, new_states_pos = rwkv(idx, new_states_pos)
    else:
        new_states_pos = None
    if use_neg_cfg:
        if neg_cfg_state is None:
            new_states_neg = []
        else:
            new_states_neg = neg_cfg_state
        # x_neg = copy.deepcopy(x.deta~ch())
        logits_neg, new_states_neg = rwkv(idx, new_states_neg)
    else:
        new_states_neg = None

    if use_pos_cfg and use_neg_cfg:
        logits = (
            logits * (1 - pos_alpha + neg_alpha)
            + logits_pos * pos_alpha
            - logits_neg * neg_alpha
        )
    elif use_pos_cfg:
        logits = logits * (1 - pos_alpha) + logits_pos * pos_alpha
    elif use_neg_cfg:
        logits = logits * (1 + neg_alpha) - logits_neg * neg_alpha
    for n in occurrence:
        logits[0, -1, n] -= alpha_presence + occurrence[n] * alpha_frequency
    next_token = sample_logits(logits[:, -1, :].squeeze(), temperature, top_p)
    # if next_token not in occurrence:
    #     occurrence[next_token] = 1
    # else:
    #     occurrence[next_token] += 1
    # speak_sequence.append(next_token)
    # if debug_mode:
    #     xxx = tokenizer.decode([next_token])
    #     print(xxx, end="")
    #     sys.stdout.flush()
    return next_token, states, new_states_pos, new_states_neg


def speak(
    rwkv,
    start_with_token: int,
    states,
    temperature,
    top_p,
    alpha_presence,
    alpha_frequency,
    tokenizer,
    token_stop=[65535],
    max_tokens=100,
    m_postfix_token=[],
    debug_mode=False,
    use_pos_cfg=False,
    use_neg_cfg=False,
    pos_alpha=0.4,
    neg_alpha=0.4,
):
    with torch.no_grad():
        rwkv.eval()
        speak_sequence = []
        next_token = start_with_token
        args = rwkv.args
        B, T = 1, 1
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert C == H * args.head_size

        if states is None:
            new_states = None
        else:
            new_states = copy.deepcopy(states)

        occurrence = {}
        for i in range(max_tokens):
            next_token, new_states, _, _ = speak_next_token(
                rwkv,
                next_token,
                new_states,
                occurrence,
                temperature,
                top_p,
                alpha_presence,
                alpha_frequency,
                use_pos_cfg,
                use_neg_cfg,
                pos_alpha,
                neg_alpha,
            )
            if 49 <= next_token <= 58:
                pass
            elif next_token not in occurrence:
                occurrence[next_token] = 1
            else:
                occurrence[next_token] += 1
            speak_sequence.append(next_token)
            if debug_mode:
                xxx = tokenizer.decode([next_token])
                print(xxx, end="")
                sys.stdout.flush()

            if next_token in token_stop:
                if m_postfix_token:
                    postfix = torch.tensor([[m_postfix_token]], dtype=torch.long).to(
                        next(rwkv.parameters()).device
                    )
                    logits, new_states = rwkv(postfix, new_states)
                speak_sequence.pop()
                break
        return speak_sequence, new_states


def train_forward(rwkv, batch: dict, states=None):
    # pre calc
    seq = batch["input_ids"]
    mask = batch.get("masks", None)

    # data process
    idx = seq[:-1]
    targets = seq[1:]
    if mask == None:
        mask = [int(x != 0) for x in idx]

    # data into tensor
    targets = torch.tensor([targets], dtype=torch.long).to(
        next(rwkv.parameters()).device
    )

    # process mask
    mask = torch.tensor([mask], dtype=torch.float32).to(next(rwkv.parameters()).device)
    mask = mask.view(-1)
    sum_mask = torch.sum(mask).item()
    logits, new_states = rwkv(idx, states)

    if sum_mask == mask.shape[0]:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # print('rank', rwkv.global_rank, 'loss', loss.item())
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        # loss_raw = loss
        loss = torch.sum(loss * mask)
        if sum_mask > 0:
            loss = loss / sum_mask
    return L2Wrap.apply(loss, logits), new_states
