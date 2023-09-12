import os, sys
import numpy as np
import torch
from torch.nn import functional as F
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

class Tokenizer:
    def __init__(self):
        self.tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

    def encode(self, x):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
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

if __name__ == "__main__":
    tokenizer =  Tokenizer()
    print("=====",tokenizer.encode("你好啊,小猪佩奇"))
    print("=====",tokenizer.encode(tokenizer.decode(tokenizer.encode("你好啊,小猪佩奇"))))
    print("=====",tokenizer.decode(tokenizer.encode("你好啊,小猪佩奇")))
    print(tokenizer.decode([10293, 16503, 16747]))
