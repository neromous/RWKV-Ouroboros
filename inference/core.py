import torch
from torch.nn import functional as F
import numpy as np
import gc
from rwkv_model.model_infer import RWKV_RNN
from models.message import Message
from models.scene import Scene
import copy
import types
from tqdm import tqdm
from utils import log, load_config

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

config = load_config()



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


def my_func(tmp):
    print(tmp, end="", flush=True)

class InferenceWithState:
    def __init__(self,
                 scene_name = "default",
                 sampler_fn=sample_logits,
                 n_layer = None,
                 ctx_len = None,
                 n_embd = None):
        self.ctx_len = ctx_len or config['model']['ctx_len']
        self.n_layer = n_layer or config['model']['n_layer']
        self.n_embd = n_embd or config['model']['n_embd']
        self.model = None
        self.scene = Scene.new({"title": scene_name})
        self.state = None
        self.init_state = None
        self.sampler_fn = sampler_fn

    def set_init_state(self):
        self.init_state = copy.deepcopy(self.state)
        torch.cuda.empty_cache()
        gc.collect()
        return self

    def reset_state(self):
        self.state = copy.deepcopy(self.init_state)
        torch.cuda.empty_cache()
        gc.collect()
        return self

    @classmethod
    def get_special_token(cls, role, pos):
        r = config["role"]
        return r[role][pos]

    @classmethod
    def is_valid_token(cls, token):
        if 65535 >= token and token >= 0:
            return True
        else:
            return False

    @classmethod
    def is_valid_role(cls, role):
        return role in config['role'].keys()

    @classmethod
    def encode(cls, text) -> list:
        return tokenizer.encode(text)

    @classmethod
    def decode(cls, tokens) -> str:
        tokens = [x for x in tokens if cls.is_valid_token(x)]
        return tokenizer.decode(tokens)

    def generate(self,model,message:Message,callback=my_func):
        tokens = message.to_tokens()
        token_count = message.token_count
        token_ban = message.token_ban
        token_stop = message.token_stop
        temperature =  message.temperature
        top_p = message.top_p
        alpha_presence = message.alpha_presence
        alpha_frequency = message.alpha_frequency
        alpha_decay = message.alpha_decay
        out_str = ""
        occurrence = {}
        logits= []
        all_tokens = []
        out_last = 0
        for token in tokens:
            logits , self.state = model(token, self.state)
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
            logits, self.state = model(token, self.state)
        message.generated =  True
        message.response = out_str
        return message

    def generate_no_state(self, logits_fn,message:Message,callback=my_func):
        tokens = message.tokens
        token_count = message.token_count
        temperature =  message.temperature
        token_ban = message.token_ban
        token_stop = message.token_stop
        top_p = message.top_p
        alpha_presence = message.alpha_presence
        alpha_frequency = message.alpha_frequency
        alpha_decay = message.alpha_decay
        out_str = ""
        occurrence = {}
        logits= []
        ctx_len = self.ctx_len
        all_tokens = tokens[token_count - ctx_len:]
        out_last = 0
        for i in range(0,token_count):
            logits = model.inference(all_tokens[-ctx_len:])
            for n in token_ban:
                logits[n] = -float('inf')
            for n in occurrence:
                logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

            token = RWKV.sample_logits(logits,
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

        message.generated =  True
        message.response = out_str
        return message

    def finish(self):
        messages = []
        for message in self.messages:
            if message.generated:
                messages.append(message)
            else:
                message = self.generate(message)
                message.save()
                self.scene.add_message(message)
                messages.append(message)
        self.message = messages
        return self
