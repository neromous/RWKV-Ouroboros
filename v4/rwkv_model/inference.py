import torch
from torch.nn import functional as F
import numpy as np
import gc
from rwkv_model.model_infer import RWKV_RNN
from models.message import Message
from models.scene import Scene
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import copy
from tqdm import tqdm
tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
from utils import log, load_config
config = load_config()
import types

def my_func(tmp):
    print(tmp, end="", flush=True)

class Inference:
    def __init__(self,
                 model_name: str,
                 model_weights=None):
        self.model_weights = model_weights
        model_keys = list(model_weights.keys())
        # Lets compute the model various sizes, if they are not provided
        max_block_id = 0
        for x in model_keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        self.n_layer = max_block_id + 1
        self.n_embd = model_weights['head.weight'].shape[1]
        self.model_name = model_name
        self.model = None
        self.scene = Scene.new({"title":"测试"})
        self.state = None
        self.init_state = None

    def load_model(self):
        torch.cuda.empty_cache()
        gc.collect()
        args = types.SimpleNamespace()
        args.n_layer= self.n_layer
        args.n_embd = self.n_embd
        self.model = RWKV_RNN(self.weights,
                              args)
        return self

    def clean_model(self):
        self.model = None
        self.model_weights = None
        torch.cuda.empty_cache()
        gc.collect()
        return self

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

    @classmethod
    def sample_logits(cls, logits:torch.tensor, temperature=0.1, top_p=0.1, top_k=0):
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

    def generate(self,
                 message:Message,
                 callback=my_func,
                 state = None):
        tokens = message.tokens
        token_count = message.token_count
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
            logits , self.state = self.model(token, self.state)
        for i in range(0,token_count):
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
            logits,self.state = self.model(token, self.state)
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
