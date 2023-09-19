import torch
from torch.nn import functional as F
import numpy as np
import gc
from rwkv_model.model_run import RWKV_RNN
from models.message import Message
from models.scene import Scene
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import copy
from tqdm import tqdm
tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
from utils import log, load_config
config = load_config()


def my_func(tmp):
    print(tmp, end="", flush=True)

class Inference:
    def __init__(self,
                 model_name: str,
                 model_weights=None):
        self.model_weights = model_weights
        self.model_name = model_name
        self.model = None
        self.scene = Scene.new({"title":"测试"})
        self.state = None
        self.init_state = None

    def load_model(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.model = RWKV_RNN(self.model_name,model_weights=self.model_weights,float_type="bf16")
        return self

    def clean_model(self):
        self.model = None
        self.model_weights = None
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
        all_tokens = []
        out_last = 0
        out_str = ''
        token_count =  message.token_count
        occurrence = {}
        if token_count == 0:
            tokens = message.to_tokens()
            print(tokens)
            while len(tokens) > 0:
                #out, state = self.model.forward(tokens[0], state)
                #tokens = tokens[1:]
                out, state = self.model.forward(tokens[:message.chunk_len], state)
                tokens = tokens[message.chunk_len:]
            if self.init_state == None:
                self.init_state = copy.deepcopy(state)
            self.state = copy.deepcopy(state)
            message.generated = True
            message.response = ""
            return message
        else:
            token = False
            for i in range(token_count):
                tokens = message.to_tokens()  if i == 0 else [token]
                # forward & adjust prob.
                while len(tokens) > 0:
                    out, state = self.model.forward(tokens[:message.chunk_len], state)
                    tokens = tokens[message.chunk_len:]
                    #out, state = self.model.forward(tokens[0], state)
                    #tokens = tokens[1:]

                for n in message.token_ban:
                    out[n] = -float('inf')
                for n in occurrence:
                    out[n] -= (message.alpha_presence + occurrence[n] * message.alpha_frequency)

                # sampler
                token = self.sample_logits(out,
                                            temperature=message.temperature,
                                            top_p=message.top_p,
                                            top_k=message.top_k)

                if token in message.token_stop:
                    break
                all_tokens += [token]
                for xxx in occurrence:
                    occurrence[xxx] *= message.alpha_decay
                if token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1
                # output
                tmp = tokenizer.decode(all_tokens[out_last:])
                if '\ufffd' not in tmp: # is valid utf-8 string?
                    if callback:
                        callback(tmp)
                    out_str += tmp
                    out_last = i + 1
            # 保存state 进行初始化
            if self.init_state == None:
                self.init_state = copy.deepcopy(state)
            # 保存state到env
            self.state = copy.deepcopy(state)
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
