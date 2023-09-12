import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from .dataloader import dataloader
import random
import copy
from src.tokenizer import Tokenizer
from .config import get_environ,get_trainer,read_config



tokenizer = Tokenizer()
role_config = read_config()['role']


class PromptDataset:
    def __init__(self,data,ctx_len=1024,prefix="",postfix=""):
        self.data = data
        self.tokens = []
        self.ctx_len = ctx_len
        self.prefix = prefix
        self.postfix = postfix

    @classmethod
    def prompt2text(cls, prompt:dict,prefix="",postfix=""):
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



class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.data = dataloader(args.data_file)
        self.data_size = len(self.data)
        self.current_idx = 0
        self.current_pos = 0
        self.item = []

    def is_end(self):
        if len(self.item) == 0:
            return True
        else:
            return False

    def add_data(self,text,role="system",in_start = True):
        role = role_config[role]
        role_prefix = role['prefix']
        role_postfix = role['postfix']
        tokens = tokenizer.encode(text)
        tokens = role_prefix + tokens + role_postfix
        tokens = np.array(tokens,dtype='uint16')

        if in_start:
            self.data = self.data.insert(0,tokens)
        else:
            self.data = self.data.append(tokens)
        return self


    @classmethod
    def prefix_tokenizer(cls, text,role="system"):
        role = role_config[role]
        role_prefix = role['prefix']
        role_postfix = role['postfix']
        tokens = tokenizer.encode(text)
        tokens = role_prefix + tokens + role_postfix
        tokens = np.array(tokens,dtype='uint16')
        return tokens


    def __len__(self):
        return len(self.data)

    # def limit_sample(self,n):
    #     if len(self.data) <= n:
    #         res = random.sample(self.data, len(self.data))
    #     else:
    #         res = random.sample(self.data, n)
    #     return res

    def __getitem__(self,
                    idx:int,
                    pos=0,
                    poe=0,
                    window=True,
                    prefix="",
                    prefix_role="system",
                    debug=False):
        args = self.args
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        # 训练数据构建
        if self.is_end():
            if idx == -1:
                self.item = random.choice(self.data)
            else:
                self.item = copy.deepcopy(self.data[idx])

        # 取内容
        if pos - poe >= ctx_len + 1:
            self.item = self.item[pos:poe]

        step  = self.item[:req_len]
        step_len =  len(step)

        # 滑窗机制 砍掉 cstx/2 
        if len(self.item) > req_len and window:
            half = int(ctx_len / 2)
            self.item = self.item[half:]
        else:
            self.item = self.item[req_len:]

        if prefix != "" and len(self.item) != 0:
            extend = self.prefix_tokenizer(prefix,role=prefix_role)
            self.item = np.concatenate((extend,self.item),axis=0)

        if debug:
            print("==step==",step)
            print("==self.item===",self.item)
            print("==self.data[0]===",self.data[0])
            print("==self.data[-1]===",self.data[-1])
        dix = [0 for x in range(req_len)]
        dix[:step_len] = step
        # 生成mask
        mask = [int(x!=0) for x in dix]
        mask = mask[:-1]
        x = torch.tensor([dix[:-1]], dtype=torch.long).to('cuda')
        y = torch.tensor([dix[1:]], dtype=torch.long).to('cuda')
        z = torch.tensor([mask], dtype=torch.long).to('cuda')
        return x,y,z

