import torch
from orgparse import  load,loads
from orgparse.node import OrgRootNode,OrgNode
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import re
import json
from utils import log, load_config
import types
import random
config = load_config()
tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')

class Base:
    @classmethod
    def get_special_token(cls, role, pos):
        r = config["role"]
        return r[role][pos]

    @classmethod
    def is_valid_role(cls, role):
        return role in config['role'].keys()

    @classmethod
    def to_tensor(cls,tokens):
        tensor = torch.tensor([tokens],dtype=torch.long)
        return tensor

    @classmethod
    def is_valid_token(cls, token):
        if 65535 >= token and token >= 0:
            return True
        else:
            return False

    @classmethod
    def encode(cls, text) -> list:
        return tokenizer.encode(text)

    @classmethod
    def decode(cls, tokens) -> str:
        tokens = [x for x in tokens if cls.is_valid_token(x)]
        return tokenizer.decode(tokens)


class Page(Base):
    _padding = 0
    def __init__(self,
                 tokens,
                 ctx=2049,
                 prefix_token=[],
                 postfix_token=[],
                 role="text",
                 mask_value:float= 1.0,
                 window=512):
        self.ctx = ctx
        self.window=0
        self.role = role
        self.prefix_token = prefix_token
        self.postfix_token = postfix_token
        self.tokens = tokens
        if len(self.ctx_token) > ctx:
            raise Exception()

    @classmethod
    def new(cls,
            text,
            ctx=2049,
            prefix="",
            postfix="",
            prefix_token=[],
            postfix_token=[]):
        tokens = cls.encode(text)
        res = []
        l = prefix_token  +  cls.encode(prefix)
        r = postfix_token  + cls.encode(postfix)
        ljust = len(l)
        rjust = len(r)
        while len(tokens) > 0:
            out = tokens[:ctx - ljust - rjust]
            tokens = tokens[ctx - ljust - rjust:]
            m = cls(out ,
                    ctx=ctx,
                    prefix_token=l,
                    postfix_token=r)
            res.append(m)
        return res


    def __getitem__(self, item):
        if isinstance(item, slice):
            tokens = self.tokens[item.start:item.stop]
            if len(tokens) <= 0:
                raise Exception()
            m = Page(tokens,
                     ctx = len(self.prefix_token + tokens + self.postfix_token),
                     prefix_token=self.prefix_token,
                     postfix_token=self.postfix_token)
            return m
        else:
            m = Page(self.tokens[:int(item)],
                     ctx = len(self.prefix_token) + int(item) + len(self.postfix_token),
                     prefix_token=self.prefix_token,
                     postfix_token=self.postfix_token)
            return m


    @classmethod
    def from_txt(cls,
                 path,
                 ctx=2049,
                 prefix="",
                 postfix="",
                 prefix_token=[],
                 postfix_token=[],
                 ):
        with open(path,'r',encoding='utf-8') as f:
            text = f.read()

        m = cls.new(text,
                    ctx=ctx,
                    prefix=prefix,
                    postfix=postfix,
                    prefix_token=prefix_token,
                    postfix_token=postfix_token)
        return m

    @classmethod
    def node2page(cls,node:OrgNode):
        if node.todo != None:
            role = node.todo.lower()
        else:
            role = "text"
        text = node.heading.strip() + "\n" + node.body.strip()
        prefix_token = cls.get_special_token(role, "prefix")
        postfix_token = cls.get_special_token(role, "postfix")
        tokens = cls.encode(text)
        m = cls(tokens,
                ctx=2049,
                prefix_token=prefix_token,
                postfix_token=postfix_token,
                role=role,
                mask_value= 1.0)
        return m

    @classmethod
    def from_org_text(cls,text,shuffle=False):
        text = text.strip()
        if not text.startswith("*") and  not text.startswith("#+"):
            text = "*" + " " + text
        data = loads(text)
        root = data[0]
        results = []
        for level1 in root.children:
            m = cls.node2page(level1.children[0])
            for level2 in level1.children[1:]:
                m =  m + cls.node2page(level2)
            results.append(m)
        if shuffle:
            cache = results[1:]
            random.shuffle(cache)
            results = results[:1] + cache
        return results

    @classmethod
    def from_org(cls,path,shuffle=False):
        data = load(path)
        root = data[0]
        results = []
        for level1 in root.children:
            m = cls.node2page(level1.children[0])
            for level2 in level1.children[1:]:
                m =  m + cls.node2page(level2)
            results.append(m)
        if shuffle:
            cache = results[1:]
            random.shuffle(cache)
            results = results[:1] + cache
        return results

    @classmethod
    def from_jsonl(cls,
                   path):
        with open(path,'r',encoding='utf-8') as f:
            texts = f.readlines()
        coll = []
        for text in texts:
            item = json.loads(text)
            text = item['text']
            role = item['role']
            prefix_token = cls.get_special_token(role, "prefix")
            postfix_token = cls.get_special_token(role, "postfix")
            m = cls.new(text,
                        prefix_token=prefix_token,
                        postfix_token=postfix_token)
            coll = coll + m
        return coll

    @classmethod
    def from_dict(cls,data,ctx=2049):
        role = data.get('role','text')
        text = data['text']
        m = cls(text,ctx=ctx)
        return m

    @property
    def ctx_token(self):
        return self.prefix_token + self.tokens + self.postfix_token

    @property
    def tensor(self):
        if len(self.tokens) > 0 :
            return torch.tensor([self.ctx_token], dtype=torch.long)
        else:
            raise Exception()

    def yield_token(self,ctx):
        token = self.prefix_token + self.tokens + self.postfix_token
        while len(token) > 0:
            output = token[:ctx]
            token = token[1024:]
            yield output

    def __str__(self):
        return self.decode(self.tokens).strip()

    @property
    def text(self):
        return self.__str__()

    def __add__(self, item):
        tokens = self.prefix_token +self.tokens +self.postfix_token + item.prefix_token + item.tokens + item.postfix_token
        m = Page(tokens,
                 ctx = len(tokens),
                 prefix_token=[],
                 postfix_token=[])
        return m

    def json(self):
        return self.__dict__
