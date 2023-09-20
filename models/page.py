import torch
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import re
import json
from utils import log, load_config
import types
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
    def __init__(self,
                 tokens,
                 ctx=2048,
                 prefix_token=[],
                 postfix_token=[],
                 role="text"
                 ):
        self.ctx = ctx
        self.prefix_token = prefix_token
        self.postfix_token = postfix_token
        self.tokens = tokens
        if len(self.ctx_token) > ctx:
            raise Exception()

    @property
    def ctx_token(self):
        return self.prefix_token + self.tokens + self.postfix_token

    @classmethod
    def new(cls,
            text,
            ctx=2048,
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
                    postfix_token=r
                    )
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
                  ctx=2048,
                  prefix="",
                  postfix="",
                  prefix_token=[],
                  postfix_token=[]):
        with open(path,'r',encoding='utf-8') as f:
            text = f.read()

        m = cls.new(text,
                    ctx=ctx,
                    prefix=prefix,
                    postfix=postfix,
                    prefix_token=prefix_token,
                    postfix_token=postfix_token
                    )
        return m

    @classmethod
    def org_header(cls,text:str):
        print(text)
        role, text = text.split(" ",1)
        item = types.SimpleNamespace()
        #if item.prefix_token == [65530,65531]:
        if role == "SYSTEM":
            item.prefix_token = [65530,65531]
            item.postfix_token = [65535]                        
        elif role == "USER":
            item.prefix_token = [65530,65532]
            item.postfix_token = [65535]                        
        elif role == "ROBOT":
            item.prefix_token = [65530,65534]
            item.postfix_token = [65535]                        
        elif role == "THINK":
            item.prefix_token = [65530,65533]
            item.postfix_token = [65535]            
        else:
            item.prefix_token = []
            item.postfix_token = []
        item.text =   text
        res = cls(cls.encode(item.text),
                  prefix_token= item.prefix_token,
                  postfix_token= item.postfix_token
                  )
        return res


    @classmethod
    def from_org(cls,
                  path,
                  ctx=2048,
                  prefix="",
                  postfix="",
                  prefix_token=[],
                  postfix_token=[]):
        with open(path,'r',encoding='utf-8') as f:
            text = f.read()
        results = []
        parts = re.split("\n\* ",text)
        parts = [x.strip() for x in parts ]
        parts = [x for x in parts if x != '']
        for part in parts:
            coll = re.split("\n\*+\s+",part)
            coll = coll[1:]
            coll = [x.strip() for x in coll if not x.startswith("\n#+") and not x.startswith("#+")]
            coll = [x for x in coll if x !=""]
            coll = [cls.org_header(x) for x in coll]
            results.append(coll)
        return results

    @property
    def org_node(self):
        if self.prefix_token == [65530,65531]:
            role = "SYSTEM"
        elif self.prefix_token == [65530,65532]:
            role = "USER"
        elif self.prefix_token == [65530,65534]:
            role = "ROBOT"
        elif self.prefix_token == [65530,65533]:
            role = "THINK"
        else:
            role = "BOOK"
        text = "\n**" + " " + role +" " +self.text
        return text

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
    def from_dict(cls,
                  data,
                  ctx=2048):
        role = data.get('role','text')
        text = data['text']
        m = cls(text,ctx=ctx)
        return m
    
    @property
    def tensor(self):
        if len(self.tokens) > 0 :
            return torch.tensor([self.ctx_token], dtype=torch.long)
        else:
            raise Exception()

    def mask(self):
        m = [1 for x in self.ctx_token]
        return torch.tensor([m],dtype=torch.float)

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
