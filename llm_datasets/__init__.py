import json
import time
from utils import log, load_config
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import torch

tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
config = load_config()


class JsonlData:
    _data = []

    @classmethod
    def source_path(cls) -> str:
        return ""

    @classmethod
    def valid_fields(cls) -> str:
        r = [["name", "", str]]
        return r

    @classmethod
    def load(cls, filter_fn=lambda x:x):
        path = cls.source_path()
        result = []
        with open(path, 'r', encoding='utf-8') as f:
            coll = f.readlines()
        for text in coll:
            item = json.loads(text)
            result.append(item)
        return result

    @classmethod
    def _new_from_dict(cls, d):
        m = cls()
        for k, v, f in cls.valid_fields():
            setattr(m, k, f(d.get(k, v)))
        return m

    @classmethod
    def all(cls):
        if len(cls._data) == 0:
            models = cls.load()
            print(f"---load-{cls.__name__}--total: {len(models)} items")
            ms = [cls._new_from_dict(m) for m in models]
            cls._data = ms
            return ms
        else:
            return cls._data

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

    def __repr__(self):
        classname = self.__class__.__name__
        properties = ['{}: ({})'.format(k, v) for k, v in self.__dict__.items()]
        s = '\n'.join(properties)
        return '< {}\n{} \n>\n'.format(classname, s)

    def json(self):
        d = self.__dict__.copy()
        return d

    @classmethod
    def find_by(cls, **kwargs):
        # log('kwargs, ', kwargs, type(kwargs))
        k, v = '', ''
        for key, value in kwargs.items():
            k, v = key, value
        all = cls.all()
        for m in all:
            if v == m.__dict__[k]:
                return m
        return None

    @classmethod
    def find_all(cls, **kwargs):
        ms = []
        k, v = '', ''
        for key, value in kwargs.items():
            k, v = key, value
        all = cls.all()
        for m in all:
            # 也可以用 getattr(m, k) 取值
            if v == m.__dict__[k]:
                ms.append(m)
        return ms

    @classmethod
    def to_tensor(self, coll) -> torch.tensor:
        res = []
        for x in coll:
            res += x.to_tokens()
        return torch.tensor([res], dtype=torch.long)
