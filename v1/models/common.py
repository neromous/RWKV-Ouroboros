import json
import time
import collections
import copy
import types
from utils import log, load_config
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
config = load_config()
tokenizer = TRIE_TOKENIZER(config['inference']['tokenizer'])
tokenizer_for_train = TRIE_TOKENIZER(config['trainer']['tokenizer'])

class Model(types.SimpleNamespace):
    _data = collections.OrderedDict()
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
    def encode(cls, text, for_infer=False) -> list:
        if for_infer:
            return tokenizer.encode(text)
        else:
            return tokenizer_for_train.encode(text)

    @classmethod
    def decode(cls, tokens) -> str:
        tokens = [x for x in tokens if cls.is_valid_token(x)]
        return tokenizer.decode(tokens)

    @classmethod
    def db_path(cls) -> str:
        classname = cls.__name__
        path = 'data/{}.jsonl'.format(classname)
        return path

    @classmethod
    def new(cls, form) :
        m = cls(form)
        m.save()
        return m

    @classmethod
    def _new_from_dict(cls, d):
        m = cls({})
        for k, v in d.items():
            setattr(m, k, v)
        return m

    @classmethod
    def all(cls):
        models = copy.deepcopy(cls._data.values())
        return models

    @classmethod
    def find_all(cls, **kwargs):
        ms = []
        log('kwargs, ', kwargs, type(kwargs))
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
    def find_by(cls, **kwargs):
        log('kwargs, ', kwargs, type(kwargs))
        k, v = '', ''
        for key, value in kwargs.items():
            k, v = key, value
        all = cls.all()
        for m in all:
            if v == m.__dict__[k]:
                return m
        return None

    @classmethod
    def find(cls, id):
        return cls.find_by(id=id)

    @classmethod
    def delete(cls, id):
        index = -1
        for i, e in cls._data.items():
            if e.id == id:
                index = i
                break
        if index == -1:
            return None
        else:
            obj = cls._data.pop(index)
            return obj

    def __repr__(self):
        classname = self.__class__.__name__
        properties = ['{}: ({})'.format(k, v) for k, v in self.__dict__.items()]
        s = '\n'.join(properties)
        return '< {}\n{} \n>\n'.format(classname, s)

    def save(self):
        cls = self.__class__
        if self.id is None:
            if len(cls._data) == 0:
                self.id = 1
            else:
                max_id = max(cls._data.keys())
                self.id = max_id + 1
            self._data[self.id] = self
        else:
            index = -1
            for i, m in cls._data.items():
                if m.id == self.id:
                    index = i
                    break
            cls._data[index] = self
        return self

    def json(self):
        d = self.__dict__.copy()
        return d
