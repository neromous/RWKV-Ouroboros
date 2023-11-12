import json
import time
from utils import log, load_config
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
config = load_config()
tokenizer = TRIE_TOKENIZER(config['inference']['tokenizer'])
tokenizer_for_train = TRIE_TOKENIZER(config['trainer']['tokenizer'])

class Base:
    @classmethod
    def encode(cls, text, for_infer=False) -> list:
        if for_infer:
            return tokenizer.encode(text)
        else:
            return tokenizer_for_train.encode(text)

def save(data, path,append=False):
    """
    data 是 dict 或者 list
    path 是保存文件的路径
    """
    s = json.dumps(data, ensure_ascii=False)
    if append :
        method = "a"
    else:
        method = "w+"
    with open(path, method, encoding='utf-8') as f:
        # log('save', path, s, data)
        f.write(s+"\n")


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
        # log('load', s)
    res = []
    for text in texts:
        data = json.loads(text)
        res.append(data)
    return res


class Model(object):
    _data = {}

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
    def new(cls, form):
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
        path = cls.db_path()
        models = load(path)
        ms = [cls._new_from_dict(m) for m in models]
        return ms

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
        models = cls.all()
        index = -1
        for i, e in enumerate(models):
            if e.id == id:
                index = i
                break
        # 判断是否找到了这个 id 的数据
        if index == -1:
            # 没找到
            pass
        else:
            obj = models.pop(index)
            l = [m.__dict__ for m in models]
            path = cls.db_path()
            for i,item in enumerate(l):
                if i == 0:
                    save(item, path)
                else:
                    save(item,path,append=True)
            return obj

    def __repr__(self):
        classname = self.__class__.__name__
        properties = ['{}: ({})'.format(k, v) for k, v in self.__dict__.items()]
        s = '\n'.join(properties)
        return '< {}\n{} \n>\n'.format(classname, s)

    def json(self):
        d = self.__dict__.copy()
        return d

    def save(self):
        models = self.all()
        if self.id is None:
            if len(models) == 0:
                self.id = 1
            else:
                m = models[-1]
                self.id = m.id + 1
            models.append(self)
        else:
            index = -1
            for i, m in enumerate(models):
                if m.id == self.id:
                    index = i
                    break
            log('debug', index)
            models[index] = self
        l = [ m.json() for m in models]
        path = self.db_path()
        for i, item in enumerate(l):
            if i == 0:
                save(item, path)
            else:
                save(item, path, append=True)
