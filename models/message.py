import time
from models import Model
import copy
import torch

class Page(Model):
    def __init__(self,form):
        text = form.get('text','')
        self.prefix = form.get("prefix", "")
        self.postfix = form.get("postfix", "")
        self.prefix_token = form.get("prefix_token", [])
        self.prefix_token = form.get("postfix_token", [])
        if len(text) !=0:
            self.tokens = form.get('tokens', self.encode(self.prefix + text + self.postfix))
        else:
            self.tokens = form.get('tokens',[])

class Message(Model):
    def __init__(self, form):
        self.id = None
        self.role = form.get("role", "user")
        self.text = form.get("text", "")
        self.prefix = form.get("prefix", "")
        self.postfix = form.get("postfix", "")
        self.prefix_token = form.get("prefix_token", [])
        self.prefix_token = form.get("postfix_token", [])
        self.temperature = form.get("temperature", 0.1)
        self.top_p = form.get("top_p", 0.1)
        self.top_k = form.get("top_k", 0)
        self.alpha_frequency = form.get("alpha_frequency", 0.45)
        self.alpha_presence = form.get("alpha_presence", 0.45)
        self.alpha_decay = form.get("alpha_decay", 0.996)
        self.token_ban = form.get("token_ban", [0])
        self.token_stop = form.get("token_stop", [65530,65531,65532,65533,65534,65535])
        self.chunk_len = form.get("chunk_len", 128)
        self.token_count = form.get("token_count", 256)
        self.over = form.get("over", True)
        self.scene_id = form.get("scene_id", -1)
        self.response = form.get('response', "")
        self.generated = form.get('generated', False)
        self.to_train = form.get('to_train', True)
        self.ctx = form.get('ctx', 2048)
        self.ctx_fix = form.get('ctx_fix', int(self.ctx / 8))
        if len(self.text) != 0:
            self.tokens = form.get('tokens', self.encode(self.prefix + self.text + self.postfix))
        else:
            self.tokens = form.get('tokens',[])

    def to_tokens(self) -> list:
        text = self.prefix + self.text + self.postfix
        tokens = self.encode(text)
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        self.prefix_token = token_prefix
        token_postfix = self.get_special_token(self.role, "postfix")
        self.postfix_token = token_postfix
        if self.over:
            tokens = token_prefix + tokens + token_postfix
        else:
            tokens = token_prefix + tokens
        _tokens = []
        return tokens

    def load_file(self,path):
        with open(path, 'r',encoding='utf-8') as f:
            text = f.read()
        self.text = text
        return self

    def message_as_iter(self):
        tokens = copy.deepcopy(self.tokens)
        token_prefix = self.get_special_token(self.role, "prefix")
        self.prefix_token = token_prefix
        token_postfix = self.get_special_token(self.role, "postfix")
        self.postfix_token = token_postfix
        ljust = len(self.prefix_token)
        rjust = len(self.postfix_token)
        l = self.ctx
        fix = self.ctx_fix
        template = self.__dict__
        template.pop('text')
        while len(tokens) > 0:
            tokens = token_prefix + tokens
            out = tokens[:l-rjust] +token_postfix
            tokens = tokens[l-ljust - fix :]
            template['tokens'] = out
            yield template
    
    def token_as_iter(self):
        tokens = copy.deepcopy(self.tokens)
        token_prefix = self.get_special_token(self.role, "prefix")
        self.prefix_token = token_prefix
        token_postfix = self.get_special_token(self.role, "postfix")
        self.postfix_token = token_postfix
        ljust = len(self.prefix_token)
        rjust = len(self.postfix_token)
        l = self.ctx
        fix = self.ctx_fix
        while len(tokens) > 0:
            tokens = token_prefix + tokens
            out = tokens[:l-rjust] +token_postfix
            tokens = tokens[l-ljust - fix :]
            yield out

