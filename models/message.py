import time
from models.core import Model
import copy
import torch
from utils import log, load_config
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
config = load_config()
prompt_config = config['inference']['prompt_config']
tokenizer = TRIE_TOKENIZER(config['inference']['tokenizer'])

class Message(Model):
    def __init__(self, form):
        self.id = None
        self.role = form.get("role", "user")
        self.text = form.get("text", "")
        self.prefix = form.get("prefix", "")
        self.load_state = form.get("load_state", "default")
        self.save_state = form.get("load_state", "default")
        self.postfix = form.get("postfix", "")
        self.prefix_token = form.get("prefix_token", [])
        self.prefix_token = form.get("postfix_token", [])
        self.temperature = form.get("temperature", prompt_config["temperature"])
        self.top_p = form.get("top_p", prompt_config['top_p'])
        self.top_k = form.get("top_k", prompt_config['top_k'])
        self.alpha_frequency = form.get("alpha_frequency",
                                        prompt_config['alpha_frequency'])
        self.alpha_presence = form.get("alpha_presence",
                                       prompt_config['alpha_presence'])
        self.alpha_decay = form.get("alpha_decay",
                                    prompt_config['alpha_decay'])
        self.token_ban = form.get("token_ban", prompt_config['token_ban'])
        self.token_stop = form.get("token_stop", prompt_config['token_stop'])
        self.chunk_len = form.get("chunk_len", prompt_config['chunk_len'])
        self.token_count = form.get("token_count", prompt_config['token_count'])
        self.over = form.get("over", prompt_config['over'])
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
        if len(self.text) !=0:
            tokens = self.encode(text)
        else:
            tokens = self.tokens
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        self.prefix_token = token_prefix
        token_postfix = self.get_special_token(self.role, "postfix")
        self.postfix_token = token_postfix
        if self.over:
            tokens = token_prefix + tokens + token_postfix
        else:
            tokens = token_prefix + tokens
        #_tokens = []
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

