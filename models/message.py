import time
from models.common import Model
import copy
import torch
from utils import log, load_config
config = load_config()
prompt_config = config['inference']['prompt_config']



class Message(Model):
    def __init__(self, form):
        self.id = None
        self.role = form.get("role", "user")
        self.role = form.get("no_loss", True)
        self.text = form.get("text", "")
        self.prefix = form.get("prefix", "")
        self.load_state = form.get("load_state", "default")
        self.save_state = form.get("save_state", "default")
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
        self.tokens = self.to_tokens(for_infer=True)
        self.tokens_for_train = self.to_tokens()

    def to_tokens(self,  for_infer=False) -> list:
        text = self.prefix + self.text + self.postfix
        tokens = self.encode(text, for_infer=for_infer)
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        self.prefix_token = token_prefix
        token_postfix = self.get_special_token(self.role, "postfix")
        self.postfix_token = token_postfix
        if self.over:
            tokens = token_prefix + tokens + token_postfix
        else:
            tokens = token_prefix + tokens
        return tokens

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

