from config import config, tokenizer_for_train
import copy
import re
from zhconv import convert


class PromptConfig:
    pass

def to_cn(text):
    return convert(text, 'zh-cn')

class Message:
    @classmethod
    def valid_fields(cls):
        r = [['prefix', str, ''],
             ['postfix', str, ''],
             ['prefix_tokens', list, []],
             ['postfix_tokens', list, []],
             ['role', str, 'text'],
             ['text', to_cn, ''],
             ['content', to_cn, ''],
             ['response', str, ''],
             ['over', bool, True],
             ['no_loss', bool, False],
             ['mask', float, 1.0],
             ['role_mask', float, 1.0],
             ['cfg_pos', str, ""],
             ['cfg_neg', str, ""],
             ['doing_pos',bool,False],
             ['doing_neg',bool,False]]
        return r

    @classmethod
    def tokenizer(cls, for_infer=True):
        if for_infer:
            func = config['inference']['tokenizer']
        else:
            func = config['trainer']['tokenizer']
        return func

    @classmethod
    def new(cls, args: dict, **kwargs):
        m = cls()
        for k, f, dv in cls.valid_fields():
            v = args.get(k, dv)
            setattr(m, k, f(v))
        if len(m.content) != 0:
            m.text = m.content
        return m

    def json(self):
        return copy.deepcopy(self.__dict__)

    def __str__(self):
        text = self.prefix + self.text + self.postfix
        # text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def cover_with_role(self, tokens):
        prefix = config['role'][self.role]['prefix']
        postfix = config['role'][self.role]['postfix']
        prefix = prefix + self.prefix_tokens
        postfix = self.postfix_tokens + postfix
        if self.over:
            return prefix + tokens + postfix
        else:
            return prefix + tokens

    def cover_with_role_mask(self, tokens):
        prefix = config['role'][self.role]['prefix']
        postfix = config['role'][self.role]['postfix']
        prefix_mask = [self.role_mask for x in prefix] + [self.role_mask for x in self.prefix_tokens ]
        postfix_mask = [self.role_mask for x in self.postfix_tokens ] + [self.role_mask for x in postfix]
        if self.over:
            return prefix_mask + [self.mask for x in tokens] + postfix_mask
        else:
            return prefix + [self.mask for x in tokens]


    def tokens(self, for_infer=True):
        tokens = self.tokenizer(for_infer=for_infer).encode(self.prefix + self.text + self.postfix)
        masks = self.cover_with_role_mask(tokens)
        tokens = self.cover_with_role(tokens)
        if self.no_loss:
            masks = [0 for x in tokens]
        return tokens, masks

    def cfg_tokens(self):
        if self.cfg_pos != "":
            pos = self.tokenizer().encode(self.cfg_pos)
        else:
            pos = []
        if self.cfg_neg != "":
            neg = self.tokenizer().encode(self.cfg_neg)
        else:
            neg = []
        return pos, neg
