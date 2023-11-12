from config import config
import copy
import re


class PromptConfig:
    pass


class Message:
    @classmethod
    def valid_fields(cls):
        r = [['prefix', str, ''],
             ['postfix', str, ''],
             ['role', str, 'text'],
             ['text', str, ''],
             ['response', str, ''],
             ['over', bool, True],
             ['no_loss', bool, False],
             ['mask', float, 1.0],
             ]
        return r

    @classmethod
    def tokenizer(cls, for_infer=True):
        if for_infer:
            return config['inference']['tokenizer']
        else:
            return config['trainer']['tokenizer']

    @classmethod
    def new(cls, args: dict, **kwargs):
        m = cls()
        for k, f, dv in cls.valid_fields():
            v = args.get(k, dv)
            setattr(m, k, f(v))
        return m

    def json(self):
        return copy.deepcopy(self.__dict__)

    def __str__(self):
        text = self.prefix + self.text + self.postfix
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def cover_with_role(self, tokens):
        prefix = config['role'][self.role]['prefix']
        postfix = config['role'][self.role]['postfix']
        if self.over:
            return prefix + tokens + postfix
        else:
            return prefix + tokens

    def tokens(self, for_infer=True):
        tokens = self.tokenizer(for_infer=True).encode(
            self.prefix + self.text + self.postfix)
        if self.no_loss:
            masks = [0 for x in tokens]
        else:
            masks = [self.mask for x in tokens]
        tokens = self.cover_with_role(tokens)
        return tokens, masks
