import time
from models import Model


class Text(Model):
    def __init__(self, form):
        self.id = None
        self.role = form.get('role', "text")
        self.title = form.get('title', '')
        self.description = form.get('description', '')
        self.prefix = form.get('prefix', '')
        self.postfix = form.get('postfix', '')
        self.text = form.get('text', '')

    def to_tokens(self) -> list:
        text = self.prefix + self.description + self.text + self.postfix
        tokens = self.encode(text)
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        token_postfix = self.get_special_token(self.role, "postfix")
        return token_prefix + tokens + token_postfix
