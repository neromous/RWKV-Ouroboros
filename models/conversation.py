import time
from models import Model


class Message(Model):
    def __init__(self, form):
        self.id = None
        self.role = form.get("role", "user")
        self.text = form.get("text", "")
        self.prefix = form.get("prefix", "")
        self.postfix = form.get("postfix", "")
        self.temperature = form.get("temperature", 0.1)
        self.top_p = form.get("top_p", 0.1)
        self.top_k = form.get("top_k", 0)
        self.alpha_frequency = form.get("alpha_frequency", 0.45)
        self.alpha_presence = form.get("alpha_presence", 0.45)
        self.alpha_decay = form.get("alpha_decay", 0.996)
        self.token_ban = form.get("token_ban", [])
        self.token_stop = form.get("token_stop", [0, 65535])
        self.chunk_len = form.get("chunk_len", 128)
        self.token_count = form.get("token_count", 256)
        self.over = form.get("over", True)
        self.scene_id = form.get("scene_id", -1)

    def to_tokens(self) -> list:
        text = self.prefix + self.text + self.postfix
        tokens = self.encode(text)
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        token_postfix = self.get_special_token(self.role, "postfix")
        return token_prefix + tokens + token_postfix


class Scene(Model):
    def __init__(self, form):
        self.id = None
        self.title = form.get("name", "")
        self.history = form.get("history", [])

    def messages(self):
        m = Message.find_all(scene_id=self.id)
        return m

    def add_message(self, message:Message):
        message.scene_id = self.id
        message.save()
        return message

    def set_system(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "system"
        message = Message.new(kargs)
        message.save()
        self.add_message(message)
        return message

    def add_request(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "user"
        message = Message.new(kargs)
        message.save()
        self.add_message(message)
        return message

    def add_response(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "robot"
        message = Message.new(kargs)
        message.save()
        self.add_message(message)
        return message
