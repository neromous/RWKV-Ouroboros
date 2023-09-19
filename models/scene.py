import time
import torch
from models import Model
from models.message import Message

class Scene(Model):
    def __init__(self, form):
        self.id = None
        self.title = form.get("title", "")
        #self.messages = form.get("messages", [])
        self.prefix = form.get("prefix", "")
        self.postfix = form.get("postfix", "")
        self.messages = [Message(x) for x in form.get("messages", [])]
        self.prefix_token = form.get("prefix_token", [])
        self.postfix_token = form.get("postfix_token", [])


    def json(self):
        d = self.__dict__.copy()
        messages = self.messages
        d['messages'] = [x.json() for x in messages if type(x) != dict ]
        return d


    @classmethod
    def new(cls, form):
        m = cls(form)
        messages = m.messages
        messages = [Message(x) for x in messages if type(x) ==  dict]
        m.save()
        return m

    def messages(self):
        m = Message.find_all(scene_id=self.id)
        return m

    def add_message(self, form: dict) -> Message:
        form['scene_id'] = self.id
        message = Message.new(form)
        return message

    def set_system(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "system"
        message = self.add_message(kargs)
        return message

    def add_request(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "user"
        message = self.add_message(kargs)
        return message

    def add_response(self, text, **kargs):
        kargs["text"] = text
        kargs["role"] = "robot"
        message = self.add_message(kargs)
        return message

    @classmethod
    def is_valid(cls, item:dict):
        return True

    def to_tokens(self) -> list:
        message_tokens = []
        for message in self.messages:
            message_tokens += message.to_tokens()
        prefix_text_token = self.encode(self.prefix)
        postfix_text_token = self.encode(self.postfix)
        # 增加前缀后缀
        tokens = prefix_text_token + message_tokens + postfix_text_token
        # 增加special tokne
        tokens = self.prefix_token + tokens + self.postfix_token
        tokens = [x for x in tokens if self.is_valid_token(x)]
        return tokens

    def to_tensor(self) -> torch.tensor:
        tokens = self.to_tokens()
        return torch.tensor([tokens], dtype=torch.long)
