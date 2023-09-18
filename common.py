import json
import time
# from utils import log, load_config
# from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
# import torch

# tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
# config = load_config()



# class Message(Base):
#     def __init__(self, form):
#         self.role = form.get("role", "user")
#         self.text = form.get("text", "")
#         self.prefix = form.get("prefix", "")
#         self.postfix = form.get("postfix", "")
#         self.prefix_token = form.get("prefix_token", [])
#         self.postfix_token = form.get("postfix_token", [])

#     def to_tokens(self) -> list:
#         text = self.prefix + self.text + self.postfix
#         tokens = self.encode(text)
#         token_prefix = self.get_special_token(self.role, "prefix")
#         token_postfix = self.get_special_token(self.role, "postfix")
#         tokens = token_prefix + tokens + token_postfix
#         tokens = self.prefix_token + tokens + self.postfix_token
#         tokens = [x for x in tokens if self.is_valid_token(x)]
#         return tokens


# class TrainData(Base):

#     def __init__(self, form):
#         self.prefix = form.get("prefix", "")
#         self.postfix = form.get("postfix", "")
#         self.messages = [Message(x) for x in form.get("messages", [])]
#         self.prefix_token = form.get("prefix_token", [])
#         self.postfix_token = form.get("postfix_token", [])

#     @classmethod
#     def is_valid(cls, item:dict):
#         return True

#     def to_tokens(self) -> list:
#         message_tokens = []
#         for message in self.messages:
#             message_tokens += message.to_tokens()
#         prefix_text_token = self.encode(self.prefix)
#         postfix_text_token = self.encode(self.postfix)
#         # 增加前缀后缀
#         tokens = prefix_text_token + message_tokens + postfix_text_token
#         # 增加special tokne
#         tokens = self.prefix_token + tokens + self.postfix_token
#         tokens = [x for x in tokens if self.is_valid_token(x)]
#         return tokens

#     def to_tensor(self) -> torch.tensor:
#         tokens = self.to_tokens()
#         return torch.tensor([tokens], dtype=torch.long)
