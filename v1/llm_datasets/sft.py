from llm_datasets import JsonlData
import torch

class Sft(JsonlData):
    @classmethod
    def source_path(cls):
        return "/home/neromous/Documents/blackfog/RWKV-Ouroboros/data/Sft.jsonl"

    @classmethod
    def valid_fields(cls):
        r = [["section_id", -1, int],
             ["section_sort", -1, int],
             ["role", "user", str],
             ["text", "", str]]
        return r

    def to_tokens(self) -> list:
        text = self.text
        tokens = self.encode(text)
        # 采用token的方式直接增加special token 避免分词不精确
        token_prefix = self.get_special_token(self.role, "prefix")
        token_postfix = self.get_special_token(self.role, "postfix")
        return token_prefix + tokens + token_postfix
