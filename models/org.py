from orgparse import  load,loads
from orgparse.node import OrgRootNode,OrgNode
import re
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from utils import load_config
from tqdm import tqdm
import random

tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
config = load_config()
config = config['role']
print(config)

class DocNode:
    def __init__(self,
                 node:OrgNode,
                 prefix='',
                 postfix='',
                 prefix_token=[],
                 postfix_token=[],
                 mask=1.0,
                 idx=-1):
        self.idx = idx
        self.node = node
        if node.todo != None:
            self.role = node.todo.lower()
        else:
            self.role = "text"
        if node.priority == "A":
            self.mask = 1.0
        elif node.priority == "B":
            self.mask = 0.8
        elif node.priority == "C":
            self.mask = 0.6
        else:
            self.mask = 1.0
        if len(prefix) == 0:
            prefix_token = config[self.role]['prefix']
        if len(postfix) == 0:
            postfix_token = config[self.role]['postfix']
        self.priority = node.priority
        self.text = node.heading + "\n" + node.body
        self.prefix_token = prefix_token + tokenizer.encode(prefix)
        self.postfix_token = postfix_token + tokenizer.encode(postfix)
        self.tokens = tokenizer.encode(self.text)
        self.fix_token=self.prefix_token+self.tokens +self.postfix_token
        self.length = len(self.fix_token)
        self.mask = [mask for x in range(0,self.length)]

    def __str__(self):
        return self.text


class Doc:
    def __init__(self,nodes=[],ctx=2049):
        self.ctx = ctx
        self.nodes = []
        self.cut_off = []
        for x in nodes:
            if isinstance(x, OrgNode):
                self.nodes.append(DocNode(x))
            elif isinstance(x,DocNode):
                self.nodes.append(x)
            else:
                raise Exception()

    @classmethod
    def decode(cls, x:list):
        return tokenizer.decode(x)

    def load_node(self,root_node:OrgRootNode):
        i = 0
        for sub in root_node.children:
            i += 1
            for item in sub:
                self.nodes.append(DocNode(item,idx=i))
        return self

    def add_item(self,item:DocNode):
        if self.length + item.length  <= self.ctx:
            self.nodes.append(item)
            return self
        else:
            self.cut_off.append(item)
            return False

    @property
    def length(self):
        result = 0
        for node in self.nodes:
            result += node.length
        return result

    def __add__(self,item):
        while self.length < self.ctx and len(item.nodes) > 0 :
            self.nodes.append(item.nodes.pop(0))
        self.cut_off = item.nodes
        return self

    @classmethod
    def export(cls,doc):
        all_node = doc.nodes +doc.cut_off
        output = cls()
        while len(all_node) > 0:
            if output.length + all_node[0].length <= doc.ctx:
                output.nodes.append(all_node[0])
                all_node = all_node[1:]
            else:
                yield output
                output = cls()


    @property
    def token(self):
        postfix_len = self.ctx - self.length
        postfix = [0 for x in range(0,postfix_len)]
        tokens = []
        masks = []
        for node in self.nodes:
            tokens = tokens + node.fix_token
            masks =  masks + node.mask
        tokens = tokens + postfix
        masks = masks + postfix
        return tokens

    def from_file(self,path):
        data = load(path)[0]
        return data.children

# if __name__ == "__main__":
#     m = Doc()
#     m.from_file('./data/sft.org')

#     for x in m.export(m):
#         print(len(x.token))
#         print(x.length)
