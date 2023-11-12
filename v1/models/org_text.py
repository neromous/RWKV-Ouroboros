from orgparse import  load,loads
from orgparse.node import OrgRootNode,OrgNode
import re
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import json
from tqdm import tqdm
import random
import types
from utils import log, load_config
config = load_config()

tokenizer = TRIE_TOKENIZER('./rwkv_vocab_v20230424.txt')
config = load_config()


class DataNode:
    def __init__(self,
                 text:str,
                 role:str,
                 tokens:list,
                 level:int,
                 prefix_token:list,
                 postfix_token:list,
                 priority:int,
                 min_loss:float=1.0,
                 mask:float=1.0):
        self.text=text
        self.role=role
        self.tokens=tokens
        self.level=level
        self.prefix_token=prefix_token
        self.postfix_token=postfix_token
        self.priority=priority

    def __str__(self) -> str:
        return tokenizer.decode(self.prefix_token + self.tokens + self.postfix_token)

    def json(self):
        return self.__dict__.copy()

    def yield_train_data(self,req_len=2049,window=1024):
        tokens = self.prefix_token + self.tokens + self.postfix_token
        while len(tokens) > 0 :
            output = tokens[:req_len]
            tokens = tokens[req_len-window:]
            yield output

    def __add__(self, item):
        text = self.text + item.text
        tokens = self.prefix_token + self.tokens +self.postfix_token
        tokens = tokens + item.prefix_token + item.tokens + item.postfix_token
        item = DataNode(text=text,
                        tokens=tokens,
                        priority=self.priority,
                        level=self.level,
                        prefix_token=[],
                        postfix_token=[],
                        role="")
        return item


def to_data_node(node:OrgNode) -> DataNode:
    if node.todo != None:
        role = node.todo.lower()
    else:
        print("======",node.todo)
        role = "text"
    level = node.level or -1
    prefix_token = config['role'][role]['prefix']
    postfix_token = config['role'][role]['postfix']
    priority = node.priority or 0
    text = node.heading.strip() + "\n" + node.body.replace("\\n","\n").strip()
    tokens = tokenizer.encode(text)
    item = DataNode(text=text,
                    tokens=tokens,
                    priority=int(priority),
                    level=int(level),
                    prefix_token=prefix_token,
                    postfix_token=postfix_token,
                    role=role)
    return item


def file_to_node(path:str) -> dict:
    root = load(path)
    res = {}
    tag = 0
    for node in root[1:]:
        if node.level == 1:
            tag += 1
            res[tag] = to_data_node(node)
        else:
            res[tag] +=  to_data_node(node)
        
    return res

def text_to_node(text:str) -> dict:
    root = loads(text)
    res = {}
    tag = 0
    for node in root[1:]:
        if node.level == 1:
            tag += 1
            res[tag] = to_data_node(node)
        else:
            res[tag] += to_data_node(node)
    return res


def inference(model,
              token,
              temperature = 0.1,
              top_p = 0.1,
              top_k =  0,
              alpha_frequency =  0.45,
              alpha_presence =  0.45,
              alpha_decay =  0.996,
              token_ban =  [0],
              token_stop = [65535],
              token_count =  256,
              ctx = 2048):

    # remove end 
    while token[-1] in [0,65535,261,11]:
        token = token[:-1]
    token = token[token_count - ctx:]
    all_tokens = [x for x in token]
    length = len(all_tokens)
    out_last = 0
    out_str = ''
    occurrence = {}
    alpha_presence = 0.45
    alpha_frequency = 0.45
    alpha_decay = 0.996
    for i in range(token_count):
        #get logits
        output = model.inference(all_tokens[-ctx:])
        logits = output[-1]
        for n in token_ban:
            logits[n] = -float('inf')
        for n in occurrence:
           logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
        token = model.sample_logits(logits,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k)
        if token in token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
           occurrence[xxx] *= alpha_decay
        if token not in occurrence:
           occurrence[token] = 1
        else:
           occurrence[token] += 1

        tmp = tokenizer.decode(all_tokens[length+out_last:])

        if  '\ufffd' not in tmp:
            print(tmp, end="", flush=True)
            out_str += tmp
            out_last = i + 1
    return out_str




def inference_with_state(model,
                         tokens,
                         temperature = 0.1,
                         top_p = 0.1,
                         top_k =  0,
                         chunk_len=128,
                         alpha_frequency =  0.45,
                         alpha_presence =  0.45,
                         alpha_decay =  0.996,
                         token_ban =  [0],
                         token_stop = [65535],
                         token_count =  256,
                         callback=True,
                         state= None):
    while tokens[-1] in [0,65535,261,11]:
        tokens = tokens[:-1]

    #print("======",tokens)
    # remove end
    out_str = ""
    occurrence = {}
    alpha_presence = 0.45
    alpha_frequency = 0.45
    alpha_decay = 0.996

    for token in tqdm(tokens):
        logits, state = model.inference_with_state(token, state)
    #print("==init_state==",init_state)
    all_tokens = []
    out_last = 0
    for i in range(token_count):
        for n in token_ban:
            logits[n] = -float('inf')
        for n in occurrence:
           logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

        token = model.sample_logits(logits,
                                    temperature=temperature,
                                    top_p=top_p)
        all_tokens += [token]
        for xxx in occurrence:
           occurrence[xxx] *= alpha_decay
        if token not in occurrence:
           occurrence[token] = 1
        else:
           occurrence[token] += 1

        #print("==token==",token)
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_str += tmp
            out_last = i + 1
        logits, state = model.inference_with_state(token, state)
        #print("==state==",state)
    return out_str,state




    # print("==token=",token)
    # token = token[token_count - ctx:]
    # all_tokens = [x for x in token]
    # length = len(all_tokens)
    # out_last = 0
    # out_str = ''
    # occurrence = {}
    # alpha_presence = 0.45
    # alpha_frequency = 0.45
    # alpha_decay = 0.996
    # print("==all==",all_tokens)


    # for i in range(token_count):
    #     #get logits
    #     output,state = model.inference_with_state(all_tokens[-1],state)
    #     logits = output[-1]
    #     for n in token_ban:
    #         logits[n] = -float('inf')
    #     for n in occurrence:
    #        logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
    #     token = model.sample_logits(logits,
    #                                 temperature=temperature,
    #                                 top_p=top_p,
    #                                 top_k=top_k)
    #     if token in token_stop:
    #         break
    #     all_tokens += [token]
    #     for xxx in occurrence:
    #        occurrence[xxx] *= alpha_decay
    #     if token not in occurrence:
    #        occurrence[token] = 1
    #     else:
    #        occurrence[token] += 1

    #     tmp = tokenizer.decode(all_tokens[length+out_last:])

    #     if  '\ufffd' not in tmp:
    #         print(tmp, end="", flush=True)
    #         out_str += tmp
    #         out_last = i + 1
    # return out_str , state


# class DocNode:
#     def __init__(self,
#                  node:OrgNode,
#                  prefix='',
#                  postfix='',
#                  prefix_token=[],
#                  postfix_token=[],
#                  mask=1.0,
#                  idx=-1):
#         self.idx = idx
#         self.node = node
#         if node.todo != None:
#             self.role = node.todo.lower()
#         else:
#             self.role = "text"
#         if node.priority == "A":
#             self.mask = 1.0
#         elif node.priority == "B":
#             self.mask = 0.8
#         elif node.priority == "C":
#             self.mask = 0.6
#         else:
#             self.mask = 1.0
#         if len(prefix) == 0:
#             prefix_token = config[self.role]['prefix']
#         if len(postfix) == 0:
#             postfix_token = config[self.role]['postfix']
#         self.priority = node.priority
#         self.text = node.heading + "\n" + node.body
#         self.prefix_token = prefix_token + tokenizer.encode(prefix)
#         self.postfix_token = postfix_token + tokenizer.encode(postfix)
#         self.tokens = tokenizer.encode(self.text)
#         self.fix_token=self.prefix_token+self.tokens +self.postfix_token
#         self.length = len(self.fix_token)
#         self.mask = [mask for x in range(0,self.length)]

#     def __str__(self):
#         return self.text


# class Doc:
#     def __init__(self,nodes=[],ctx=2049):
#         self.ctx = ctx
#         self.nodes = []
#         self.cut_off = []
#         for x in nodes:
#             if isinstance(x, OrgNode):
#                 self.nodes.append(DocNode(x))
#             elif isinstance(x,DocNode):
#                 self.nodes.append(x)
#             else:
#                 raise Exception()

#     @classmethod
#     def decode(cls, x:list):
#         return tokenizer.decode(x)

#     def load_node(self,root_node:OrgRootNode):
#         i = 0
#         for sub in root_node.children:
#             i += 1
#             for item in sub:
#                 self.nodes.append(DocNode(item,idx=i))
#         return self

#     def add_item(self,item:DocNode):
#         if self.length + item.length  <= self.ctx:
#             self.nodes.append(item)
#             return self
#         else:
#             self.cut_off.append(item)
#             return False

#     @property
#     def length(self):
#         result = 0
#         for node in self.nodes:
#             result += node.length
#         return result

#     def __add__(self,item):
#         while self.length < self.ctx and len(item.nodes) > 0 :
#             self.nodes.append(item.nodes.pop(0))
#         self.cut_off = item.nodes
#         return self

#     @classmethod
#     def export(cls,doc):
#         all_node = doc.nodes +doc.cut_off
#         output = cls()
#         while len(all_node) > 0:
#             if output.length + all_node[0].length <= doc.ctx:
#                 output.nodes.append(all_node[0])
#                 all_node = all_node[1:]
#             else:
#                 yield output
#                 output = cls()


#     @property
#     def token(self):
#         postfix_len = self.ctx - self.length
#         postfix = [0 for x in range(0,postfix_len)]
#         tokens = []
#         masks = []
#         for node in self.nodes:
#             tokens = tokens + node.fix_token
#             masks =  masks + node.mask
#         tokens = tokens + postfix
#         masks = masks + postfix
#         return tokens

#     def from_file(self,path):
#         data = load(path)[0]
#         return data.children
