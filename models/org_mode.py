import orgparse
from orgparse.node import OrgRootNode,OrgNode
import re
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import json
from tqdm import tqdm
import random
import types

config = {"role" : {"raw": {}},
          "priority" : {"A": 1,
                        "B": 2,
                        "C" :3,
                        "D" :4,
                        "E" :5},
          }



def org2item(node:OrgNode):
    item = types.SimpleNamespace()
    if node.todo != None:
        item.role = node.todo.lower()
    else:
        item.role = "raw"
    item.level = node.level or -1
    item.prefix = node.get_property('Prefix',"65530")
    item.postfix = node.get_property('Postfix','65535')
    item.min_loss = node.get_property('MinLoss',0.5)
    item.train_time = node.get_property('TrainTimes')
    item.priority = node.priority or 0
    item.title = node.heading.strip()
    item.body =  node.body.strip()
    item.tags = node.tags
    item.shallow_tags = node.shallow_tags
    return item
