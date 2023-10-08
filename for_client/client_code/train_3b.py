import requests
import sys
import orgparse
from orgparse import loads
import sys
import re
from model import Inference

path ='/home/neromous/trpg-value.jsonl'

env = Inference('3b')
messages = env.load_json(path,n=100)
res = []
for x in messages:
    x['role'] = 'text'
    res.append(x)

result = env.train_json(res)

print("->", result)
