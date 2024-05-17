import json
import requests
import random
from tqdm import tqdm

def train(data):
    m = requests.post("http://192.168.0.252:3011/learn/by/text",
                      json=data)
    return m

def get_ids():
    m = requests.post("http://localhost:3000/db/find-by",
                    #json={"query":"[:find [?e ...] :where [?e :g-atom/phrase ]]"}
                    json={"entity" : "g-atom/phrase"}
                    )
    coll = m.json()['coll']
    random.shuffle(coll)
    return coll[0:5000]

def item2txt(item):
    summary = item['g-atom/summary']
    phrase = item['g-atom/phrase']
    phrase = f"\n以下是关于词*{phrase}*的解释。"
    return phrase + "\n<|resp-s|>" + summary + "<|resp-e|>"

ids = get_ids()
coll = requests.post("http://localhost:3000/db/pull",
                  json={"ids" : ids})
coll = coll.json()['coll']
print(coll[0])

text = []
n = 0

for x in coll:
    n += 1
    t = item2txt(x)
    text.append(t)
    if n == 16:
        n = 0
        res = train({"text" : text})
        text = []
        print(res.json()['loss'])