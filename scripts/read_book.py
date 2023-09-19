import requests
from tqdm import tqdm

with open('/mnt/database/Datasets/materials/bonsai/sample.txt','r', encoding='utf-8') as f:
    texts = f.read()

coll = []

while len(texts) > 0 :
    text = texts[:4096]
    texts = texts[3200:]
    m = {"role":"think",
         "text": text,
         "prefix":"\n",
         "postfix": "\n"
         }
    coll.append(m)

for item in tqdm(coll):
    m = requests.post("http://0.0.0.0:3000/train/tx-data",
                      json={"messages": [item],
                            "prefix": "\n",
                            "postfix": "\n",
                            "prefix_token": [],
                            "postfix_token": [],
                            })
    print(m.json())
