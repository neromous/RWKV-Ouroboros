import requests
from tqdm import tqdm

with open('/home/neromous/Downloads/color.txt','r', encoding='utf-8') as f:
    texts = f.read()

coll = []



while len(texts) > 0 :
    text = texts[:2048]
    texts = texts[1920:]
    m = {"role":"text",
         "text": text,
         "prefix":"\n下面是《碟形世界的片段》 具体章节如下\n\n",
         "postfix": ""
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
