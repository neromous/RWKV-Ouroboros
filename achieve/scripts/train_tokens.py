import requests
from models.page import Page

coll = Page.from_org('./data/sft.org')
train_data_set = []
for nodes in coll:
    cache = nodes[0]
    for node in nodes[1:]:
        cache = cache + node
    train_data_set.append(cache)

data = train_data_set[0]

m = requests.post("http://0.0.0.0:3000/train/tokens",
                  json={'input_ids': data.tokens[:2049],
                        'attention_mask':[1 for x in range(0,2048)]})

print(m.json())
