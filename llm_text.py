from llm_datasets.sft import Sft
from models.scene import Scene
from models.message import Message
import requests
from models.page import Page
import requests
from models.page import Page
from tqdm import tqdm
coll = Page.from_org('./data/sft.org')
train_data_set = []
for nodes in coll:
    cache = nodes[0]
    for node in nodes[1:]:
        cache = cache + node
    train_data_set.append(cache)

data = train_data_set[0] + train_data_set[1] + train_data_set[2]
print(data.tokens)

for i in tqdm(range(10)):
    m = requests.post("http://0.0.0.0:3000/train/tokens",
                      json={'input_ids': data.tokens[:2049] ,
                            'attention_mask':[1 for x in range(0,2048)]})

    print(m.json())


# with open('/mnt/database/Datasets/materials/bonsai/sample.txt','r', encoding='utf-8') as f:
#     texts = f.read()


# m = Page.new(texts,
#              prefix="",
#              postfix="",
#              prefix_token=[65530],
#              postfix_token=[65535]
#              )

#m = Page.from_txt('/mnt/database/Datasets/materials/bonsai/sample.txt', ctx=8192)

# print(m[0][300])
# print("=======")
# print(m[0][100:300])

# # print(m[0][100] + m[0][100:150])

# print(len(m[0].ctx_token))
# print(m[0].tensor.shape)

# print("====",m[0].encode(""))

# #读取jsonl
# m = Page.from_jsonl('./data/Sft.jsonl')
# for x in m:
#     print(x.org_node)  #转换为文本节点
#     with open("./data/sft.org", 'a',encoding='utf-8') as f:
#         f.write(x.org_node)


# 读取格式化文本
# coll = Page.from_org('./data/sft.org')
# result = []
# for nodes in coll:
#     print(nodes)
#     cache = nodes[0]
#     for node in nodes[1:]:
#         cache = cache + node
#     result.append(cache)
# print(result[-1])
# #print(result[-2].tokens)
# print(len(result))

# for x in t:
#     for y in x:
#         print(y.org_node) #解析成文本节点
#     print("======",len(x)) # 保存回去
#     with open("./data/sft-m.org", 'a',encoding='utf-8') as f:
#         f.write(x.org_node)

#print(m[0]+m[-1])
#print(m[0].__dict__.keys())

# m = Message.new({'text':texts,'ctx':4096,'prefix':'','postfix':''})

# for x in m.message_as_iter():
#     print(x)
#     print("======",len(x['tokens']))




# data = Sft
# data.load()

# print()
# messages = []
# for x in data.find_all(section_id=1):
#     item = x.__dict__
#     item.update({"token_count":256,"over": False})
#     messages.append(item)



# train_data = Scene.new({"messages": messages})


# m = requests.post("http://0.0.0.0:7000/inference/generate",
#                   json={"messages" : messages,

#                         })



# print(train_data)
