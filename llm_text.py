import os
os.environ['RWKV_JIT_ON'] = "0"
os.environ['RWKV_TORCH_COMPILE'] = "0"
os.environ['RWKV_FLOAT_MODE'] = "fp32"
# my projects
from rwkv_model.model_origin import RWKV
from rwkv_model.model_infer import RWKV_RNN
from bottle import route, run, template, request
import torch
import gc
import deepspeed
import requests
from tqdm import tqdm
from models.page import Page
from models.org_text import DataNode,file_to_node,text_to_node, tokenizer,inference,inference_with_state
import random
import types

local_path = "/home/neromous/Documents/blackfog"

load_model = f"{local_path}/resources/train-results/oneline/save-200.pth"

origin_model = RWKV(load_model,
                    lr_init=1.0e-4,
                    dtype="fp32",
                    grad_cp=1,
                    n_layer=32,
                    n_embd=2560,
                    vocab_size=65536,
                    loaded = True)
#load_dict = torch.load(load_model, map_location="cpu")
#origin_model.load_state_dict(load_dict)
origin_model.to('cuda')
model_parameters,optimizer, lr_scheduler = origin_model.configure_optimizers()

model, optimizer, _, _ = deepspeed.initialize(model=origin_model,
                                              model_parameters=model_parameters,
                                              optimizer=optimizer,
                                              lr_scheduler=lr_scheduler,
                                              config="fp32_ds_config.config")

ds = model.module.state_dict()

# coll = file_to_node('./data/sft.org')
# datasets = []
# for k,v in coll.items():
#     datasets.append(v)

# for v in datasets[-1:]:
#     for token in v.yield_train_data(req_len=2049,window=1024):
#         m = model.training_step(token)
#         loss = m.item()
#         model.backward(m)
#         model.step()
# torch.cuda.empty_cache()
# gc.collect()

# model_dict = {}
# for n, lp in model.named_parameters():
#     model_dict[n]=lp
    #print(f"-> {n} -> {lp}")
# model_dict = model.named_parameters()
#load_dict = torch.load(load_model, map_location="cpu")
#cache_dict = torch.load(load_model, map_location="cpu")
#model_dict = {}
#for k,v in cache_dict.items():
#    model_dict[k] =  v.cuda()
tokens  = tokenizer.encode("hello, what's your name")
state = None
logits= None
args = types.SimpleNamespace()
args.MODEL_NAME = load_model
args.n_layer = 32
args.n_embd = 2560

infer_model = RWKV_RNN(ds,args)
for token in tokens:
    logits,state = infer_model(token, state)
for x in range(0,100):
    token = RWKV.sample_logits(logits)
    text = tokenizer.decode([token])
    logits,state = infer_model(token,state)
    print( text, end="", flush= True)


# for k,v in model_dict.items():
#     cache_dict[k] = v.cpu()

# for k,v in model_dict.items():
#     cache_dict[k] = v.cuda()

# for k,v in ds.items():
#     print(f"=={k}===={v}=")


# # for n, lp in model.named_parameters():
# #     print(f"=={n}=={lp}==")

# #model.load_checkpoint(load_dir="./save.pth")




# import os
# os.environ['RWKV_JIT_ON'] = "1"
# os.environ['RWKV_TORCH_COMPILE'] = "0"
# os.environ['RWKV_FLOAT_MODE'] = "fp16"
# # my projects
# from rwkv_model.model_origin import RWKV
# import torch
# import gc
# import deepspeed
# import requests
# from tqdm import tqdm
# from models.page import Page
# import random
# data_path = "/home/neromous/Documents/blackfog/resources/train-results/oneline/save-200.pth"
# data_path = "/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth"

# model = RWKV(data_path,lr_init=2.0e-6)

# optimizer, lr_scheduler = model.configure_optimizers()

# model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
#                                                      optimizer=optimizer,
#                                                      lr_scheduler=lr_scheduler,
#                                                      config="ds_config_origin.config")


# datasets = Page.from_org('./data/sft.org',shuffle=True)
# data = datasets[0]
# for x in datasets[1:]:
#     data += x
# avg_loss = 5.0
# total_loss = 0
# step = 0
# while avg_loss > 0.5:
#     for tokens in data.yield_token(2049):
#         step += 1
#         idx = tokens
#         m = model_engine.training_step(idx)
#         loss = m.item()
#         total_loss += loss
#         avg_loss = total_loss / step
#         model_engine.backward(m)
#         model_engine.step()
#         print(f"\n[item-loss:{loss}  avg-loss:{avg_loss}]")
#         if step % 20 == 0 or loss >= 2.0:
#             out = [x for x in tokens]
#             print(f"\n===question=={loss}===\n",Page.decode(out))
#             print("\nanswers->")
#             res = ""
#             all_tokens = [x for x in out]
#             token_ban = [0]
#             token_stop = [65530,65531,65532,65533,65534,65535]
#             token_coll = list(set(tokens))
#             out_last = 0
#             out_str = ''
#             occurrence = {}
#             alpha_presence = 0.45
#             alpha_frequency = 0.45
#             alpha_decay = 0.996
#             for i in range(256):
#                 #get logits
#                 output = model_engine.inference(all_tokens[-1024:])
#                 logits = output[-1]
#                 for n in token_ban:
#                     logits[n] = -float('inf')
#                 # for n in token_coll:
#                 #     logits[n] *= 1.5
#                 for n in occurrence:
#                    logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
#                 # get token
#                 token = model.sample_logits(logits)
#                 if token in token_stop:
#                     break
#                 all_tokens += [token]
#                 for xxx in occurrence:
#                    occurrence[xxx] *= alpha_decay
#                 if token not in occurrence:
#                    occurrence[token] = 1
#                 else:
#                    occurrence[token] += 1

#                 tmp = Page.decode(all_tokens[2049 + out_last:])

#                 if  '\ufffd' not in tmp:
#                     print(tmp, end="", flush=True)
#                     out_str += tmp
#                     out_last = i + 1
#             gc.collect()
#             torch.cuda.empty_cache()


#         if  step % 100 == 0 :
#             print("==save===")
#             model_engine.to(torch.device('cpu'))
#             torch.save(model_engine.module.state_dict(), f"/home/neromous/Documents/blackfog/resources/train-results/oneline/save-{step}.pth")
#             model_engine.to(torch.device('cuda'))

#     print("===load data===")
#     datasets = Page.from_org('./data/sft.org',shuffle=True)
#     data = datasets[0]
#     for x in datasets[1:]:
#         data += x



# coll = Page.from_org('./data/sft.org')
# train_data_set = []
# for nodes in coll:
#     cache = nodes[0]
#     for node in nodes[1:]:
#         cache = cache + node
#     train_data_set.append(cache)

# data = train_data_set[0] + train_data_set[1] + train_data_set[2]
# print(data.tokens)

# for i in tqdm(range(10)):
#     m = requests.post("http://0.0.0.0:3000/train/tokens",
#                       json={'input_ids': data.tokens[:2049] ,
#                             'attention_mask':[1 for x in range(0,2048)]})

#     print(m.json())


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
