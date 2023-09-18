from rwkv_model.model import RWKV
import torch
import deepspeed
from bottle import route, run, template, request
import json
from models.instructon import Instruction
from models.conversation import Message, Scene
from llm_datasets.sft import Sft
from rwkv_model.inference import Inference

m = Inference(model_name="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth")

m.load_model()
n = m.scene.add_message({"text":"User: 你好啊","role":"user","token_count":0})
t = m.scene.add_message({"text":"Assistant: ",
                         "role":"robot",
                         "token_count":256,
                         "over":False})
msg = m.generate(n)
print(msg)
msg = m.generate(t)
print(msg)
print(m.state)
print(m.init_state)
# n = Todo.new({"title":"dddddd"})

# tt = Instruction.new({"instruction":"dfafdfa"})

# n.delete(4)

# print(Todo.find_by(title="dddddd"))

# ################################
# message = Message.new({"content" :"dfasddfa"})
# print(message)
# print(message.to_tokens())

# scene = Scene.new({"title":"测试用例"})
# scene.add_request({"text":"dfasdfs"})
# print(scene)
# with open('./data/sft.jsonl','r',encoding='utf-8') as f:
#     texts = f.readlines()

# data = []
# n = 0
# for text in texts:
#     items = json.loads(text)
#     i =  0
#     for item in items:
#         item['section_id'] = n
#         item['section_sort'] = i
#         Sft.new(item)
#         i += 1
#     n += 1

# m = Sft.all()
# print(m[-1].__dict__)
# print(m[-1].to_tokens())
# print(Sft.find_all(section_id=1))
