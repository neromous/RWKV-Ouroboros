from rwkv_model.model import RWKV
import torch
import deepspeed
from bottle import route, run, template, request
import json
from models.todo import Todo
from models.instructon import Instruction
from models.conversation import Message, Scene
from data_models.sft import Sft

n = Todo.new({"title":"dddddd"})

tt = Instruction.new({"instruction":"dfafdfa"})

n.delete(4)

print(Todo.find_by(title="dddddd"))

################################
message = Message.new({"content" :"dfasddfa"})
print(message)
print(message.to_tokens())

scene = Scene.new({"title":"测试用例"})
scene.add_request({"text":"dfasdfs"})
print(scene)
with open('./data/sft.jsonl','r',encoding='utf-8') as f:
    texts = f.readlines()

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

m = Sft.all()
print(m[-1].__dict__)
print(m[-1].to_tokens())
print(Sft.find_all(section_id=1))
