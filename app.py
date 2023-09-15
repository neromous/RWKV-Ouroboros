from rwkv_model.model import RWKV
import torch
import deepspeed
from bottle import route, run, template, request
import json
from models.todo import Todo
from models.instructon import Instruction
from models.conversation import Message, Scene
from llm_datasets.sft import Sft



model = RWKV(load_model="/home/neromous/Documents/blackfog/resources/train-results/0.4b/rwkv-0.pth",
             n_embd= 1024,
             n_layer=24,
             vocab_size=65536,
             lr_init=1.0e-5,
             lr_final=1.0e-6,
             warmup_steps=4)

optimizer, lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config="ds_config.config",
                                                     )

@route('/train', method='POST')
def index():
    global model_engine
    data = request.json
    coll = Sft.all()
    coll = Sft.find_all(section_id=1)
    coll =Sft.to_tensor(coll)
    batch = {"input_ids": coll.to('cuda'),
             "attention_mask": None}
    m = model_engine.compute_loss(model_engine, batch, None, True)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    return {"loss" : loss}


run(host='0.0.0.0', port=3000)
