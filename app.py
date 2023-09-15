from rwkv_model.model import RWKV
import torch
import deepspeed
from bottle import route, run, template, request
import json
from models.todo import Todo
from models.instructon import Instruction
from models.conversation import Message, Scene
from llm_datasets.sft import Sft
from common import TrainData

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

@route('/save-weight', method='POST')
def save_weight():
    global model_engine
    item = request.json
    model_engine.to(torch.device('cpu'))
    torch.save(model_engine.module.state_dict(), "save.pth")
    return {"response": "model save"}


@route('/train/tx-data', method='POST')
def train():
    global model_engine
    item = request.json
    # parse
    if type(item) == dict:
        train_data = TrainData(item)
    else:
        return {"message": "failed for unvalid data, request should be a dict"}
    batch = {"input_ids": train_data.to_tensor().to('cuda'),
             "attention_mask": None}
    m = model_engine.compute_loss(model_engine, batch, None, True)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    return {"loss": loss}

run(host='0.0.0.0', port=3000)
