import os
os.environ['RWKV_JIT_ON'] = "1"
os.environ['RWKV_TORCH_COMPILE'] = "0"

# my projects
from rwkv_model.model import RWKV
import torch
import gc
import deepspeed
from bottle import route, run, template, request
import json
from models.scene import Scene
from models.page import Page
from utils import save_data
from rwkv_model.inference import Inference
import copy
from tqdm import tqdm

model = RWKV(load_model="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth",
             #n_embd=2560,
             #n_layer=32,
             #vocab_size=65536,
             lr_init=1.0e-4,
             lr_final=1.0e-6,
             warmup_steps=4)

optimizer, lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config="ds_config.config")

infer_model = Inference(model_name="./save.pth")


@route('/inference/load-model', method='POST')
def load_model():
    global infer_model, model_engine
    item = request.json
    in_cpu = item.get('in_cpu')
    infer_model.clean_model()
    gc.collect()
    torch.cuda.empty_cache()
    infer_model.model_weights =  model_engine.module.state_dict()
    infer_model.load_model()
    return {"response": "model save"}


@route('/inference/remove-model', method='POST')
def remove_model():
    global infer_model
    item = request.json
    infer_model.clean_model()
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}


@route('/state/init', method='POST')
def init():
    global infer_model
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        msg = infer_model.scene.add_message(message)
        msg = infer_model.generate(msg, state=infer_model.state)
        msg.save()
        resp.append(msg.json())
    infer_model.set_init_state()
    print(infer_model.state)
    print(infer_model.init_state)
    return {"messages": resp}


@route('/state/reset', method='POST')
def reset_state():
    global infer_model
    print(infer_model.state)
    print(infer_model.init_state)
    infer_model.reset_state()
    return {"messages": 'reset'}


@route('/inference/generate', method='POST')
def generate():
    global infer_model
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        msg = infer_model.scene.add_message(message)
        msg = infer_model.generate(msg, state=infer_model.state)
        msg.save()
        resp.append(msg.json())
    return {"messages": resp}


@route('/train/weight-to-cpu', method='POST')
def weight_to_cpu():
    global model_engine
    item = request.json
    model_engine.to(torch.device('cpu'))
    model_engine.zero_optimization.free_bf16_param_memory()
    return {"response": "model save"}


@route('/train/weight-to-cpu', method='POST')
def weight_to_cuda():
    global model_engine
    item = request.json
    model_engine.to(torch.device('cuda'))
    return {"response": "model save"}


@route('/train/save-weight', method='POST')
def save_weight():
    global model_engine
    item = request.json
    model_engine.to(torch.device('cpu'))
    torch.save(model_engine.module.state_dict(), "save.pth")
    model_engine.to(torch.device('cuda'))
    return {"response": "model save"}


@route('/train/tx-data', method='POST')
def train():
    global model_engine
    item = request.json
    # parse
    if type(item) == dict:
        train_data = Scene.new(item)
    else:
        return {"message": "failed for unvalid data, request should be a dict"}
    batch = {"input_ids": train_data.to_tensor().to('cuda'),
             "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    # save_data(item)
    return {"loss": loss}

@route('/train/token', method='POST')
def train_token():
    global model_engine
    item = request.json
    input_ids = item['input_ids']
    attention_mask = item.get('attention_mask',None)
    batch = { "input_ids": torch.tensor([input_ids],dtype=torch.long).to('cuda'),
              "attention_mask": torch.tensor([attention_mask],dtype=torch.bfloat16).to('cuda')}
    m = model_engine.training_step(batch,model_engine=model_engine)
    loss = m.item()
    print("->", loss)
    model_engine.backward(m)
    model_engine.step()
    # save_data(item)
    return {"loss": loss}


@route('/train/org', method='POST')
def train_org():
    global model_engine
    item = request.json
    text = item['org']
    coll = Page.org_parser(text)
    train_data_set = []
    for nodes in coll:
        cache = nodes[0]
        for node in nodes[1:]:
            cache = cache + node
        train_data_set.append(cache)
        print(cache)
    losses = []
    for train_data in tqdm(train_data_set):
        batch = {"input_ids": train_data.tensor.to('cuda'),
                 "attention_mask": None}
        m = model_engine.training_step(batch,
                                       model_engine=model_engine)
        loss = m.item()
        print("->", loss)
        losses.append(loss)
        model_engine.backward(m)
        model_engine.step()
    # save_data(item)
    return {"loss": losses}


@route('/train/sft', method='POST')
def train_sft():
    global model_engine
    item = request.json
    coll = Page.from_org('./data/sft.org')
    train_data_set = []
    for nodes in coll:
        cache = nodes[0]
        for node in nodes[1:]:
            cache = cache + node
        train_data_set.append(cache)
    losses = []
    for train_data in tqdm(train_data_set):
        batch = {"input_ids": train_data.tensor.to('cuda'),
                 "attention_mask": None}
        m = model_engine.training_step(batch,model_engine=model_engine)
        loss = m.item()
        print("->", loss)
        losses.append(loss)
        model_engine.backward(m)
        model_engine.step()
    # save_data(item)
    return {"loss": losses}


run(host='0.0.0.0', port=3000)
