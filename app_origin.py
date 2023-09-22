import os
os.environ['RWKV_JIT_ON'] = "1"
os.environ['RWKV_TORCH_COMPILE'] = "0"
os.environ['RWKV_FLOAT_MODE'] = "fp16"
# my projects
from rwkv_model.model_origin import RWKV
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
import types



model = RWKV("/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth")

optimizer, lr_scheduler = model.configure_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config="ds_config_origin.config")

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
    infer_model.state = None
    for message in messages:
        msg = infer_model.scene.add_message(message)
        msg = infer_model.generate(msg, state=infer_model.state)
        msg.save()
        resp.append(msg.json())
    infer_model.init_state = copy.deepcopy(infer_model.state)
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
        # print(infer_model.state)
        # print(infer_model.init_state)
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


@route('/train/tokens', method='POST')
def train_tokens():
    global model_engine
    item = request.json
    # parse
    tokens = item['input_ids']
    mask = item.get('attention_mask',None)
    ctx_len = item.get('ctx_len',2048)
    req_len = ctx_len + 1
    dix = [0 for x in range(req_len)]
    dix[:req_len] = tokens[:req_len]
    if mask == None:
        mask = [1 for x in dix]
        mask = mask[:-1]

    batch = (torch.tensor([dix[:-1]],dtype=torch.long).to('cuda'),
             torch.tensor([dix[1:]],dtype=torch.long).to('cuda'),
             torch.tensor([mask], dtype=torch.float16).to('cuda'))
    m = model_engine(batch)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    # save_data(item)
    return {"loss": loss}


run(host='0.0.0.0', port=3000)
