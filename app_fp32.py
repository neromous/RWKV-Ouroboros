import os
os.environ['RWKV_JIT_ON'] = "0"
os.environ['RWKV_TORCH_COMPILE'] = "0"
os.environ['RWKV_FLOAT_MODE'] = "32"
# my projects
from rwkv_model.model_origin import RWKV
from bottle import route, run, template, request
import torch
import gc
import deepspeed
import requests
from tqdm import tqdm
from models.page import Page
from models.org_text import DataNode,file_to_node,text_to_node, tokenizer,inference,inference_with_state
import random

local_path = "/home/neromous/Documents/blackfog"


origin_model = RWKV(f"{local_path}/resources/train-results/oneline/save-200.pth",
                    lr_init=2.0e-6,
                    dtype="fp32",
                    grad_cp=1)

optimizer, lr_scheduler = origin_model.configure_optimizers()

model, optimizer, _, _ = deepspeed.initialize(model=origin_model,
                                              optimizer=optimizer,
                                              lr_scheduler=lr_scheduler,
                                              config="fp32_ds_config.config")


@route('/train/by-org', method='POST')
def train_by_org():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    text = item['org']
    todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
    if not "".startswith("#+TODO"):
        text = todo +text
    coll = text_to_node(text)
    losses = []
    for k,v in coll.items():
        for token in v.yield_train_data(req_len=2049,window=0):
            m = model.training_step(token)
            loss = m.item()
            losses.append(loss)
            model.backward(m)
            model.step()
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": losses}

@route('/train/sft', method='POST')
def train_sft():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    coll = file_to_node('./data/sft.org')
    losses = []
    datasets = []
    for k,v in coll.items():
        datasets.append(v)
    start = datasets[0]
    end = datasets[1:]
    random.shuffle(end)
    datasets = [start] + end
    for v in datasets:
        for token in v.yield_train_data(req_len=2049,window=1024):
            m = model.training_step(token)
            loss = m.item()
            losses.append(loss)
            model.backward(m)
            model.step()
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": losses}


@route('/inference/by-org', method='POST')
def inference_by_org():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    temperautre = item.get('temperature', 0.1)
    top_p = item.get('top_p', 0.1)
    text = item['org']
    todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
    if not "".startswith("#+TODO"):
        text = todo +text
    coll = text_to_node(text)
    item_id = max(coll.keys())
    item = coll[item_id]
    print(item)
    output = inference(model,
                       item.tokens,
                       token_ban=[0],
                       token_stop=[65535],
                       token_count=256,
                       temperature = temperautre,
                       top_p = temperautre)
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": output}


@route('/inference/by-inf', method='POST')
def inference_by_inf():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    temperautre = item.get('temperature', 0.1)
    top_p = item.get('top_p', 0.1)
    text = item['org']
    todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
    if not "".startswith("#+TODO"):
        text = todo +text
    coll = text_to_node(text)
    item_id = max(coll.keys())
    item = coll[item_id]
    print("==start===" )
    res = []
    state = None
    text,state = inference_with_state(model,
                                      [65530],
                                      temperature=0.1,
                                      top_p=0.1,
                                      token_count=1,
                                      state=None)
    gc.collect()
    torch.cuda.empty_cache()

    return {"response": text}

@route('/inference/by-token', method='POST')
def inference_by_token():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    temperautre = item.get('temperature', 0.1)
    top_p = item.get('top_p', 0.1)
    text = item['org']

    output = inference(model,
                       item.tokens,
                       token_ban=[0],
                       token_stop=[65535],
                       token_count=256,
                       temperature = temperautre,
                       top_p = temperautre)
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": text}



@route('/train/save-weight', method='POST')
def save_weight():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    model.save_checkpoint(save_dir="./save.pth")
    # item = request.json
    # save_name = item.get('save_path','save.pth')
    # model.to(torch.device('cpu'))
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.save(model.module.state_dict(),
    #            f"{local_path}/resources/train-results/oneline/{save_name}")
    # gc.collect()
    # torch.cuda.empty_cache()
    # model.to(torch.device('cuda'))
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}



run(host='0.0.0.0', port=3000)


# @route('/train/by-token', method='POST')
# def train_by_token():
#     pass


# @route('/inference/by-token', method='POST')
# def inference_by_token():
#     global model
#     data = request.json
#     tokens = data.get('tokens',False)
#     token_count = data.get('token_count',False)
#     token_stop = data.get('token_stop',False)
#     token_stop = data.get('token_stop',False)
#     return ""




# @route('/model_weights/save', method='POST')
# def model_weights_save():
#     pass
