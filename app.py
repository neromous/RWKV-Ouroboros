import os, warnings, math, datetime, sys, time
os.environ["RWKV_JIT_ON"] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader
import deepspeed
from src.dataset_finetune import MyDataset
import types
from tqdm import tqdm
import random
# ---------- 加载tokenizer ----------
from torch.nn import functional as F
from src.tokenizer import Tokenizer
from src.model_run import RWKV_RNN,rwkv_generate
import copy
import time
import cmd
import src.config as config
import gc
import json
from tqdm import tqdm
from bottle import route, run, template, request
import json

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def init_args(args):
    if args.load_model == "":
        args.load_model = get_model(args.proj_dir)
    args.epoch_begin = next_model(args.proj_dir)
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)
    return args

def my_func(tmp):
    print(tmp, end="", flush=True)


tokenizer = Tokenizer()

from argparse import ArgumentParser
from utils import get_model,next_model

train_args = types.SimpleNamespace()

for k, v in config.read_config()['trainer'].items():
    setattr(train_args, k, v)

for k, v in config.read_config()['model'].items():
    setattr(train_args, k, v)

args = init_args(train_args)
os.environ['RWKV_FLOAT_MODE'] = "fp16"
os.environ['RWKV_T_MAX'] = str(args.ctx_len)
os.environ['RWKV_MY_TESTING'] = ""

infer_args = types.SimpleNamespace()
for k, v in config.read_config()['model'].items():
    setattr(infer_args, k, v)

for k, v in config.read_config()['inference']['service_config'].items():
    setattr(infer_args, k, v)

from src.model import RWKV


#########################################################################
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# allow tf32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
##################################################################################
train_data = MyDataset(train_args)
train_args.data_file = train_args.text_book
textbook = MyDataset(train_args)
args.vocab_size = train_data.vocab_size
model = RWKV(train_args)
load_dict = torch.load(args.load_model, map_location="cpu")
model.load_state_dict(load_dict)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.configure_optimizers(),
                                                     config="ds_config.config",
                                                     device_id=0)



infer_model = None
infer_state = None

def train(model_engine,data):
    loss = model_engine(data)
    model_engine.backward(loss )
    model_engine.step()
    return loss.item()


def my_func(tmp):
    print(tmp, end="", flush=True)


@route('/textbook',method='POST')
def train_batch():
    global model_engine
    resp = request.json
    batch_ids = resp["batch_ids"]
    prefix = resp.get("prefix","")
    role = resp.get("role","system")
    pos = resp.get("pos",0)
    poe = resp.get("poe",0)
    step = resp.get("step",16)
    losses = []
    while len(batch_ids) != 0 and step != 0:
        loss = train(model_engine,textbook.__getitem__(batch_ids[0],
                                                       window=False,
                                                          pos=pos,
                                                          poe=poe,
                                                          prefix=prefix,
                                                          prefix_role=role,
                                                          debug=False
                                                          ))

        losses.append(loss)
        if textbook.is_end():
            batch_ids = batch_ids[1:]
        step =  step - 1
        print(f"\n-> loss:{loss} remain: {len(train_data.item)} ")

    torch.cuda.empty_cache()
    gc.collect()
    return {"loss" : losses}



@route('/train',method='POST')
def train_batch():
    global model_engine
    resp = request.json
    batch_ids = resp["batch_ids"]
    prefix = resp.get("prefix","")
    role = resp.get("role","system")
    pos = resp.get("pos",0)
    poe = resp.get("poe",0)
    step = resp.get("step",16)
    losses = []
    while len(batch_ids) != 0 and step != 0:
        loss = train(model_engine, train_data.__getitem__(batch_ids[0],
                                                          window=False,
                                                          pos=pos,
                                                          poe=poe,
                                                          prefix=prefix,
                                                          prefix_role=role,
                                                          debug=False
                                                          ))

        losses.append(loss)
        if train_data.is_end():
            batch_ids = batch_ids[1:]
        step =  step - 1
        print(f"\n-> loss:{loss} remain: {len(train_data.item)} steps：{step}")

    torch.cuda.empty_cache()
    gc.collect()
    return {"loss" : losses}



@route('/save-weight')
def save_weight():
    global infer_model
    model_engine.to(torch.device('cpu'))
    torch.save(model_engine.module.state_dict(), "save.pth")
    return {"response": "model save"}


@route('/load-model')
def load_model():
    global infer_model
    infer_model = None
    torch.cuda.empty_cache()
    gc.collect()
    ds = model_engine.module.state_dict()
    cache_dict = {}
    for k,v in ds.items():
        cache_dict[k] = v.cpu()
    infer_model = RWKV_RNN(infer_args, cache_dict)
    return {"response": "model loadd suceess"}


@route('/unload-model')
def unload_model():
    global infer_model
    infer_model = None
    torch.cuda.empty_cache()
    gc.collect()
    #  load to inference
    return {"response": "model loadd suceess"}


@route('/reset-state')
def reset_state():
    global infer_state
    infer_state = None
    torch.cuda.empty_cache()
    gc.collect()
    #  load to inference
    return {"response": "model loadd suceess"}


@route('/inference', method='POST')
def inference():
    global infer_model,infer_state
    text = request.json
    if infer_model == None:
        load_model()
    message = config.read_config()['inference']['prompt_config']
    results = []
    m = []
    for x in text['messages']:
        message.update(x)
        print("\n->",message["content"])
        res,infer_state = rwkv_generate(infer_model,message,state=infer_state)
        results.append(res)
    return {"results": results}

run(host='0.0.0.0', port=3000)



# while True:
#     for data in tqdm(train_data):
#         train(model_engine,data)
    #     if flag:

    #     flag = False
    #     n += 1

    # # loss
    # loss = model_engine(data)
    # # 更新标志
    # if loss.item() < 0.1:
    #     flag = True
    # print('\nloss->',loss.item())
    # model_engine.backward(loss )
    # model_engine.step()

    # if n ==  16:
    #     from tqdm import tqdm
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     #  load to inference
    #     ds = model_engine.module.state_dict()
    #     cache_dict = {}
    #     for k,v in ds.items():
    #         cache_dict[k] = v.cpu()

    #     model_inf = RWKV_RNN(infer_args, cache_dict)

    #     # load default message
    #     message = config.read_config()['inference']['prompt_config']
    #     m = []
    #     role = {"content": "测试",
    #             "over": True,
    #             "role" : "user",
    #             "token_count": 0
    #             }
    #     message.update(role)
    #     print("\n->",message["content"])
    #     res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func)

    #     m.append(res)
    #     new_state = state.clone()

    #     role = {"content": "",
    #             "over": False,
    #             "role" : "robot",
    #             "token_count": 256
    #             }
    #     message.update(role)
    #     res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func,state=new_state)
    #     m.append(res)


    #     for x in m:
    #         with open("./bonsai.jsonl","a",encoding="utf-8") as f:
    #             f.write(json.dumps(x,ensure_ascii=False))


    #     i = 0
    #     model_inf = None
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     n = 0
