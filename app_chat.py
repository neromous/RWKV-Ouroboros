import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_CUDA_ON'] = '0'

import torch
import gc
import json
from bottle import route, run, template, request
import random
import sys
# 加载推理模型
from rwkv_model.model_chat import RWKV_RNN
from models.scene import Scene
from utils import save_data
# 加载推理实现
from models.inference_helper import InferenceWithState
# 用于解析org文件
from models.org_text import DataNode, file_to_node,text_to_node
import copy
from tqdm import tqdm

model_path = '/home/neromous/pyblackfog/rwkv-origin-models/world-7b.pth'
# 加载推理类
inferencer = InferenceWithState()
rwkv_rnn = None
state = None
init_state = None
not_cleane_yet = True
state_storage = {}
train_states =  None

@route('/inference/load-model', method='POST')
def load_model():
    global inferencer, model_engine,rwkv_rnn,state,init_state
    item = request.json
    if rwkv_rnn == None:
        rwkv_rnn = RWKV_RNN(model_path, 'cuda:1 fp16')
    state = None
    init_state = None
    return {"response": "model save"}

load_model()

@route('/inference/remove-model', method='POST')
def remove_model():
    global inferencer,state,init_state,rwkv_rnn
    item = request.json
    rwkv_rnn = None
    state = None
    init_state = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}

@route('/state/init', method='POST')
def state_init():
    global inferencer,rwkv_rnn,state,init_state
    state = None
    init_state = None
    item = request.json
    messages = item.get('messages',[])
    resp = []
    state = None
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg, state = inferencer.generate_by_vec(rwkv_rnn,msg,state=state)
        msg.save()
        resp.append(msg.json())
    init_state = copy.deepcopy(state).cpu()
    return {"messages": resp}

@route('/state/reset', method='POST')
def reset_state():
    global inferencer, state,init_state
    #print("\ninit_state",init_state)
    #print("\nstate->",state)
    state = copy.deepcopy(init_state)
    if init_state != None:
        state = state.cuda()
    print("\n====after reset======")
    #print("\ninit_state",init_state)
    #print("\nstate->",state)
    return {"messages": 'reset'}

@route('/state/save', method='POST')
def save_state():
    global inferencer, state_storage, state, init_state
    item = request.json
    state_name = item.get('save_state', 'default')
    state_storage[state_name] = copy.deepcopy(state).cpu()
    return {"messages": 'save-state'}

@route('/state/load', method='POST')
def load_state():
    global inferencer, state_storage, state, init_state
    item = request.json
    save_name = item.get('save_name', "default")
    load_name = item.get('load_name', "default")
    if state is not None:
        state_storage[save_name] = copy.deepcopy(state).cpu()
    if state_storage[load_name] is not None:
        state = copy.deepcopy(state_storage[load_name]).cuda()
    return {"messages": 'reset'}

@route('/inference/generate', method='POST')
def inference_generate():
    global inferencer,rwkv_rnn, state_storage, state, init_state
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        print("\n----",message)
    #print("\n---begin->", state)
    in_state = copy.deepcopy(state)
    for message in messages:
        msg = inferencer.scene.add_message(message)
        if msg.load_state != "default":
            state = copy.deepcopy(state_storage.get(msg.load_state,state)).cuda()
        msg, in_state = inferencer.generate_by_vec(rwkv_rnn, msg, state=in_state)
        if msg.save_state != "default" and state is not None:
            state_storage[msg.save_state] = copy.deepcopy(state).cpu()
        msg.save()
        resp.append(msg.json())
    state = copy.deepcopy(in_state)
    #print("\n---after->", state)
    return {"messages": resp}


run(host='0.0.0.0', port=3011)
