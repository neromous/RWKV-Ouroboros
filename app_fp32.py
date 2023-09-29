import os
from models import org_text
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
import copy
import requests
from tqdm import tqdm
from models.page import Page
from models.org_text import DataNode,file_to_node,text_to_node, tokenizer,inference,inference_with_state
import random
import types

local_path = "/home/neromous/Documents/blackfog"


origin_model = RWKV(f"{local_path}/resources/train-results/0.4b/rwkv-0.pth",
                    lr_init=1.0e-4,
                    dtype="fp32",
                    grad_cp=1,
                    n_embd=2560,
                    n_layer=32,
                    vocab_size=65536,
                    loaded = False)

optimizer, lr_scheduler = origin_model.configure_optimizers()

model, optimizer, _, _ = deepspeed.initialize(model=origin_model,
                                              optimizer=optimizer,
                                              lr_scheduler=lr_scheduler,
                                              config="fp32_ds_config.config")

model.load_checkpoint(load_dir="./save.pth")

rnn_model = None
rnn_init_state = None
rnn_state = None
args = types.SimpleNamespace()
args.n_layer = 32
args.n_embd = 2560

@route('/rnn/load_model', method='POST')
def rnn_load_model():
    global model,rnn_model
    item = request.json
    gc.collect()
    torch.cuda.empty_cache()
    weight = model.module.state_dict()
    rnn_model = RWKV_RNN(weight,args)
    return {"message":"load suceess"}


@route('/rnn/inference', method='POST')
def rnn__inference():
    global rnn_model,rnn_state, rnn_init_state
    item = request.json
    gc.collect()
    torch.cuda.empty_cache()
    if rnn_model == None:
        rnn_load_model()
    #================================
    tokens = item['tokens']
    org_text = item.get('org_text')
    token_count = item.get('token_count', 256)
    token_stop =  item.get('token_stop', [65535])
    token_ban = item.get('token_ban', [0])
    alpha_presence = item.get('alpha_presence', 0.45)
    alpha_frequency = item.get('alpha_frequency', 0.45)
    temperature=item.get('temperature', 0.1)
    top_p=item.get('top_p', 0.1)
    alpha_decay = item.get('alpha_decay', 0.996)
    reset_state = item.get('reset_state', True)
    init_state =  item.get('init_state', False)
    debug = item.get('debug',False)

    if not org_text.startswith("#+TODO"):
        todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
        text = todo + org_text
    coll = []
    try:
        coll = text_to_node(text)
    except:
        print("not valid")
    if debug:
        tokens =  tokenizer.encode("你好啊，请问你叫什么名字？\n" )
    #================================
    if reset_state:
        rnn_state = copy.deepcopy(rnn_init_state)
    out_str = ""
    occurrence = {}
    logits= []
    all_tokens = []
    out_last = 0
    for token in tokens:
        logits , rnn_state = rnn_model(token, rnn_state)
    if init_state:
        rnn_init_state = copy.deepcopy(rnn_state)
    for i in range(0,token_count):
        for n in token_ban:
            logits[n] = -float('inf')
        for n in occurrence:
            logits[n] -= (alpha_presence + occurrence[n] * alpha_frequency)

        token = RWKV.sample_logits(logits,
                                   temperature=temperature,
                                   top_p=top_p)
        if token in token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
           occurrence[xxx] *= alpha_decay
        if token not in occurrence:
           occurrence[token] = 1
        else:
           occurrence[token] += 1
        text = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in text: # only print when we have a valid utf-8 string
            print(text, end="", flush=True)
            out_str += text
            out_last = i + 1
        logits,rnn_state = rnn_model(token, rnn_state)
        print( text, end="", flush= True)
    # weight = model.module.state_dict()
    # rnn_model = RWKV_RNN(weight,args)
    return {"message":"load suceess"}

@route('/rnn/by-inf', method='POST')
def rnn_by_inf():
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
    tokens = tokenizer.encode("who are you? what's your name?")
    print("===",tokens)
    res = []
    state = None
    text,state = inference_with_state(model,
                                      tokens,
                                      temperature=0.2,
                                      top_p=0.2,
                                      token_count=256,
                                      state=None)
    gc.collect()
    torch.cuda.empty_cache()

    return {"response": text}



@route('/train/by-token', method='POST')
def train_by_token():
    global model,rnn_model
    rnn_model = None
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    tokens = item['tokens']
    losses = []
    while len(tokens) > 0 :
        token = tokens[:2049]
        tokens = tokens[2049 - 512:]
        m = model.training_step(token)
        loss = m.item()
        losses.append(loss)
        model.backward(m)
        model.step()
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": losses}


@route('/train/by-org', method='POST')
def train_by_org():
    global model,rnn_model
    rnn_model = None
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    text = item['org']
    todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
    if not text.startswith("#+TODO"):
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
    global model,rnn_model
    rnn_model = None
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    coll = file_to_node('./data/sft.org')
    losses = []
    total_loss = 0
    mean_loss = 0
    datasets = []
    for k,v in coll.items():
        datasets.append(v)
    start = datasets[0]
    end = datasets[1:]
    random.shuffle(end)
    datasets = [start] + end
    i = 0
    for v in datasets:
        for token in v.yield_train_data(req_len=2049,window=1024):
            i += 1
            m = model.training_step(token)
            loss = m.item()
            total_loss += loss
            mean_loss = total_loss / i
            print(f"-> item_loss {loss} batch_loss {mean_loss}")
            losses.append(loss)
            model.backward(m)
            model.step()
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": mean_loss}


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



@route('/inference/by-token', method='POST')
def inference_by_token():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    temperautre = item.get('temperature', 0.1)
    top_p = item.get('top_p', 0.1)
    tokens = item['tokens']
    output = inference(model,
                       tokens,
                       token_ban=[0],
                       token_stop=[65535],
                       token_count=256,
                       temperature = temperautre,
                       top_p = top_p)
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": output}



@route('/train/save-weight', method='POST')
def save_weight():
    global model
    gc.collect()
    torch.cuda.empty_cache()
    model.save_checkpoint(save_dir="../")
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
