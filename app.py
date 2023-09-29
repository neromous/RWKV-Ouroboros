import os
from utils import log, load_config
import torch
import gc
import deepspeed
import json
from bottle import route, run, template, request
config = load_config()
#===============pico 配置项=================
os.environ['RWKV_JIT_ON'] = config['environ']['RWKV_JIT_ON']
os.environ['RWKV_FLOAT_MODE'] = config['environ']['RWKV_FLOAT_MODE']

if config['infctx_on']:
    if config['infctx_type'] == "wani-boat":
        ctx_parts =  config['trainer']['ctx_parts']
        ctx_len = config['model']['ctx_len']
        os.environ['RWKV_PARTS'] = str(ctx_parts)
        os.environ['RWKV_STATE'] = config['environ']['RWKV_STATE'] #开启后编译WKV_STATE的cuda kernal
        os.environ["RWKV_T_MAX"] = str((ctx_len+ctx_parts-1) // ctx_parts)
        ds_config =  "./offload_ds_config.config"
        from rwkv_model.model_state import RWKV
    elif config['infctx_type'] == "pico":
        os.environ['RWKV_TORCH_COMPILE'] = config['environ']['RWKV_TORCH_COMPILE']
        from rwkv_model.model import RWKV
        ds_config =  "./ds_config.config"
else:
    from rwkv_model.model_origin import RWKV
    ds_config =  "./fp32_ds_config.config"


from rwkv_model.model_infer import RWKV_RNN
from models.scene import Scene
from models.page import Page
from utils import save_data
from inference.core import InferenceWithState
import copy
from tqdm import tqdm

model = RWKV(load_model=config['model_path'],
             n_layer= config['model']['n_layer'],
             n_embd= config['model']['n_embd'],
             vocab_size = config['model']['vocab_size'],
             grad_cp = config['trainer']['grad_cp'],
             lora = config['lora'],
             lr_init=1.0e-4,
             lr_final=1.0e-6,
             dtype =  config['environ']['RWKV_FLOAT_MODE'],
             warmup_steps=config['trainer']['warmup_steps'])

optimizer, lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config=ds_config)

infer_model = InferenceWithState(model_engine)



@route('/inference/load-model', method='POST')
def load_model():
    global infer_model, model_engine
    item = request.json
    infer_model.clean_model()
    gc.collect()
    torch.cuda.empty_cache()
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


@route('/train/by-org-text', method='POST')
def train_by_org_text():
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


if config['debug'] :
    print("===train test start==")
    train_data = [x for x in range(0,4096)]
    batch = {"input_ids": train_data,
             "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    print("===train test over==",loss)
    infer_model.load_model()
    messages = [{"text" :"你好啊", "role" : "text","over":False, "token_count":128 } ]
    for message in messages:
        msg = infer_model.scene.add_message(message)
        msg = infer_model.generate(msg)
        print("==msg==",msg)

    print("===train test start==")
    train_data = [x for x in range(0,4096)]
    batch = {"input_ids": train_data,
             "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    print("===train test over==",loss)
    for message in messages:
        msg = infer_model.scene.add_message(message)
        msg = infer_model.generate(msg)
        print("==msg==",msg)



if not config['debug']:
    run(host='0.0.0.0', port=3000)
