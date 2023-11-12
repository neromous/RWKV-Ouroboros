import os
from utils import log, load_config
import torch
import gc
import deepspeed
import json
from bottle import route, run, template, request
import random
import sys

config = load_config()
#===============pico 配置项=================
# os.environ['RWKV_JIT_ON'] = config['environ']['RWKV_JIT_ON']
os.environ['RWKV_FLOAT_MODE'] = config['environ']['RWKV_FLOAT_MODE']
os.environ['RWKV_MY_TESTING'] = config['environ']['RWKV_MY_TESTING']
# 对logits 正则化 防止logits过大
os.environ['WN_FIX_L2WRAP'] = config['environ']['WN_FIX_L2WRAP']
# use window when you use fixed ctx.
# when you use infctx that value should be 0
window = config['trainer']['window']
min_loss_fix = config['trainer']['min_loss_fix']
max_loss_fix = config['trainer']['max_loss_fix']
min_loss = config['trainer']['min_loss']
max_loss = config['trainer']['max_loss']
proj_dir = config['proj_dir']
ctx_len = int(config['model']['ctx_len'])
ctx_parts = int(config['trainer']['ctx_parts'])

if config['infctx_on']:
    # 推荐值 128-256， 其state值是跨batch传递的。
    os.environ["RWKV_T_MAX"] = str(ctx_len)
    from rwkv_model.model_infctx_v2 import RWKV
else:
    # 不推荐分段
    os.environ['RWKV_STATE'] = config['environ']['RWKV_STATE']
    os.environ['RWKV_PARTS'] = str(ctx_parts)
    # 不推荐设置RWKV_PARTS的值。该功能会在后面的版本移除，由infctx模式替代。
    if os.environ['RWKV_PARTS'] != "0" :
        os.environ["RWKV_T_MAX"] = str((ctx_len+ctx_parts-1) // ctx_parts)
    else:
        # 推荐值 2048 在24g下
        os.environ["RWKV_T_MAX"] = str(ctx_len)
    from rwkv_model.model_v5 import RWKV

# 使用不同的ds——config文件
if config['environ']['RWKV_FLOAT_MODE'] == "fp32":
    ds_config =  "./ds_config/fp32_ds_config.config"
elif config['environ']['RWKV_FLOAT_MODE'] == "fp16":
    ds_config =  "./ds_config/fp16_ds_config.config"
elif config['environ']['RWKV_FLOAT_MODE'] == "bf16":
    ds_config =  "./ds_config/bf16_ds_config.config"


# 加载推理模型
from rwkv_model.model_infer import RWKV_RNN
from models.scene import Scene
from utils import save_data
# 加载推理实现
from models.inference_helper import InferenceWithState
# 用于解析org文件
from models.org_text import DataNode, file_to_node,text_to_node
import copy
from tqdm import tqdm

model = RWKV(load_model=config['model_path'],
             n_layer= config['model']['n_layer'],
             ctx_len= config['model']['ctx_len'],
             n_embd= config['model']['n_embd'],
             vocab_size = config['model']['vocab_size'],
             dropout = config['trainer']['dropout'],
             grad_cp = config['trainer']['grad_cp'],
             lora = config['lora'],
             lr_init=1.0e-6,
             lr_final=1.0e-6,
             dtype =  config['environ']['RWKV_FLOAT_MODE'],
             warmup_steps=config['trainer']['warmup_steps'])

# 加载优化器
optimizer, lr_scheduler = model.get_optimizers()

# 初始化deepspeed
model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config=ds_config)

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
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())
    state = None
    init_state = None
    return {"response": "model save"}

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
    if rwkv_rnn == None:
        load_model()
    item = request.json
    messages = item.get('messages',[])
    resp = []
    state = None
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg, state = inferencer.generate(rwkv_rnn,msg,state=state)
        msg.save()
        resp.append(msg.json())
    init_state = copy.deepcopy(state).cpu()
    return {"messages": resp}

@route('/state/reset', method='POST')
def reset_state():
    global inferencer, state,init_state
    print("\ninit_state",init_state)
    print("\nstate->",state)
    state = copy.deepcopy(init_state)
    if init_state != None:
        state = state.cuda()
    print("\n====after reset======")
    print("\ninit_state",init_state)
    print("\nstate->",state)
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

@route('/inference/generate-by-token', method='POST')
def inference_generate_by_token():
    global inferencer,rwkv_rnn, state_storage, state, init_state
    if rwkv_rnn == None:
        load_model()
    prompt = request.json
    load_state_name = prompt.get('load_state','default')
    save_state_name = prompt.get('save_state','default')
    if not prompt or type(prompt) is not dict:
        return {'error':'no avalible prompt'}
    prompt_state = copy.deepcopy(state)
    #prompt = inferencer.scene.add_prompt(prompt)
    if load_state_name != "default":
        state = copy.deepcopy(state_storage.get(load_state_name,state)).cuda()
    prompt, prompt_state = inferencer.generate_by_token(rwkv_rnn, prompt, state=prompt_state)
    if save_state_name != "default" and state is not None:
        state_storage[save_state_name] = copy.deepcopy(state).cpu()
    state = copy.deepcopy(prompt_state)
    return {"prompt": prompt}

@route('/inference/generate', method='POST')
def inference_generate():
    global inferencer,rwkv_rnn, state_storage, state, init_state
    if rwkv_rnn == None:
        load_model()
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        print("\n----",message)
    print("\n---begin->", state)
    in_state = copy.deepcopy(state)
    for message in messages:
        msg = inferencer.scene.add_message(message)
        if msg.load_state != "default":
            state = copy.deepcopy(state_storage.get(msg.load_state,state)).cuda()
        msg, in_state = inferencer.generate(rwkv_rnn, msg, state=in_state)
        if msg.save_state != "default" and state is not None:
            state_storage[msg.save_state] = copy.deepcopy(state).cpu()
        msg.save()
        resp.append(msg.json())
    state = copy.deepcopy(in_state)
    print("\n---after->", state)
    return {"messages": resp}

@route('/inference/generate-no-state', method='POST')
def inference_generate_no_state():
    global inferencer,model
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate_no_state(model, msg)
        msg.save()
        resp.append(msg.json())
    return {"messages": resp}


@route('/train/save-weight', method='POST')
def save_weight():
    global model_engine,model
    item = request.json
    model_name = item.get("model_name","default")
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(model_engine.module.state_dict())
    # ===============save=================
    fpath = f"{proj_dir}/{model_name}.pth"
    torch.save(model.state_dict(), fpath)
    print(f"===saved=={fpath}==")
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}


@route('/train/tx-data', method='POST')
def train_tx_data():
    global model_engine, ctx_len, window,train_states
    item = request.json
    gc.collect()
    torch.cuda.empty_cache()
    # parse
    if type(item) == dict:
        train_data = Scene.new(item)
    else:
        return {"message": "failed for unvalid data, request should be a dict"}
    total = 0
    mean_loss = 0
    i = 0
    data_iter = train_data.yield_tokens(ctx_len=ctx_len, window=window)
    if train_states is not None:
        states = copy.deepcopy(train_states)
    else:
        states = train_states
    for token in data_iter:
        print(f'==len==>{len(token)}')
        if len(token) < 3 :
            break
        i += 1
        batch = {"input_ids": token,
                 "attention_mask": None}
        m, states = model_engine.training_step(batch, states = states )
        loss = m.item()
        if loss < min_loss:
            m = m * min_loss_fix
        elif loss > max_loss:
            m = m * max_loss_fix
        total += loss
        mean_loss = total / i
        model_engine.backward(m)
        model_engine.step()
        print(f"\nmean-loss->{mean_loss}")
    train_states = states
    states = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": mean_loss}


@route('/train/by-token', method='POST')
def train_by_token():
    global model_engine, ctx_len, window,train_states
    item = request.json
    origin_tokens = item.get('token') or item.get('tokens')
    if len(origin_tokens) <2:
        return {"message":"no tokens"}

    gc.collect()
    torch.cuda.empty_cache()
    total = 0
    mean_loss = 0
    i = 0
    tokens = []
    while len(origin_tokens) > 0:
        out = origin_tokens[:ctx_len + 1]
        origin_tokens = origin_tokens[ctx_len + 1:]
        tokens.append(out)
    if train_states is not None:
        states = copy.deepcopy(train_states)
    else:
        states = train_states
    for token in tokens:
        print(f'==len==>{len(token)}')
        if len(token) < 3:
            break
        i += 1
        batch = {"input_ids": token,
                 "attention_mask": None}
        m, states = model_engine.training_step(batch, states = states )
        loss = m.item()
        if loss < min_loss:
            m = m * min_loss_fix
        elif loss > max_loss:
            m = m * max_loss_fix
        total += loss
        mean_loss = total / i
        model_engine.backward(m)
        model_engine.step()
        print(f"\nmean-loss->{mean_loss}")
    train_states = copy.deepcopy(states)
    states = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": mean_loss}


if config['debug'] :
    messages = [{"text" :"你好啊",
                 "role" : "text",
                 "over":False, "token_count":128 }]


    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate_no_state(model, msg)
        print("=====msg",msg)
    pass

    print("===train test start==")
    train_data = [x for x in range(0,6144)]
    batch = {"input_ids": train_data,
             "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    gc.collect()
    torch.cuda.empty_cache()

    print("===train test over==",loss)
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())

    messages = [{"text" :"你好啊", "role" : "text","over":False, "token_count":128 } ]
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate(rwkv_rnn,msg)
        print("==msg==",msg)
    print("===train test start==")
    rwkv_rnn = None
    gc.collect()
    torch.cuda.empty_cache()

    train_data = [x for x in range(0,6144)]
    batch = {"input_ids": train_data,
             "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    gc.collect()
    torch.cuda.empty_cache()
    print("===train test over==",loss)
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate(rwkv_rnn,msg)
        print("==msg==",msg)

    rwkv_rnn = None
    gc.collect()
    torch.cuda.empty_cache()

    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    print("===train test over==",loss)


if not config['debug']:
    run(host='0.0.0.0', port=config['port'])
