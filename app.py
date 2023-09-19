# my projects
from rwkv_model.model import RWKV
import torch
import gc
import deepspeed
from bottle import route, run, template, request
import json
from models.scene import Scene
from utils import save_data
from rwkv_model.inference import Inference
import copy


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
    infer_model.reset_state()
    print(infer_model.state)
    print(infer_model.init_state)
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
    m = model_engine.training_step(batch, None,model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    save_data(item)
    return {"loss": loss}




run(host='0.0.0.0', port=3000)
