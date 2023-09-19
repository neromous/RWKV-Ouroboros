from bottle import route, run, template, request
import json
from rwkv_model.inference import Inference
from models.message import Message
from models.scene import Scene
import copy

model = Inference(model_name="./save.pth")
model.load_model()

@route('/state/init', method='POST')
def init():
    global model
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        msg = model.scene.add_message(message)
        msg = model.generate(msg, state=None)
        msg.save()
        resp.append(msg.json())
    model.init_state = copy.deepcopy(model.state)
    return {"messages": resp}

@route('/inference/generate', method='POST')
def generate():
    global model
    item = request.json
    messages = item.get('messages',[])
    resp = []
    for message in messages:
        msg = model.scene.add_message(message)
        msg = model.generate(msg, state=model.state)
        msg.save()
        resp.append(msg.json())
    return {"messages": resp}

@route('/state/reset', method='POST')
def reset_state():
    global model
    model.reset_state()
    print(model.state)
    print(model.init_state)
    return {"messages": 'reset'}

def load_model():
    pass

def reload_model():
    pass

def train_model():
    pass






run(host='0.0.0.0', port=7000)
