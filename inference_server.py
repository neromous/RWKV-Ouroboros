from bottle import route, run, template, request
import json
from rwkv_model.inference import Inference
from models.message import Message
from models.scene import Scene

model = Inference(model_name="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth")
model.load_model()


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

def save_state():
    pass

def load_model():
    pass

def reload_model():
    pass


def train_model():
    pass






run(host='0.0.0.0', port=7000)
