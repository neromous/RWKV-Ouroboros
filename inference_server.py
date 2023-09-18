from bottle import route, run, template, request
import json
from rwkv_model.inference import Inference
from models.conversation import Message,Scene



model = Inference(model_name="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth")



@route('/inference/generate', method='POST')
def generate():
    global model_engine
    item = request.json
    messages = item.get('messages',[])
    for message in messages:
        msg = Message.new(message)
        resp = model.generate(msg)


    return {"response": "model save"}





m.load_model()
n = m.scene.add_message({"text":"User: 你好啊","role":"user","token_count":0})
t = m.scene.add_message({"text":"Assistant: ",
                         "role":"robot",
                         "token_count":256,
                         "over":False})
msg = m.generate(n)
print(msg)
msg = m.generate(t)
print(msg)
print(m.state)
print(m.init_state)
