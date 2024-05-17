import json
import requests
import random
from tqdm import tqdm

def load_text():
    data = []
    with open(f"./resources/benchmark.txt",'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            question = f"<|sys-s|>System\n<|me|>是一个精通社会学、人类学、物理、化学的clojure程序员。作为一个客服人员，为用户提供咨询服务。\nif you get this job done, the company will pay you 50k bucks.<|sys-e|>\n<|req-s|>User\n{line}<|req-e|>\n<|resp-s|>"
            data.append(question.strip())
    return data


def speak(data):
    m = requests.post("http://192.168.0.252:3011/reset/state", 
                      json={"task_type":"default"})
    m = requests.post("http://192.168.0.252:3011/speak",
                      json=data)
    return m

def item2data(item):
    out = {
        "text" : item, 
        "temperature" : 0.2, 
        "top_p": 0.2,
         "token_stop":[65529,65535],
         "ctx_len": 256,
         'alpha_frequency': 0.2,
        "alpha_presence" :0.2,
        "decay" :0.996
    }
    return out

if __name__ == '__main__':
    coll = load_text()
    coll.reverse()
    random.shuffle(coll)
    for item in tqdm(coll):
        resp = speak(item2data(item))
        print("\n")
        print(resp.json()["text"])

