import requests
import sys
import orgparse
from orgparse import loads
from orgparse.node import OrgRootNode, OrgNode
import sys
import re
import types
from tqdm import tqdm
import random
import json

class Inference:
    _config = {'3b': ["172.16.2.197", "40011", ["", "", ""]],
               '7b': ["172.16.2.197", "3000", ["","" , ""]]}
    _interface = {'generate': '/inference/generate',
                  'generate-no-state': '/inference/generate-no-state',
                  'reset-state': '/state/reset',
                  'load-model': '/inference/load-model',
                  'save-weight': '/train/save-weight',
                  'train': '/train/tx-data',
                  'train-sft': '/train/sft'}
    _org_head = "#+TODO: REQUEST RESPONSE SYSTEM ANALYSIS CLAUDE TEXT BOOK INSTRUCTION ANSWR THINK  | CANCELED\n"

    def __init__(self,model_name="3b"):
        self.model_name = model_name
        self.master = self._config[model_name][-1][0]
        self.robot = self._config[model_name][-1][-1]
        self.think = self._config[model_name][-1][1]

    def load_json(self,path, n = 50):
        with open(path,'r',encoding='utf-8') as f:
            data = f.readlines()
        data = random.sample(data, n)
        messages = [json.loads(x) for x in data]
        return messages

    def train_json(self, messages):
        losses = []
        for message in tqdm(messages):
            print(f"---> {message}")
            loss = self.teach([message])
            print(f"-> {loss}")
            losses.append(loss)
        return losses

    @classmethod
    def node2item(cls,node: OrgNode):
        item = types.SimpleNamespace()
        item.role = (node.todo or "text").lower().strip()
        item.level = node.level or -1
        item.prefix = node.get_property('Prefix', '')
        item.postfix = node.get_property('Postfix', '')
        #item.load_state = node.get_property('LoadState', 'default')
        #item.save_state = node.get_property('SaveState', 'default')
        item.temperature = float(node.get_property('temperature', '0.2'))
        item.top_p = float(node.get_property('TopP', '0.2'))
        #item.alpha_frequency = float(node.get_property('AlphaFrequency', '0.45'))
        #item.alpha_presence = float(node.get_property('AlphaPresence', '0.45'))
        #item.alpha_decay = float(node.get_property('AlphaDecay', '0.996'))
        item.min_loss = float(node.get_property('MinLoss', 0.5))
        item.train_time = int(node.get_property('TrainTimes', 1))
        item.token_count = int(node.get_property('TokenCount', 0))
        item.token_ban = node.get_property('TokenBan', " 0 ")
        item.token_ban = [int(x) for x in item.token_ban.split(" ") if x != '']
        item.token_stop = node.get_property('TokenStop', "65535 65530")
        item.token_stop = [int(x) for x in item.token_stop.split(" ") if x != '']
        item.next_role = node.get_property('NextRole', 'request')
        item.over = node.get_property('isOver', 'True')
        if item.over == "False":
            item.over = False
        else:
            item.over = True
        item.priority = node.priority or 0
        title = node.heading.strip()
        body = node.body.strip()
        item.text = title + "\n" + body
        item.text = item.text.strip()
        item.tags = [x for x in node.tags]
        item.shallow_tags = [x for x in node.shallow_tags]
        return item


    def init_prompt(self) -> list:
        message = [{"role": "system",
                    "text": "伊芙是一个聪明的ai",
                    "token_count": 0,
                    "over": True
                    }]
        return message


    def text2message(self, text):
        try:
            text = self._org_head + text
            messages = []
            nodes = loads(text)
            for node in nodes[1:]:
                item = self.node2item(node).__dict__
                messages.append(item)
            assert len(messages) >0
        except:
            message = {"text" : text}
            messages = [message]
        return messages

    def middleware_for_question(self, messages):
        item = messages[0]
        if 'reset' in messages[0]['shallow_tags']:
            self.reset_state()
        return messages


    def _question(self, messages):
        item = messages[-1]
        role = item['role']
        if role == 'request':
            # messages.append({"role": "analysis",
            #                  "prefix": "",
            #                  #"text": "听到master的话语，你想：“",
            #                  "text": "",
            #                  "over": False,
            #                  "token_ban": [],
            #                  "token_stop": [],
            #                  "token_count": 1024})

            messages.append({"role": "response",
                             "prefix": "",
                             "text": '',
                             #"text": '在思考后，你的回答是：“',
                             "token_stop": [65535],
                             "token_ban": [],
                             "over": False,
                             "token_count": 1024})
        elif role == 'system':
            messages.append({"role": "analysis",
                             "prefix": "",
                             #"text": "读完system内描述的设定，你的思考如下:\n",
                             "text": "",
                             "over": False,
                             "token_ban": [],
                             "token_stop": [65535],
                             "token_count": 1024})
        elif role == 'analysis':
            messages.append({"role": "response",
                             "prefix": "",
                             "text": "",
                             #"text": "在思考后，你的回答是：“",
                             "over": False,
                             "token_ban": [],
                             "token_stop": [65535],
                             "token_count": 2048})
        elif role == 'instruction':
            messages.append({"role": "think",
                             "prefix": "## Think:\n",
                             #"text": "",
                             "text": "你想：“",
                             "over": False,
                             "token_ban": [],
                             "token_stop": [65535],
                             "token_count": 1024})
            messages.append({"role": "answer",
                             "prefix": "## Answer:\n",
                             "text": "",
                             #"text": "你回答：“",
                             "over": False,
                             "token_ban": [],
                             "token_stop": [65535],
                             "token_count": 1024})


        else:
            messages[-1]['over'] =  False
            messages[-1]['token_stop'] = [65535]
            messages[-1]['token_count'] = 1024
            messages[-1]['resonse'] = []
        return messages

    def add_role(self, message:dict):
        master = self.master
        robot = self.robot
        if message['role'] in ['request', 'text']:
            message['prefix'] = master
        elif message['role'] in ['system']:
            message['prefix'] = "system: "
        elif message['role'] in ['analysis']:
            message['prefix'] = self.think
        elif message['role'] in ['instruction']:
            message['prefix'] = "## Question:\n"
        elif message['role'] in ['think']:
            message['prefix'] = "## Think:\n"
        elif message['role'] in ['answer']:
            message['prefix'] = "## Answer:\n"
        else:
            message['prefix'] = ""
        return message

    def load_model(self):
        host = self._config[self.model_name][0]
        port = self._config[self.model_name][1]
        method = self._interface['load-model']
        url = f'http://{host}:{port}{method}'
        m = requests.post(url, json={})
        resp = m.json()
        return resp

    def reset_state(self):
        self.load_model()
        host = self._config[self.model_name][0]
        port = self._config[self.model_name][1]
        method = self._interface['reset-state']
        url = f'http://{host}:{port}{method}'
        m = requests.post(url, json={})
        resp = m.json()
        return resp

    def question(self, messages):
        host = self._config[self.model_name][0]
        port = self._config[self.model_name][1]
        method = self._interface['generate']
        url = f'http://{host}:{port}{method}'
        if messages[0]['role'] != 'system':
            messages = self.init_prompt() + messages
        m = requests.post(url, json={"messages": messages})
        resp = m.json()
        return resp

    def teach_sft(self):
        host = self._config[self.model_name][0]
        port = self._config[self.model_name][1]
        method = self._interface['train-sft']
        url = f'http://{host}:{port}{method}'
        resp = requests.post(url,json={'title': 'train-sft'})
        resp = resp.json()
        return resp

    def teach_by_org(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        messages = self.text2message(text)
        #print("====",messages[0])
        messages = [x for x in messages if x.get('text',False) and x.get('role',False)]
        messages = random.sample(messages, 1000)
        for message in messages:
            loss = self.teach([message])
            yield loss

    def teach(self, messages):
        host = self._config[self.model_name][0]
        port = self._config[self.model_name][1]
        method = self._interface['train']
        url = f'http://{host}:{port}{method}'
        # messages = self.text2message(text)
        resp = requests.post(url,json={'title': 'from org-model',
                                       'prefix': '',
                                       'postfix': '',
                                       'messages': messages})
        resp = resp.json()
        #except:
        #    resp = {"message" : "error"}
        return resp

    def message2text(self,messages):
        res = ""
        for message in messages:
            title = message['text']
            body = message['response']
            res = res + title +"\n" + body +"\n\n"
        return res

    def ask_cluade(self,text):
        url = "https://xqtd520qidong.com/v1/chat/completions"
        headers = {
            "Authorization": "sk-MxpiqiX7zsNCW5lC8d1aEb6eEe5548AeBbE2EbBc82E24287",
            "content-type": "application/json"
        }
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompts,
                }
            ],
            "model": "claude-instant-1-100k",
            "max_tokens_to_sample": 300,
        }

        response = requests.post(url, headers=headers, json=data)
        res = response.json()
        res = res['choices'][0]['message']['content']
        res = res.split("\n\n")
        res = "\n".join(res)
        return res



if __name__ == "__main__":
    with open('/home/neromous/Documents/Workspace/projects/sft/sft.org', 'r', encoding='utf-8') as f:
        text = f.read()
    result = "你好啊  小猪佩奇"
    #result = text2item(result)
    #print(result)
    m = Inference('3b')
    comm = m.question('* ddddddfasdfasdfasdfsadfasfsadf')
    print(comm)
    test =m.teach(result)
    print(test)

# def add_question(messages):
#     item = messages[-1]
#     if item['role'] == 'request':
#         messages.append({"role": "analysis",
#                          "prefix": "",
#                          "text": "我想：",
#                          "over": False,
#                          "token_ban": [],
#                          "token_stop": [65535,0,65530],
#                          "token_count": 512})

#         messages.append({"role": "response",
#                          "prefix": "",
#                          "text": '我一边回答：“',
#                          "token_stop": [65535,65530],
#                          "token_ban": [],
#                          "over": False,
#                          "token_count": 512})
#     elif item['role'] == 'system':
#         messages.append({"role": "analysis",
#                          "prefix": "",
#                          "text": "读完system内描述的设定，我认为",
#                          "over": False,
#                          "token_ban": [],
#                          "token_stop": [65535, 65530],
#                          "token_count": 512})
#     elif item['role'] == 'response' or item['role'] == 'analysis' :
#         item['over'] =  False
#         item['token_count'] = 512
#     return messages



# def text2item(text: str):
#     nodes = loads(text)
#     result = []

#     try:
#         text = self._org_head + text
#         messages = text2item(text)
#         assert len(messages) >0
#     except:
#         message = {"text" : text}
#         messages = [message]

#     for node in nodes[1:]:
#         item = node2item(node).__dict__
#         result.append(item)
#     return result
