import os, warnings, math, datetime, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import deepspeed
from src.dataset_finetune import MyDataset
import types
from tqdm import tqdm
import random
from torch.nn import functional as F
os.environ["RWKV_JIT_ON"] = '1'
from src.tokenizer import Tokenizer
from src.model_run import RWKV_RNN, sample_logits
import copy
import time
import cmd
import gc
import json
import random
from route import route,load_train_config,load_infor_config
from utils import get_model, next_model, parse_str
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import bottle
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# allow tf32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
#from bottle import route, run, template, request

envrionment = {"model_engine": None,
               "optimizer": None,
               "model_inf": None}

def train():
    model_engine = envrionment['model_eigine']
    train_state = {'model_engine': model_engine}
    data = []
    for x in tqdm(range(0, 8)):
        data.append(train_data.__getitem__(0))
    train_state["train_data"] = data
    train_state = route("train", train_state)
    print('\nloss->', train_state['train_loss'])

def my_func(tmp):
    print(tmp, end="", flush=True)

if __name__ == "__main__":
    tokenizer = Tokenizer()
    train_args= load_train_config()
    os.environ['RWKV_FLOAT_MODE'] = "fp16"
    os.environ['RWKV_T_MAX'] = str(train_args.ctx_len)
    os.environ['RWKV_MY_TESTING'] = ""
    infor_args = load_infor_config()
    from src.model import RWKV
    #########################################################################
    ##################################################################################
    train_data = MyDataset(train_args)
    model = RWKV(train_args)
    load_dict = torch.load(train_args.load_model, map_location="cpu")
    model.load_state_dict(load_dict)
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                         model_parameters=model.configure_optimizers(),
                                                         config="ds_config.config")
    envrionment['model_eigine'] = model_engine
    envrionment['optimizer'] = optimizer

    train_state = {'model_engine': model_engine}
    while True:
        train()
        command = input("\ncmd-> ")
        if command.startswith("+train-forward"):
            text = command[len("+train-forward"):]
            data = []
            for x in range(0, 8):
                data.append(train_data.__getitem__(0))
            train_state["train_data"] = data
            train_state = route("train", train_state)
            print('\nloss->', train_state['train_loss'])
        if command.startswith("+train-again"):
            if not data:
                data = []
                for x in range(0, 8):
                    data.append(train_data.__getitem__(0))
            random.shuffle(data)
            train_state["train_data"] = data
            train_state = route("train", train_state)
            print('\nloss->', train_state['train_loss'])
        elif command.startswith("+inference"):
            torch.cuda.empty_cache()
            gc.collect()
            #  load to inference
            ds = model_engine.module.state_dict()
            cache_dict = {}
            for k, v in ds.items():
                cache_dict[k] = v.cpu()
            model_inf = RWKV_RNN(infer_args, cache_dict)
            x = input("\nenter you prompt: ")
            while not x.startswith("+over"):
                if x.startswith("+reset"):
                    x = x.replace("+reset", "", 1)
                    print("reset-> ", x)
                    state = None
                message = config.read_config()['inference']['prompt_config']
                role = {"content": "user" + x.strip(),
                        "over": True,
                        "role": "user",
                        "token_count": 0
                        }
                message.update(role)
                res, state = model_inf.rwkv_generate(model_inf, message, callback=my_func)
                role = {"content": "assistant: ",
                        "over": False,
                        "role": "robot",
                        "token_count": 256
                        }
                message.update(role)
                res, state = model_inf.rwkv_generate(model_inf, message, callback=my_func, state=state)
                x = input("\nenter you prompt: ")
            torch.cuda.empty_cache()
            gc.collect()
            state = None
            new_state = None
        else:
            pass



















        # # loss
        # loss = model_engine(data)
        # # 更新标志
        # if loss.item() < 0.1:
        #     flag = True
        # print('\nloss->',loss.item())
        # model_engine.backward(loss )
        # model_engine.step()

        # if n ==  16:
        #     from tqdm import tqdm
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     #  load to inference
        #     ds = model_engine.module.state_dict()
        #     cache_dict = {}
        #     for k,v in ds.items():
        #         cache_dict[k] = v.cpu()

        #     model_inf = RWKV_RNN(infer_args, cache_dict)

        #     # load default message
        #     message = config.read_config()['inference']['prompt_config']
        #     m = []
        #     role = {"content": "测试",
        #             "over": True,
        #             "role" : "user",
        #             "token_count": 0
        #             }
        #     message.update(role)
        #     print("\n->",message["content"])
        #     res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func)

        #     m.append(res)
        #     new_state = state.clone()

        #     role = {"content": "",
        #             "over": False,
        #             "role" : "robot",
        #             "token_count": 256
        #             }
        #     message.update(role)
        #     res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func,state=new_state)
        #     m.append(res)


        #     for x in m:
        #         with open("./bonsai.jsonl","a",encoding="utf-8") as f:
        #             f.write(json.dumps(x,ensure_ascii=False))


        #     i = 0
        #     model_inf = None
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     n = 0
