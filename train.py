import os, warnings, math, datetime, sys, time
os.environ["RWKV_JIT_ON"] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader
import deepspeed
from src.dataset_finetune import MyDataset
import types
from tqdm import tqdm
import random
# ---------- 加载tokenizer ----------
from torch.nn import functional as F
from src.tokenizer import Tokenizer
from src.model_run import RWKV_RNN, sample_logits
import copy
import time
import cmd
import src.config as config
import gc
import json
np.set_printoptions(precision=4, suppress=True, linewidth=200)

def init_args(args):
    if args.load_model == "":
        args.load_model = get_model(args.proj_dir)
    args.epoch_begin = next_model(args.proj_dir)
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)
    return args

def my_func(tmp):
    print(tmp, end="", flush=True)

if __name__ == "__main__":

    tokenizer = Tokenizer()
    from argparse import ArgumentParser
    from utils import get_model,next_model
    train_args = types.SimpleNamespace()

    #args.load_model = "/mnt/develop/blackfog/rwkv-trainer/output/0.1b/rwkv-0.pth"
    for k, v in config.read_config()['trainer'].items():
        setattr(train_args, k, v)

    for k, v in config.read_config()['model'].items():
        setattr(train_args, k, v)
    args = init_args(train_args)
    os.environ['RWKV_FLOAT_MODE'] = "bf16"
    os.environ['RWKV_T_MAX'] = str(args.ctx_len)
    os.environ['RWKV_MY_TESTING'] = ""
    print(train_args)

    infer_args = types.SimpleNamespace()
    #args.load_model = "/mnt/develop/blackfog/rwkv-trainer/output/0.1b/rwkv-0.pth"
    for k, v in config.read_config()['model'].items():
        setattr(infer_args, k, v)

    for k, v in config.read_config()['inference']['service_config'].items():
        setattr(infer_args, k, v)


    print(train_args)


    from src.model import RWKV


    #########################################################################
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # allow tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ##################################################################################
    train_data = MyDataset(train_args)
    args.vocab_size = train_data.vocab_size
    model = RWKV(train_args)
    load_dict = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(load_dict)
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                         model_parameters=model.configure_optimizers(),
                                                         config="ds_config.config")

    n = 0
    flag = True
    real_loss = 2.0
    data  =  train_data.__getitem__(0)
    while True:

        if flag:
            data  =  train_data.__getitem__(0)
            flag = False
            n += 1

        # loss
        loss = model_engine(data)
        # 更新标志
        if loss.item() < 0.1:
            flag = True
        print('\nloss->',loss.item())
        model_engine.backward(loss )
        model_engine.step()

        if n ==  16:
            from tqdm import tqdm
            torch.cuda.empty_cache()
            gc.collect()
            #  load to inference
            ds = model_engine.module.state_dict()
            cache_dict = {}
            for k,v in ds.items():
                cache_dict[k] = v.cpu()

            model_inf = RWKV_RNN(infer_args, cache_dict)

            # load default message
            message = config.read_config()['inference']['prompt_config']
            m = []
            role = {"content": "测试",
                    "over": True,
                    "role" : "user",
                    "token_count": 0
                    }
            message.update(role)
            print("\n->",message["content"])
            res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func)

            m.append(res)
            new_state = state.clone()

            role = {"content": "",
                    "over": False,
                    "role" : "robot",
                    "token_count": 256
                    }
            message.update(role)
            res,state = model_inf.rwkv_generate(model_inf,message,callback=my_func,state=new_state)
            m.append(res)


            for x in m:
                with open("./bonsai.jsonl","a",encoding="utf-8") as f:
                    f.write(json.dumps(x,ensure_ascii=False))


            i = 0
            model_inf = None
            torch.cuda.empty_cache()
            gc.collect()
            n = 0

            # print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
            # init_state = None
            # tokens = tokenizer.encode(context)
            # init_out, init_state = model_inf.forward(tokens, init_state)

            # for TRIAL in tqdm(range(NUM_TRIALS)):
            #     print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
            #     all_tokens = []
            #     out_last = 0
            #     out, state = init_out.clone(), init_state.clone()
            #     for i in range(LENGTH_PER_TRIAL):
            #         token = sample_logits(out,temperature=TEMPERATURE,top_p=TOP_P)
            #         all_tokens += [token]
            #         tmp = tokenizer.decode(all_tokens[out_last:])
            #         if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            #             print(tmp, end="", flush=True)
            #             out_last = i + 1
            #             out, state = model_inf.forward([token], state)
            # print('\n')

        #     model_v2 = Inference.new(kargs,ds)
        #     message = types.SimpleNamespace()

        #     message.temperature = 0.1
        #     message.top_p = 0.1
        #     message.top_k = 0
        #     message.alpha_frequency = 0.4
        #     message.alpha_presence = 0.4
        #     message.alpha_decay = 0.996 # gradually decay the penalty
        #     message.token_ban = []
        #     message.token_stop = [0,65535]
        #     message.chunk_len = 128
        #     message.token_count = 256
        #     message.over = True
        #     message.content = "Question: 介绍陈璇的详细信息和背景\n\nAnswer:"
        #     res  = model_v2.generate(message)
        #     print(res + "\n")
        #     message.content = "Question: 说明陈璇的死因\n\nAnswer:"
        #     res  = model_v2.generate(message)
        #     print(res+"\n")
        #     message.content = "Question: 谁是杀死陈璇的凶手?\n\nAnswer:"
        #     res  = model_v2.generate(message)
        #     print(res+"\n")
        #     message.content = "Question: 凶手杀死陈璇后如何处理她的尸体?\n\nAnswer:"
        #     res  = model_v2.generate(message)
        #     print(res+"\n")

        #     result = model_v2.unload()

        #     print(result)
        #     i = 0


        # #     print("===where===",model_v2)


        # print("->", loss)





        # logits = logits
        # output = []
        # print("===loss===",loss)
        # if n == 1:
        #     for token_prob in logits:
        #         sam = sample_logits(token_prob)
        #         #print("======",type(sam))
        #         output.append(sam)
        #         #output.append(tokenizer.decode([sam]))
        #         output = [x for x in output if x != 0]
        #         text = tokenizer.decode(output)
        #         n = 0
        #     print("+------>",text)
        # n += 1
        # for token_prob in layer_logits[10]:
        #     sam = sample_logits(token_prob)
        #     #print("======",type(sam))
        #     output.append(sam)
        #     #output.append(tokenizer.decode([sam]))
        # output = [x for x in output if x != 0]
        # text = tokenizer.decode(output)
        # print("+---layer 10--->",text)
        #print("+------>",output)
        # print("=======", logits)
        # print("=======", logits.shape)
        # print("=======", logits.shape)

        # print("===logits===",logits)
        # print("===layer logits[0]===",layer_logits[0])
        # print("===layer logits[0]===",layer_logits[0].shape)
        # print("===layer logits[0]===",layer_logits[0].view(-1,layer_logits[0].size(-1)))
        # print("===layer logits[0]===",layer_logits[0].view(-1,layer_logits[0].size(-1)).shape)
