import os
import time
import json
import re


def sorted_fn(text):
    n = re.findall(r'rwkv\-(.+?)\.', text)
    n = n[0]
    n = int(n)
    return n


def get_model(proj_path):
    files = os.listdir(proj_path)
    model_weights = [x for x in files if x.endswith('.pth')]
    model_weights =  sorted(model_weights,key=sorted_fn)
    model = model_weights[-1]
    model = f'{proj_path}/{model}'
    return model

def next_model(proj_path):
    files = os.listdir(proj_path)
    model_weights = [x for x in files if x.endswith('.pth')]
    model_weights =  sorted(model_weights,key=sorted_fn)
    model = model_weights[-1]
    n = re.findall(r'rwkv\-(.+?)\.', model)
    n = n[0]
    n = int(n)
    return n + 1


def parse_str(text):
    text = text.strip()
    args_map = {}
    if text != "":
        args = re.split(text,"\s+")
        args = [x for x in args if x != ""]
        for k, v in args[0:2]:
            args_map[k] = v
    else:
        args_map[k] = v
    return args_map
