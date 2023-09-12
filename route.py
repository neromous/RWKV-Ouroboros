import gc
import torch
import copy
from tqdm import tqdm
import src.config as config
import types
from utils import get_model,next_model
import os

################################
def load_train_config():
    train_args = types.SimpleNamespace()
    for k, v in config.read_config()['trainer'].items():
        setattr(train_args, k, v)
    for k, v in config.read_config()['model'].items():
        setattr(train_args, k, v)
    for k, v in config.read_config()['environ'].items():
        setattr(train_args, k, v)
    if train_args.load_model == "":
        train_args.load_model = get_model(train_args.proj_dir)
    train_args.epoch_begin = next_model(train_args.proj_dir)
    if not os.path.exists(train_args.proj_dir):
        os.makedirs(train_args.proj_dir)
    return train_args

def load_infor_config():
    infer_args = types.SimpleNamespace()
    for k, v in config.read_config()['model'].items():
        setattr(infer_args, k, v)
    for k, v in config.read_config()['inference']['service_config'].items():
        setattr(infer_args, k, v)
    return infer_args



################################

def chat(kwargs: dict):
    model_engine = kwargs['model_engine']
    train_data = kwargs['train_data']

    loss = model_engine(train_data)
    model_engine.backward(loss)
    model_engine.step()
    kwargs['train_loss'] = loss
    return kwargs


def inference(kwargs: dict):
    text = kwargs['text']
    r = {}
    results = []
    for k, v in r.items():
        if text.startswith(k):
            res = [k, k(text[len(k):])]
            results.append(res)
    return results


def train(kwargs: dict):
    model_engine = kwargs['model_engine']
    train_data = kwargs['train_data']
    gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 4)
    kwargs['train_loss'] = []
    if type(train_data) == list:
        for data in tqdm(train_data):
            loss = model_engine(data)
            model_engine.backward(loss)
            if len(train_data) % gradient_accumulation_steps == 0:
                model_engine.step()
            kwargs['train_loss'].append(loss.item())
    else:
        loss = model_engine(train_data)
        model_engine.backward(loss)
        model_engine.step()
        kwargs['train_loss'].append(loss.item())
    return kwargs


def route(events: str, kwargs: dict):
    r = {"train": train,
         "inference": inference}

    return r[events](kwargs)
