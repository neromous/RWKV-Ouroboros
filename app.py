import os
from utils import log, load_config
import torch
import gc
import deepspeed
import json
from bottle import route, run, template, request
import random

config = load_config()
# ===============pico 配置项=================
os.environ["RWKV_JIT_ON"] = config["environ"]["RWKV_JIT_ON"]
os.environ["RWKV_FLOAT_MODE"] = config["environ"]["RWKV_FLOAT_MODE"]
os.environ["RWKV_MY_TESTING"] = config["environ"]["RWKV_MY_TESTING"]

window = config["trainer"]["window"]
min_loss_fix = config["trainer"]["min_loss_fix"]
max_loss_fix = config["trainer"]["max_loss_fix"]
min_loss = config["trainer"]["min_loss"]
max_loss = config["trainer"]["max_loss"]
proj_dir = config["proj_dir"]
ctx_len = config["model"]["ctx_len"]
ctx_parts = config["trainer"]["ctx_parts"]
if config["infctx_on"]:
    if config["infctx_type"] == "wani-boat":
        ctx_parts = config["trainer"]["ctx_parts"]
        os.environ["RWKV_PARTS"] = str(ctx_parts)
        os.environ["RWKV_STATE"] = config["environ"]["RWKV_STATE"]
        os.environ["RWKV_T_MAX"] = str((ctx_len + ctx_parts - 1) // ctx_parts)
        ds_config = "./stage1_offload_ds_config.config"
        from rwkv_model.model_state import RWKV
    elif config["infctx_type"] == "pico":
        os.environ["RWKV_TORCH_COMPILE"] = config["environ"]["RWKV_TORCH_COMPILE"]
        from rwkv_model.model import RWKV

        ds_config = "./ds_config.config"
else:
    os.environ["RWKV_STATE"] = config["environ"]["RWKV_STATE"]
    os.environ["RWKV_PARTS"] = str(ctx_parts)
    if os.environ["RWKV_PARTS"] != "0":
        os.environ["RWKV_T_MAX"] = str((ctx_len + ctx_parts - 1) // ctx_parts)
    else:
        os.environ["RWKV_T_MAX"] = str(ctx_len)

    from rwkv_model.model_lora import RWKV

    if config["environ"]["RWKV_FLOAT_MODE"] == "fp32":
        ds_config = "./fp32_ds_config.config"
    elif config["environ"]["RWKV_FLOAT_MODE"] == "fp16":
        ds_config = "./fp16_ds_config.config"
    elif config["environ"]["RWKV_FLOAT_MODE"] == "bf16":
        ds_config = "./bf16_ds_config.config"

from rwkv_model.model_infer import RWKV_RNN
from models.scene import Scene
from models.page import Page
from utils import save_data
from models.inference_helper import InferenceWithState
from models.org_text import DataNode, file_to_node, text_to_node
import copy
from tqdm import tqdm

model = RWKV(
    load_model=config["model_path"],
    n_layer=config["model"]["n_layer"],
    ctx_len=config["model"]["ctx_len"],
    n_embd=config["model"]["n_embd"],
    vocab_size=config["model"]["vocab_size"],
    grad_cp=config["trainer"]["grad_cp"],
    lora=config["lora"],
    lr_init=1.0e-5,
    lr_final=1.0e-5,
    dtype=config["environ"]["RWKV_FLOAT_MODE"],
    warmup_steps=config["trainer"]["warmup_steps"],
)

optimizer, lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, config=ds_config
)

inferencer = InferenceWithState()
rwkv_rnn = None
state = None
init_state = None


@route("/inference/load-model", method="POST")
def load_model():
    global inferencer, model_engine, rwkv_rnn, state, init_state
    item = request.json
    gc.collect()
    torch.cuda.empty_cache()
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())
    state = None
    init_state = None
    return {"response": "model save"}


@route("/inference/remove-model", method="POST")
def remove_model():
    global inferencer
    item = request.json
    rwkv_rnn = None
    state = None
    init_state = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}


@route("/state/init", method="POST")
def init():
    global inferencer, rwkv_rnn, state, init_state
    if rwkv_rnn == None:
        load_model()
    item = request.json
    messages = item.get("messages", [])
    resp = []
    state = None
    init_state = None
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg, state = inferencer.generate(rwkv_rnn, msg, state=state)
        msg.save()
        resp.append(msg.json())
    init_state = copy.deepcopy(state)
    return {"messages": resp}


@route("/state/reset", method="POST")
def reset_state():
    global inferencer, state, init_state
    print(inferencer.state)
    print(inferencer.init_state)
    state = copy.deepcopy(init_state)
    return {"messages": "reset"}


@route("/inference/generate", method="POST")
def inference_generate():
    global inferencer, rwkv_rnn, state
    if rwkv_rnn == None:
        load_model()
    item = request.json
    messages = item.get("messages", [])
    resp = []
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg, state = inferencer.generate(rwkv_rnn, msg, state=state)
        msg.save()
        resp.append(msg.json())
    return {"messages": resp}


@route("/inference/generate_by_local_state", method="POST")
def inference_generate_by_local_state():
    global inferencer, rwkv_rnn
    if rwkv_rnn == None:
        load_model()
    item = request.json
    conversations = item.get("conversations", [])
    save_dir = item.get("sv_dir", None)
    load_dir = item.get("ld_dir", None)
    resp = []
    msg = item["msg"]
    msg, _ = inferencer.generate_by_state(
        rwkv_rnn, load_dir, save_dir, state, conversations, msg
    )
    resp.append(msg.json())
    return {"messages": resp}


@route("/inference/generate-no-state", method="POST")
def inference_generate_no_state():
    global inferencer, model
    item = request.json
    messages = item.get("messages", [])
    resp = []
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate_no_state(model, msg)
        msg.save()
        resp.append(msg.json())
    return {"messages": resp}


@route("/train/save-weight", method="POST")
def save_weight():
    global model_engine, model
    item = request.json
    model_name = item.get("model_name", "default")
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(model_engine.module.state_dict())
    # ===============save=================
    fpath = f"{proj_dir}/{model_name}.pth"
    torch.save(model.state_dict(), fpath)
    print("===saveved====")
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": "model save"}


fault_time = 0


@route("/train/tx-data", method="POST")
def train_tx_data():
    global model_engine, ctx_len, window, fault_time
    item = request.json
    gc.collect()
    torch.cuda.empty_cache()
    try:
        # parse
        if type(item) == dict:
            train_data = Scene.new(item)
        else:
            return {"message": "failed for unvalid data, request should be a dict"}
        total = 0
        mean_loss = 0
        i = 0
        data_iter = train_data.yield_tokens(ctx_len=ctx_len, window=window)
        for token in data_iter:
            i += 1
            batch = {"input_ids": token, "attention_mask": None}
            m = model_engine.training_step(batch, model_engine=model_engine)
            loss = m.item()
            if loss < min_loss:
                m = m * min_loss_fix
            elif loss > max_loss:
                m = m * max_loss_fix
            total += loss
            mean_loss = total / i
            model_engine.backward(m)
            model_engine.step()
    except:
        fault_time += 1
        if fault_time == 20:
            raise Exception("The maximum number of failures was reached.")
        print(f"[Warning] Training Process Faulted, Total: {fault_time}")
    # save_data(item)
    return {"loss": mean_loss}


@route("/train/token", method="POST")
def train_token():
    global model_engine, ctx_len, window
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    input_ids = item["input_ids"]
    attention_mask = item.get("attention_mask", None)
    if attention_mask == None:
        attention_mask = [1 for x in input_ids]
    assert len(input_ids) == len(attention_mask)
    assert len(input_ids) > 0
    losses = []
    while len(input_ids) > 0:
        output = input_ids[:ctx_len]
        masks = masks[:ctx_len]
        input_ids = input_ids[ctx_len - window :]
        attention_mask = attention_mask[ctx_len - window :]

        batch = {"input_ids": output, "attention_mask": masks}
        m = model_engine.training_step(batch, model_engine=model_engine)
        loss = m.item()
        losses.append(loss)
        print("->", loss)
        model_engine.backward(m)
        model_engine.step()
    # save_data(item)
    return {"loss": losses}


@route("/train/sft", method="POST")
def train_sft():
    global model_engine, ctx_len, window
    rnn_model = None
    gc.collect()
    torch.cuda.empty_cache()
    item = request.json
    coll = file_to_node("./data/sft.org")
    losses = []
    total_loss = 0
    mean_loss = 0
    datasets = []
    for k, v in coll.items():
        datasets.append(v)
    start = datasets[0]
    end = datasets[1:]
    random.shuffle(end)
    datasets = [start] + end
    i = 0
    for v in tqdm(datasets):
        for token in v.yield_train_data(req_len=ctx_len, window=window):
            i += 1
            batch = {"input_ids": token, "attention_mask": None}
            m = model_engine.training_step(batch)
            loss = m.item()
            if loss < min_loss:
                m = m * min_loss_fix
            elif loss > max_loss:
                m = m * max_loss_fix
            losses.append(loss)
            model_engine.backward(m)
            model_engine.step()
            # 计算loss
            total_loss += loss
            mean_loss = total_loss / i
            losses.append(mean_loss)
            print(f"-> item_loss {loss} batch_loss {mean_loss}, real_loss {m.item()}")
    gc.collect()
    torch.cuda.empty_cache()
    return {"loss": mean_loss}


if config["debug"]:
    messages = [{"text": "你好啊", "role": "text", "over": False, "token_count": 128}]
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate_no_state(model, msg)
        print("=====msg", msg)
    pass

    print("===train test start==")
    train_data = [x for x in range(0, 6144)]
    batch = {"input_ids": train_data, "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    gc.collect()
    torch.cuda.empty_cache()

    print("===train test over==", loss)
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())

    messages = [{"text": "你好啊", "role": "text", "over": False, "token_count": 128}]
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate(rwkv_rnn, msg)
        print("==msg==", msg)
    print("===train test start==")
    rwkv_rnn = None
    gc.collect()
    torch.cuda.empty_cache()

    train_data = [x for x in range(0, 6144)]
    batch = {"input_ids": train_data, "attention_mask": None}
    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    gc.collect()
    torch.cuda.empty_cache()
    print("===train test over==", loss)
    rwkv_rnn = RWKV_RNN(model_engine.module.state_dict())
    for message in messages:
        msg = inferencer.scene.add_message(message)
        msg = inferencer.generate(rwkv_rnn, msg)
        print("==msg==", msg)

    rwkv_rnn = None
    gc.collect()
    torch.cuda.empty_cache()

    m = model_engine.training_step(batch, model_engine=model_engine)
    loss = m.item()
    model_engine.backward(m)
    model_engine.step()
    print("===train test over==", loss)


if not config["debug"]:
    run(host="0.0.0.0", port=3000)
