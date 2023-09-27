import types
import os
import torch
import gc
import deepspeed
import requests
from bottle import route, run, template, request
import torch
import gc
ctx_parts = 16
ctx_len = 8169
os.environ['RWKV_PARTS'] = str(ctx_parts)
os.environ['RWKV_STATE'] = "1" #开启后编译WKV_STATE的cuda kernal
os.environ["RWKV_T_MAX"] = str((ctx_len+ctx_parts-1) // ctx_parts)
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_FLOAT_MODE"] = "bf16"
from rwkv_model.model_lora import RWKV, LORA_CONFIG, LoraLinear
from models.org_text import DataNode,file_to_node,text_to_node, tokenizer,inference,inference_with_state

local_path = "/home/neromous/Documents/blackfog"


# os.environ["RWKV_T_MAX"] = str(2048)
# else:
origin_model = RWKV(f"{local_path}/resources/train-results/oneline/save-200.pth",
             vacab_size = -1,
             n_embd  = -1,
             n_layer = -1,
             head_qk = 0,
             ctx_len = 2048,
             pre_ffn = 0,
             lr_init = 1e-4,
             lr_final = 1e-5,
             warmup_steps = 9,
             beta1 =  0.9,
             beta2 =  0.999,
             adam_eps = 1e-8,
             accelerator =  "gpu",
             devices =  1,
             precision = "bf16",
             grad_cp = 0,
             accumulate_grad_batches = 1,
             strategy = "deepspeed_stage_2_offload",
             lora = False,
             # lora_r = 16,
             # lora_alpha = 32,
             # lora_dropout = 0.01,
             # lora_parts = "ffn",
             ctx_parts = None,
             weight_decay=0,
             my_pos_emb = 0,
             tiny_att_dim=0,
             tiny_att_layer=-999)

optimizer, lr_scheduler = origin_model.configure_optimizers()

model, optimizer, _, _ = deepspeed.initialize(model=origin_model,
                                              optimizer=optimizer,
                                              lr_scheduler=lr_scheduler,
                                              config="bf16_ds_config.config")



@route('/inference/by-org', method='POST')
def inference_by_org():
    global model
    item = request.json
    temperautre = item.get('temperature', 0.1)
    top_p = item.get('top_p', 0.1)
    text = item['org']
    todo = "#+TODO: USER ROBOT SYSTEM TEXT BOOK THINK CLAUDE TITLE | CANCELED\n"
    if not "".startswith("#+TODO"):
        text = todo +text
    coll = text_to_node(text)
    item_id = max(coll.keys())
    item = coll[item_id]
    print(item)
    output = inference(model,
                       item.tokens,
                       token_count=2560,
                       token_stop=[],
                       temperature = temperautre,
                       top_p = temperautre)
    return {"response": output}




run(host='0.0.0.0', port=3000)
