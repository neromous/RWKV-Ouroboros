from rwkv_model.model import RWKV
import torch
import deepspeed
from bottle import route, run, template, request
import json
from models.scene import Scene
from utils import save_data

model = RWKV(load_model="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth",
             #n_embd=2560,
             #n_layer=32,
             #vocab_size=65536,
             lr_init=1.0e-5,
             lr_final=1.0e-6,
             warmup_steps=20,
             substep_cuda_cache_clear=True
             )

optimizer, lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler,
                                                     config="ds_config.config")

n = Scene.new({"text" :"","messages": []})
m = n.add_message({"text":""})
m.load_file("/mnt/database/Datasets/materials/bonsai/bonsai_extend.txt")
#print(m.to_tokens())
i = 0
total = 0
from tqdm import tqdm
for d in tqdm(m.yield_tokens()):
    i += 1
    batch = {"input_ids": torch.tensor([d], dtype=torch.long).to('cuda'),
             "attention_mask": None}
    m = model_engine.training_step(batch, None,model_engine=model_engine)
    total += m.item()
    model_engine.backward(m)
    model_engine.step()

    if i % 8 == 0 and i != 0:
        mean_loss = total / i
        print(f'\nloss-> {mean_loss}')

    if i % 100 == 0 and i != 0:
        print("---->save")
        model_engine.to(torch.device('cpu'))
        torch.save(model_engine.module.state_dict(), f"save-{i}.pth")
        model_engine.to(torch.device('cuda'))
