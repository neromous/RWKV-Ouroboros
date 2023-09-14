from rwkv_model.model import RWKV
import torch
import deepspeed


model = RWKV(load_model="/home/neromous/Documents/blackfog/resources/train-results/0.4b/rwkv-0.pth",
             n_embd= 1024,
             n_layer=24,
             vocab_size=65536)


model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.configure_optimizers(),
                                                     config="ds_config.config",
                                                     )

data = [x for x in range(0,2048)]
mask = [1 for x in range(0,2048)]
data = torch.tensor([data],dtype=torch.long).to('cuda')
batch = {"input_ids" : data,
         "attention_mask":None}
m = model_engine(data,None,True)

print(m)
