from rwkv_model.model import RWKV
import torch
import deepspeed


model = RWKV(load_model="/home/neromous/Documents/blackfog/resources/train-results/3b/rwkv-4.pth",
             n_embd= 2560,
             n_layer=32,
             vocab_size=65536,
             lr_init=1.0e-5,
             lr_final=1.0e-6,
             warmup_steps=4)

optimizer ,lr_scheduler = model.get_optimizers()

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     optimizer= optimizer,
                                                     lr_scheduler = lr_scheduler,
                                                     config="ds_config.config",
                                                     )

data = [x for x in range(0,10240)]
mask = [1 for x in range(0,10240)]
data = torch.tensor([data],dtype=torch.long).to('cuda')
batch = {"input_ids" : data,
         "attention_mask":None}
m = model_engine.compute_loss(model_engine,batch,None,True)
#n = model_engine.compute_loss(batch,None,True,model_engine)

model_engine.backward(m)
#model_engine.backward(n,retain_graph=True)
model_engine.step()

print(m)
