import os, sys
from utils import load_config
config = load_config('./resources/config/lora.json')
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
os.environ['RWKV_CTXLEN'] = str(config.model.ctx_len)

from util.prefix_tokenizer import prefix_tokenizer
import types
import gc
import copy
import torch
import deepspeed

import json
from tqdm import tqdm
import random
import copy
import requests
import json

from lion_pytorch import Lion
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from model.v6.rwkv_state.model import RWKV
from model.v6.rwkv_state.block import BlockStateList
from model.v6.rwkv_state.functions import train_forward, speak, ppl
import os


def split_tokens(tokens):
    l = len(tokens)
    n = int(l / config.model.chunk_len)
    m = l % config.model.chunk_len
    part_len = n * config.model.chunk_len
    head = tokens[:part_len]
    tail = tokens[part_len:]
    return head , tail


metrics = {
    'loss': 0.0,
    'reseted': "unkonwn",
}

def JsonlDefaultParser(x):
    return x['text']


DataParser = {
    'default': JsonlDefaultParser,
    'text': lambda x: x['text'],
    'instruction': lambda x: x['system'] + "\n" + x['input'] + "\n" + x["output"],
    'conversations': lambda x: "\n".join(x['conversations']),
    'system-user-assistant': lambda x: x['system'] + "\n" + x['user'] + "\n" + x["assistant"]}

def qa_mask_fn(seq):
    robot_speak = False
    mask = []
    for x in seq:
        if x == 65534:
            robot_speak = True
        elif x == 65535:
            robot_speak =  False
        else:
            pass
        if robot_speak:
            mask.append(int(x != 0))
        else:
            mask.append(0)
    return mask

def build_dynic_len_seq(seq):
    res = []
    while len(seq) > 0:
        step = random.randrange(512, 4096)
        res.append(seq[:step])
        seq = seq[step:]
    return res

class Robot:
    def __init__(self):
        "docstring"
        self.args = config
        self.train_tokenizer = prefix_tokenizer(file_name=self.args.inference.tokenizer)
        self.infer_tokenizer = prefix_tokenizer(file_name=self.args.inference.tokenizer)
        self.ds_config = {
            #"zero_force_ds_cpu_optimizer": False,
            #"zero_allow_untested_optimizer": True,
            "bfloat16": {
                "enabled": "auto"
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e6,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e6,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": self.args.optimizer.gradient_accumulation_steps,
            "gradient_clipping": self.args.trainer.grad_cp,
            "train_micro_batch_size_per_gpu": 1}
        self.ego = None
        self.current_states = None
        self.model = RWKV(self.args)
        self.model_engine = self.model_online()

        #
        self.database_url = "http://localhost:3000"

        # 
        self.scratch_tokens = []
        self.poll = []

        # speak config
        self.temperature = 0.2
        self.top_p = 0.2
        self.alpha_frequency = 0
        self.alpha_presence = 0
        self.decay = 1
        self.robot_prefix = self.infer_tokenizer.encode("Assistant:")
        self.usr_prefix = self.infer_tokenizer.encode("User:")
        self.usr_postfix = self.infer_tokenizer.encode("\n<|req-s|>主人\n")
        self.postfix = [261]

    def save_state_to_disk(self, n="default"):
        fpath = f"{self.args.proj_dir}/states/{n}.pth"
        ego_path = f"{self.args.proj_dir}/states/ego-{n}.pth"
        if self.ego is not None: 
            self.ego.save(self.ego, ego_path)
            print("==save ego==")   
        if self.current_states is not None:
            self.current_states.save(self.current_states ,fpath)
            print("==save states==")
        gc.collect()
        torch.cuda.empty_cache()

    def save_weight_to_disk(self, n="default"):
        fpath = f"{self.args.proj_dir}/{n}.pth"
        self.model.load_state_dict(self.model_engine.module.state_dict())
        torch.save(self.model.state_dict(), fpath)
        print("======save weight=======")
        gc.collect()
        torch.cuda.empty_cache()

    def save_lora_to_disk(self):
        args = self.args
        fpath = args.lora.path 
        res = {}
        for k, v in self.model_engine.module.state_dict().items():
            if "lora_" in k:
                res[k] = v
            elif (".ln" in k) and ("ln" in args.lora.parts):
                res[k] = v
            elif "time_" in k:
                res[k] = v
            elif ("emb.weight" == k) and ("emb" in args.lora.parts):
                res[k] = v
            # elif ("head.weight" == k) and ("head" in args.lora.parts):
            #     res[k] = v
            else:
                pass
        torch.save(res, fpath)
        gc.collect()
        torch.cuda.empty_cache()

    def load_state_from_disk(self, n="default"):
        try:
            fpath = f"{self.args.proj_dir}/states/{n}.pth" 
            data = torch.load(fpath)
            self.current_states =  BlockStateList.load(fpath) 
        except:
            print("未能成功初始化state")
        try:
            fpath = f"{self.args.proj_dir}/states/ego-{n}.pth" 
            data = torch.load(fpath)
            self.ego =  BlockStateList.load(fpath)  
        except:
            print("未能成功初始化ego state， 以current state替代")
            if self.current_states is not None:
                self.ego = copy.deepcopy(self.current_states)
            else:
                print("当前ego state为空")
        gc.collect()
        torch.cuda.empty_cache()

    def reset_currernt_state(self, task_type="ego", s1_part=0.5, s2_part=0.5):
        if task_type ==  "ego":
            self.current_states = copy.deepcopy(self.ego)
        elif task_type == "merge":
            self.current_states = self.current_states(
                self.ego,
                self.current_states,
                s1_part,
                s2_part
            )
        else:
            self.current_states = None
        return self


    def model_online(self):
        if self.args.trainer.optimzer_style =="sgd":
            self.optimizer = torch.optim.SGD(
                self.model.get_optim_groups(),
                lr=self.args.trainer.lr_init)
            self.lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                self.optimizer,
                warmup_min_lr=0.2 * self.args.trainer.lr_init,
                warmup_max_lr=self.args.trainer.lr_init,
                warmup_num_steps=self.args.trainer.warmup_steps,
                warmup_type="linear",
            )

        elif  self.args.trainer.optimzer_style =="lion":
            self.optimizer = Lion(
                self.model.get_optim_groups(),
                lr=1e-4,
                weight_decay=1e-2)

            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                optimizer=self.optimizer,
                config=self.ds_config
            )

            self.load_state_from_disk()
            return self.model_engine

        else:
            self.optimizer =  DeepSpeedCPUAdam(
                self.model.get_optim_groups(),
                lr=self.args.trainer.lr_init,
                betas=(self.args.trainer.beta1, self.args.trainer.beta2),
                eps=self.args.trainer.adam_eps,
                adamw_mode=self.args.trainer.adamw_mode,
                weight_decay=self.args.trainer.weight_decay,
                amsgrad=False,
                bias_correction=True,
            )

            self.lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                self.optimizer,
                warmup_min_lr=0.2 * self.args.trainer.lr_init,
                warmup_max_lr=self.args.trainer.lr_init,
                warmup_num_steps=self.args.trainer.warmup_steps,
                warmup_type="linear",
            )

        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self.ds_config
        )
        self.load_state_from_disk()
        return self.model_engine

    def listen(self, tokens, state, viod_token=11):
        chunk_len = 24
        if state is not None:
            state = state.cuda()    
        if isinstance(tokens, str):
            tokens = self.infer_tokenizer.encode(tokens)
        with torch.no_grad():
            self.model_engine.eval()
            if len(tokens) > 2:
                start_with_token = tokens[-1]
                tokens = tokens[:-1]
                head, tail = split_tokens(tokens)
                if len(head) > 0:
                    logits, state = self.model_engine(head, state)
                for token in tail:
                    logits, state = self.model_engine([token], state)
            elif len(tokens) == 1:
                state = state
                start_with_token = tokens[-1]
            else:
                temp_state = state
                start_with_token = viod_token
            self.poll.append(tokens[:-1])
            return state, start_with_token

    def speak(self,
              tokens,
              state,
              n_max_tokens=240,
              token_stop=[65535],
              void_token=11,
              use_pos_cfg=False,
              use_neg_cfg=False,
              pos_alpha=0.4,
              neg_alpha=0.2,
              token_ban=[0],
              m_postfix_token=[11]):
        with torch.no_grad():
            self.model_engine.eval()
            if isinstance(tokens, list):
                state, start_with_token = self.listen(tokens, state)
            elif isinstance(tokens, str):
                state, start_with_token = self.listen(tokens, state)
            elif isinstance(tokens, int):
                start_with_token = tokens
            else:
                raise Exception()
            if state is not None:
                state = state.cuda()   
            speak_tokens, out_state = speak(self.model_engine,
                  start_with_token,
                  state,
                  self.temperature,
                  self.top_p,
                  self.alpha_presence,
                  self.alpha_frequency,
                  self.infer_tokenizer,
                  token_stop=token_stop,
                  max_tokens=n_max_tokens,
                  debug_mode=True,
                  decay=self.decay)
            speak_str = self.infer_tokenizer.decode(speak_tokens)
            self.poll.append(speak_tokens)
            return speak_tokens, speak_str, out_state.cpu()

    def reflex(self, inputs, state, reset_by_step=True):
        assert len(inputs) != 0
        all_tokens = [self.train_tokenizer.encode(x) if isinstance(x, str) else x for x in inputs]
        result = []
        with torch.no_grad():
            self.model_engine.eval()
            for tokens in all_tokens:
                mean_loss = 0
                total = 0
                i = 0
                while len(tokens) > 0:
                    i += 1
                    output = tokens[:self.args.trainer.ctx_len]
                    tokens = tokens[self.args.trainer.ctx_len - self.args.trainer.window:]
                    logits, state = self.model_engine(output, state)
                    loss = ppl(tokens, logits)
                    total += loss
                    mean_loss = total / i

                result.append([mean_loss, tokens])
                if reset_by_step:
                    state = None
        return result

    def learn(self,
              inputs,
              states,
              reset_by_step=True,
              window=0,
              min_loss=0.5,
              min_loss_fix=1.0,
              max_loss=2.0,
              max_loss_fix=1.0,
              debug=True,
              text=False,
              qa_mask=False,
              ):
        """
        [[t_dataset1],[t_dataset2],...]
        """
        self.model_engine.train()
        assert len(inputs) != 0
        assert isinstance(inputs[0], list)
        assert isinstance(inputs[0][0], int)
        if states is not None:
            states = states.cuda()   
        ctx_len = self.args.trainer.ctx_len
        all_tokens = [x for x in inputs]
        losses = []
        total = 0
        i = 0
        mean_loss = 0
        for tokens in all_tokens:
            origin_tokens = [x for x in tokens]
            if reset_by_step:
                states = self.ego
            else:
                states = None
            if qa_mask:
                masks = qa_mask_fn(tokens)
            while len(tokens) > 0:
                i += 1
                output = tokens[:ctx_len]
                tokens = tokens[ctx_len - window:]
                if qa_mask:
                    mask_output = masks[:ctx_len]
                    masks = masks[ctx_len - window:]
                else:
                    mask_output = None

                if len(output) < 16:
                    continue
                batch = {"input_ids": output, "masks": mask_output}

                m, states = train_forward(self.model_engine, batch, self.ego)
                loss = m.item()
                if loss < min_loss:
                    m = m * min_loss_fix
                elif loss > max_loss:
                    m = m * max_loss_fix
                self.model_engine.backward(m)
                self.model_engine.step()
                total += loss
                mean_loss = total / i
                losses.append(mean_loss)
            if debug:
                print("\n")
                print(f"> 总数:{i}/{len(all_tokens)}  内容长度{len(origin_tokens)}  学习误差->{mean_loss:.3f}")
                print("\n")
            # if i % 64 == 0:
            #     self.model_engine.eval()
            #     self.speak("\nUser:水泥好吃吗？\n\nAssistant:", states, token_stop=[261])
            #     self.model_engine.train()   
                # ================
                # update the tqdm bar
                # ================
                #metrics['loss'] = mean_loss
                #pbar.set_postfix(metrics)
                #pbar.update(1)
        self.current_states = states.cpu()
        self.ego = states.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        return losses

    def read(self,
             content,
             states,
             source_type='txt',
             jsonl='text',
             n=10,
             **kwargs):

        if source_type == 'txt':
            content = "\n".join(content)
            tokens = self.infer_tokenizer.encode(content)
            tokens = build_dynic_len_seq(tokens)
        elif source_type == 'jsonl':
            coll = [json.loads(x) for x in content]
            fn = DataParser.get(jsonl, DataParser['default'])
            coll = [fn(x) for x in coll]
            tokens = [self.infer_tokenizer.encode(x) for x in coll]
            tokens = [x for x in tokens if len(x) >= 24]
            random.shuffle(tokens)
            tokens = tokens[ : n*20]
        losses = self.learn(tokens, states=states, **kwargs)
        return losses

    def db_transact(self, domain, title, content):
        query = {}
        url = self.args.db.host + self.args.db.api_for_transact
        resp = requests.post(url, json=query)
        text = resp.json()['content']
        return text

    def db_find(self, domain="g-atom/phrase", title=False):
        query = {}
        url = self.database_url + "/db/find-by"  
        query = {}
        query['entity'] =  domain
        if title:
            query["search"] = title
        res = requests.post("http://localhost:3000/db/find-by",json=query)
        return res

    def db_remove(self, domain, title):
        query = {}
        url = self.args.db.host + self.args.db.api_for_query
        resp = requests.post(url, json=query)
        text = resp.json()['content']
        return text

def main():
    app = Robot()
    datapath = "/home/neromous/data/lima.jsonl"
    #datapath = "/home/neromous/Desktop/临高启明.txt"
    i = 0
    states = None
    for x in range(10):
        # app.speak("\nUser:水泥好吃吗？\n\nAssistant:", None, token_stop=[261])
        # states, last_token = app.listen("<|im-s|>user:接下来我会向你提供一个小说的片断，这个小说名为《临高启明》，是一部穿越小说。请你仔细阅读我的输入，当你阅读完毕，请你说‘好的’\n**小说正文**\n",states)
        loss = app.read(datapath, None, source_type='jsonl', jsonl='conversations')


if __name__ == '__main__':
    main()
