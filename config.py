from utils.tokenizer import TRIE_TOKENIZER

tokenizer = TRIE_TOKENIZER("./resources/vocab_file/rwkv_vocab_v20230424.txt")

config = {
    "model_name": "default",
    "proj_dir": "/home/neromous/Documents/ouroboros/RWKV-Ouroboros/",
    "port": 3000,
    "debug": False,
    "model": {
        "load_model": "/home/neromous/RWKV-Ouroboros/resources/weights/default.pth",
        "rwkv_version": "v5",
        "n_embd":  2560,
        "n_layer": 32,
        "vocab_size": 65536,
        "ctx_len": 3200,
        "dtype": "fp32",
        "head_size_a": 64,
        "head_size_divisor": 8,
        "ds_config": './ds_config/fp32_config.config',

    },
    "lora_config": {
        "lora": False,
        "r": 0,
        "alpha": 0,
        "dropout": 0,
        "parts": ["att", "ln", "time"],
        "layers": None
    },
    "trainer": {
        "infctx_on": True,
        "warmup_steps": 8,
        "grad_cp": 1,
        "dropout": 0.001,
        "window": 0,
        "min_loss": 0.5,
        "max_loss": 1.5,
        "min_loss_fix": 0.05,
        "max_loss_fix": 1.05,
        "ctx_parts": "0",
        "tokenizer": "./resources/vocab_file/rwkv_vocab_v20230424train.txt",
        "head_size": 64,
        "weight_decay": 0.01,
        "dim_att": 0,
        "dim_ffn": 0,
        "my_qa_mask": 0,
        "my_pos_emb": 0,
        "tiny_att_dim": 0,
        "pre_ffn": 0,
        "head_qk": 0,
        "layerwise_lr": 1,
        "lr_init": 1.0e-5,
        "beta1": 0.9,
        "beta2": 0.99,
        "adam_eps": 1.0e-8,
        "pre_ffn": 0,
        "adamw_mode": True,
    },
    "inference": {
        "tokenizer": "./resources/vocab_file/rwkv_vocab_v20230424.txt",
        "prompt_config": {
            "temperature": 0.2,
            "top_p": 0.2,
            "top_k": 0,
            "alpha_frequency": 0.40,
            "alpha_presence": 0.40,
            "alpha_decay": 0.996,
            "token_ban": [0],
            "token_stop": [65535],
            "chunk_len": 128,
            "token_count": 0,
            "over": True
        }
    },
    "environ": {
        "RWKV_T_MAX":  256,
        "RWKV_TORCH_COMPILE": "",
        "WN_FIX_L2WRAP": True,
        "RWKV_MY_TESTING": "",
    },
    "role": {
        "system": {
            "prefix": [65500],
            "postfix": [65535]
        },
        "request": {
            "prefix":  [65510],
            "postfix": [65535]
        },
        "think": {
            "prefix": [65520],
            "postfix": [65535]
        },
        "response": {
            "prefix": [65530],
            "postfix": [65535]
        },
        "text": {
            "prefix": [],
            "postfix": [261]
        },
        "text-raw": {
            "prefix": [],
            "postfix": []
        },

        "question": {
            "prefix": tokenizer.encode('Question: '),
            "postfix": [261]
        },
        "answer": {
            "prefix": tokenizer.encode('Answer: '),
            "postfix": [261]
        },
        "user": {
            "prefix": tokenizer.encode('User: '),
            "postfix": [261]
        },
        "assistant": {
            "prefix": tokenizer.encode('Assistant: '),
            "postfix": [261]        },
        "instruction": {
            "prefix": tokenizer.encode('## Instruction:\n'),
            "postfix": [261]
        },
        "input": {
            "prefix": tokenizer.encode('## Input:\n'),
            "postfix": [261]
        },
        "output": {
            "prefix": tokenizer.encode('## Output:\n'),
            "postfix": [261]
        },
        "robot.listen.from-user": {
            "prefix":  [65510],
            "postfix": [65535]
        },
        "robot.thoughts": {
            "prefix": [65520],
            "postfix": [65535]
        },

        "robot.reasoning": {
            "prefix": [65522],
            "postfix": [65535]
        },

        "robot.plan": {
            "prefix": [65524],
            "postfix": [65535]
        },

        "robot.plan": {
            "prefix": [65524],
            "postfix": [65535]
        },
        "robot.response": {
            "prefix": [65530],
            "postfix": [65535]
        },
        "data.text": {
            "prefix": [],
            "postfix": [261]
        },
        "data.text.raw": {
            "prefix": [],
            "postfix": []
        },
        "infer.form.question": {
            "prefix": tokenizer.encode('Question: '),
            "postfix": [261]
        },
        "infer.form.answer": {
            "prefix": tokenizer.encode('Answer: '),
            "postfix": [261]
        },
        "infer.form.q": {
            "prefix": tokenizer.encode('Q: '),
            "postfix": [261]
        },
        "infer.form.a": {
            "prefix": tokenizer.encode('A: '),
            "postfix": [261]
        },

        "infer.form.user": {
            "prefix": tokenizer.encode('User: '),
            "postfix": [261]
        },
        "infer.form.assistant": {
            "prefix": tokenizer.encode('Assistant: '),
            "postfix": [261]        },
        "infer.form.instruction": {
            "prefix": tokenizer.encode('## Instruction:\n'),
            "postfix": [261]
        },
        "infer.form.input": {
            "prefix": tokenizer.encode('## Input:\n'),
            "postfix": [261]
        },
        "infer.form.output": {
            "prefix": tokenizer.encode('## Output:\n'),
            "postfix": [261]
        },
        "org.level1": {
            "prefix": tokenizer.encode('\n* '),
            "postfix": [261]
        },
        "org.level2": {
            "prefix": tokenizer.encode('\n** '),
            "postfix": [261]
        },
        "org.level3": {
            "prefix": tokenizer.encode('\n*** '),
            "postfix": [261]
        },
        "org.level4": {
            "prefix": tokenizer.encode('\n**** '),
            "postfix": [261]
        },
        "code.md": {
            "prefix": tokenizer.encode('\n```\n'),
            "postfix": tokenizer.encode('\n```\n'),
        },
        "code.md.javascript": {
            "prefix": tokenizer.encode('\n```javascript\n'),
            "postfix": tokenizer.encode('\n```\n'),
        },
        "code.md.python": {
            "prefix": tokenizer.encode('\n```python\n'),
            "postfix": tokenizer.encode('\n```\n'),
        },
        "code.md.cmd": {
            "prefix": tokenizer.encode('\n```cmd\n'),
            "postfix": tokenizer.encode('\n```\n'),
        },
        "code.md.clojure": {
            "prefix": tokenizer.encode('\n``` clojure\n'),
            "postfix": tokenizer.encode('\n```\n'),
        },
        "code.org": {
            "prefix": tokenizer.encode('\n#+begin_src \n'),
            "postfix": tokenizer.encode('\n#+end_src\n'),
        },
        "code.org.clojure": {
            "prefix": tokenizer.encode('\n#+begin_src clojure\n'),
            "postfix": tokenizer.encode('\n#+end_src\n'),
        },
        "code.org.python": {
            "prefix": tokenizer.encode('\n#+begin_src python\n'),
            "postfix": tokenizer.encode('\n#+end_src\n'),
        },
        "code.org.edn": {
            "prefix": tokenizer.encode('\n#+begin_data edn\n'),
            "postfix": tokenizer.encode('\n#+end_data\n'),
        },






    }}

config['trainer']['tokenizer'] = TRIE_TOKENIZER(config['trainer']['tokenizer'])
print(f"===train===={config['trainer']['tokenizer'].decode([65528])}==")
config['inference']['tokenizer'] = TRIE_TOKENIZER(config['inference']['tokenizer'])
print(f"===inference===={config['inference']['tokenizer'].decode([65528])}==")
