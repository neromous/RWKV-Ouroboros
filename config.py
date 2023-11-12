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
            "token_ban": [0, 65530, 65531, 65532, 65533, 65534],
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
            # "prefix": tokenizer.encode('System: '),
            # "postfix": [261]
            "prefix": [65530, 65531],
            "postfix": [65535]
        },
        "ego": {
            "prefix": [65530, 65531],
            "postfix": [65535]
        },
        "master": {
            "prefix": [65530, 65532],
            "postfix": [65535]
        },
        "request": {
            "prefix": [65530, 65532],
            #"prefix": tokenizer.encode('User: '),
            "postfix": [261]
        },
        "user": {
            "prefix": [65530, 65532],
            "postfix": [65535]
        },
        "think": {
            "prefix": [65530, 65533],
            "postfix": [65535]
        },
        "analysis": {
            "prefix": [65530, 65533],
            "postfix": [65535]
        },
        "response": {
            "prefix": [65530, 65534],
            #"prefix": tokenizer.encode('Assistant: '),
            "postfix": [261]
        },
        "robot": {
            "prefix": [65530, 65534],
            "postfix": [65535]
        },
        "claude": {
            "prefix": [65530, 65534],
            "postfix": [65535]
        },
        "book": {
            "prefix": [],
            "postfix": [261]
        },
        "context": {
            "prefix": [],
            "postfix": []
        },
        "other": {
            "prefix": [],
            "postfix": []
        },
        "text": {
            "prefix": [],
            "postfix": [261]
        }
    }}

config['trainer']['tokenizer'] = TRIE_TOKENIZER(config['trainer']['tokenizer'])
print(f"===train===={config['trainer']['tokenizer'].decode([65528])}==")
config['inference']['tokenizer'] = TRIE_TOKENIZER(config['inference']['tokenizer'])
print(f"===inference===={config['inference']['tokenizer'].decode([65528])}==")
