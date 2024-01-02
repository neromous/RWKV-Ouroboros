from utils.tokenizer import TRIE_TOKENIZER

## to config
tokenizer = TRIE_TOKENIZER("./resources/vocab_file/rwkv_vocab_v20230424.txt")

## data
config = {
    "model_name": "default",
    "proj_dir": "/home/xu/liubintao/RWKV-Ouroboros",
    "port": 3000,
    "debug": False,
    "model": {
        "load_model": "/home/neromous/RWKV-Ouroboros/resources/weights/default.pth",
        # "load_model": "/home/xu/Liu Bintao/LM Model/rwkv-v5-7B-0.4-long-ctx-16k.pth",
        "rwkv_version": "v5",
        "n_embd":  2560,
        "n_layer": 32,
        "vocab_size": 65536,
        "ctx_len": 4096,
        "dtype": "bf16",
        "head_size": 64,
        "head_size_a": 64,
        "head_size_divisor": 8,
        "ds_config": './ds_config/ds_config.config',
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
        #"dropout": 0.001,
        "dropout": 0,
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
        "lr_init": 2.0e-5,
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
            "prefix": [65531],
            "postfix": [65535]
        },
        "request": {
            "prefix":  [65532],
            "postfix": [65535]
        },

        "think": {
            "prefix": [65533],
            "postfix": []
        },
        "observe": {
            "prefix": [65533],
            "postfix": [65535],
        },
        "response": {
            "prefix": [65534],
            "postfix": [65535]
        },
        "text": {
            "prefix": [],
            "postfix": []
        },
        "raw-text": {
            "prefix": [],
            "postfix": []
        },
        "text-raw": {
            "prefix": [],
            "postfix": []
        },
        "raw": {
            "prefix": [],
            "postfix": []
        },
        "tokens": {
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
            "postfix": [261]},
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
        }},
    # 自定义词表
    "vocab": {
        "<|system|>": [65531],
        "<|request|>": [65532],
        "<|response|>": [65534],
        "<|over|>": [65535],
        "<|page|>": [11,65530,11],
        "<|page-description|>": [11,65533,11],
        "<|paragraph|>":  [11,65533,11],
        "<|page-over|>": [11,65535,11]
    }}


tokenizer_for_train = TRIE_TOKENIZER(config['trainer']['tokenizer'], sp_map=config['vocab'])
tokenizer_for_inference = TRIE_TOKENIZER(config['inference']['tokenizer'],  sp_map=config['vocab'])


config['trainer']['tokenizer'] = tokenizer_for_train
print(f"===train===={config['trainer']['tokenizer'].decode([65528])}==")
config['inference']['tokenizer'] = tokenizer_for_inference
print(f"===inference===={config['inference']['tokenizer'].decode([65528])}==")


test_string ="<|system|><|dfadsfa|>dfasdfds<|page-over|><|system|>dfasdfads<|over|><dfad><|request|>\n"
test_token = tokenizer.encode(test_string)

print("==token==",tokenizer_for_inference.encode(test_string) )
print("---string----",tokenizer_for_inference.decode(test_token))
