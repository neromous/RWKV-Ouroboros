from utils.tokenizer import TRIE_TOKENIZER


## to config
tokenizer = TRIE_TOKENIZER("./resources/vocab_file/rwkv_vocab_v20230424.txt")

## data
config = {
    "model_name": "default",
    "proj_dir": "/home/neromous/RWKV-Ouroboros",
    "port": 3000,
    "debug": False,
    "model": {
        "load_model": "/home/neromous/RWKV-Ouroboros/resources/weights/default.pth",
        "rwkv_version": "v5",
        "n_embd":  2560,
        "n_layer": 32,
        "vocab_size": 65536,
        "ctx_len": 4096,
        "dtype": "fp32",
        "head_size": 64,
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
        "grad_cp": True,
        #"dropout": 0.001,
        "dropout": 0,
        "window": 0,
        "min_loss": 0.5,
        "max_loss": 1.5,
        "min_loss_fix": 0.05,
        "max_loss_fix": 1.05,
        "ctx_parts": "0",
        "tokenizer": "./resources/vocab_file/rwkv_vocab_v20230424.txt",
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
        "lr_init": 2.5e-5,
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
            "token_ban": [0, 65514, 65515],
            "token_stop": [23, 24, 65530,65531,65532,65533,65534,65535],
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
            "prefix": tokenizer.encode('<|sys-s|>system\n') ,
            "postfix": tokenizer.encode('<|sys-e|>\n') ,
        },
        "request": {
            "prefix":  [65532],
            "postfix": [65533]
        },
        "user": {
            "prefix":  [65532],
            "postfix": [65533]
        },
        "think": {
            "prefix": [65528],
            "postfix": [65529]
        },
        "observe": {
            "prefix": [] + tokenizer.encode('## Observe:') + [11],
            "postfix": [],
        },
        "response": {
            "prefix": [65534],
            "postfix": [65535]
        },
        "bot": {
            "prefix": [65534],
            "postfix": [65535]
        },

        "text": {
            "prefix": [],
            "postfix": []
        },
        "raw": {
            "prefix": [],
            "postfix": []
        },
        "im": {
            "prefix": [23, 11],
            "postfix": [24, 11]
        },
        "tokens": {
            "prefix": [],
            "postfix": []
        }
    },
    # 自定义词表
    "vocab": {
        "<|im-s|>": [23],
        "<|im-e|>": [24],
        "<|page|>": [65514],
        "<|page-s|>": [65514],
        "<|page-e|>": [65515],
        "<|env-s|>": [65516],
        "<|env-e|>": [65517],
        "<|cmd-s|>": [65518],
        "<|cmd-e|>": [65519],
        "<|data-s|>": [65520],
        "<|data-e|>": [65521],
        "<|tool-s|>": [65522],
        "<|tool-e|>": [65523],
        "<|in-s|>": [65524],
        "<|in-e|>": [65525],
        "<|out-s|>": [65526],
        "<|out-e|>": [65527],
        "<|think-s|>": [65528],
        "<|think-e|>": [65529],
        "<|sys-s|>": [65530],
        "<|sys-e|>": [65531],
        "<|req-s|>": [65532],
        "<|req-e|>": [65533],
        "<|resp-s|>": [65534],
        "<|resp-e|>": [65535]}}


# "<|system|>": [65531],
# "<|request|>": [65532],
# "<|response|>": [65534],
# "<|over|>": [65535],
# "<|page|>": [11,65530,11],
# "<|page-description|>": [11,65533,11],
# "<|paragraph|>":  [11,65533,11],
# "<|page-over|>": [11,65535,11]

tokenizer_for_train = TRIE_TOKENIZER(config['trainer']['tokenizer'], sp_map=config['vocab'])
tokenizer_for_inference = TRIE_TOKENIZER(config['inference']['tokenizer'],  sp_map=config['vocab'])

config['trainer']['tokenizer'] = tokenizer_for_train
print(f"===train===={config['trainer']['tokenizer'].decode([65528])}==")
config['inference']['tokenizer'] = tokenizer_for_inference
print(f"===inference===={config['inference']['tokenizer'].decode([65528])}==")

test_string ="<|sys-s|><|dfadsfa|>dfasdfds<|sys-e|><|req-s|>dfasdfads<|req-e|><dfad><|resp-s|>dfasdfdsaf<|resp-e|>dfasdf\n"
test_token = tokenizer.encode(test_string)

print("==token==",tokenizer_for_inference.encode(test_string) )
print("---string----",tokenizer_for_inference.decode(test_token))
