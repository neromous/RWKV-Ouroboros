import json
from types import SimpleNamespace
from util.prefix_tokenizer import prefix_tokenizer


def dict_to_namespace(d) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def load_config(path) -> SimpleNamespace:
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    config_data = json.loads(text)
    args = dict_to_namespace(config_data)
    return args


def load_tokenizer(config, usage, sp_map=False):
    tokenizer_path = config['tokenizer'][usage]
    if sp_map:
        return prefix_tokenizer(file_name=tokenizer_path, sp_map=sp_map)
    else:
        return prefix_tokenizer(file_name=tokenizer_path)
