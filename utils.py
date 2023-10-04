import os.path
import time
import json
import sys

def save_data(item: dict):
    text = json.dumps(item, ensure_ascii=False)
    text = text + "\n"
    with open('./data/log.jsonl', 'a', encoding='utf-8') as f:
        f.write(text)

def load_config(config_path="./config_scripts/config.json") -> dict:
    if sys.argv[1] == "3b":
        config_path = "./config_scripts/config_3b.json"
    elif sys.argv[1] == "7b":
        config_path = "./config_scripts/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    res = json.loads(text)
    return res


def log(*args, **kwargs):
    # time.time() 返回 unix time
    # 如何把 unix time 转换为普通人类可以看懂的格式呢？
    format = '%H:%M:%S'
    value = time.localtime(int(time.time()))
    dt = time.strftime(format, value)
    with open('log.txt', 'a', encoding='utf-8') as f:
        print(dt, *args, file=f, **kwargs)
