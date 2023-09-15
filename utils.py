import os.path
import time
import json

def load_config() -> dict:
    with open("./config.json","r",encoding="utf-8") as f:
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
