import json


def read_config() -> dict:
    with open('./config.json', 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res


def get_environ(k) -> dict:
    config = read_config()
    return config["environ"].get(k, False)


def get_trainer(k) -> dict :
    config = read_config()
    return config["trianer"].get(k, False)

if __name__ == "__main__":
    print(read_config())
    print("====", get_environ("RWKV_JIT_ON"))
