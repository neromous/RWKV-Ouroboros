# config = {
#     "dummy1": [65528],  # 这个token是一个冗余，还没想好做什么
#     "search": [65529],   # 这个token表示被包含的内容为检索结果
#     "me": [65530],  # 这个token表示与AI交互
#     "system": [65531],  # 这个token表示被包含的内容是系统消息
#     "conversation": [65532],  # 这个token表示被包含的内容是对话
#     "think": [65533],  # 这个token表示被包含的内容是AI的思维过程
#     "env": [65534],  # 这个token表示被包含的内容为环境描写
#     "over": [65535],  # 动作结束
# }

# special_tokens = {
#     "chat": [config["me"], config["conversation"]],
#     "think": [config["me"], config["think"]],
#     "search": [config["me"], config["search"]],
#     "env": [config["me"], config["text"]],
#     "common": [config["me"]],
#     "eos": [config["over"]],
# }

import os
import json
import jsonlines
import random
import shutil
import tqdm

# 键为语料中的内容，键不要改
mapping_dict = {
    "data": "sample",  # 每一行的键
    "text": "text",  # 每一个元素的键
    "conversation": ["conversation", "query", "answer"],  # 每一个元素的键
    "system": ["system", "instruct"],  # 每一个元素的键
    "search": "search",  # 每一个元素的键
    "env": "env",  # 每一个元素的键
    "common": "common",
    "think": "think",
    "summary": "summary",
}


class Dataset:
    def __init__(
            self, save_dir, jsonl_folder: str, db_trunk_size=2500, n_sample=100000
    ) -> None:
        dataset_name = None
        mapping = mapping_dict
        ld = os.listdir(jsonl_folder)
        config_file = os.path.basename(jsonl_folder)
        if f"{config_file}.json" in ld:
            with open(
                    os.path.join(jsonl_folder, f"{config_file}.json"), "r", encoding="utf-8"
            ) as cf:
                config = json.load(cf)
                if "n_sample" in config:
                    n_sample = config["n_sample"]
                    del config["n_sample"]
                if "dataset_name" in config:
                    dataset_name = config["dataset_name"] + f":{n_sample}"
                    del config["dataset_name"]
                mapping.update(config)
        if dataset_name is None:
            dataset_name = config_file + f":{n_sample}"
        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)

        jsonl_list = [f for f in ld if f.endswith(".jsonl")]
        for jsonl_path in jsonl_list:
            with open(
                    os.path.join(jsonl_folder, jsonl_path), "r", encoding="utf-8"
            ) as f:
                trunk = []
                n = 0
                loop = tqdm.tqdm(f)
                for line in loop:
                    try:
                        loop.set_postfix(jsonl=jsonl_path)
                        data = json.loads(line)
                    except Exception as e:
                        print(f"line{line} cannot read by json:{e}")
                        continue

                    data["data"] = data.pop(mapping["data"])
                    for ele in data["data"]:
                        old_key = list(ele.keys())[0]
                        if self.match_vs(old_key, mapping):
                            ele[self.find_key_by_value(mapping, old_key)] = ele.pop(
                                old_key
                            )
                    trunk.append(data)
                    if len(trunk) >= db_trunk_size:
                        save_path = os.path.join(
                            save_dir,
                            dataset_name,
                            f"{jsonl_path}-{db_trunk_size * n}-{db_trunk_size * (n + 1)}.jsonl",
                        )
                        # 打开JSONL文件并写入字典数据
                        with jsonlines.open(save_path, mode="w") as writer:
                            writer.write_all(trunk)
                        n += 1
                        trunk = []
                if len(trunk) > 0:
                    save_path = os.path.join(
                        save_dir,
                        dataset_name,
                        f"{jsonl_path}-{db_trunk_size * n + 1}-.jsonl",
                    )
                    # 打开JSONL文件并写入字典数据
                    with jsonlines.open(save_path, mode="w") as writer:
                        writer.write_all(trunk)
                    n += 1
                    trunk = []

    def find_key_by_value(self, dictionary, value_to_find):
        for key, value in dictionary.items():
            if self.match_key(value_to_find, value):
                return key
        # 如果值不存在于字典中，可以返回None或者其他适当的默认值
        return None

    def match_key(self, value, unk):
        if isinstance(unk, str):
            return value == unk
        else:
            return value in unk

    def match_vs(self, value_to_find, unks):
        # 遍历 mapping 的 values()
        for value in unks.values():
            # 如果值是字符串，直接比较
            if isinstance(value, str):
                if value == value_to_find:
                    return True
            else:
                if value_to_find in value:
                    return True
        return False


class DataDB:
    """
    --'db_path'
      --'dataset_folder:n_sample'
        --'data_trunk1.jsonl'
            -{"data":{"text","conversation","system","search","env"}}
        --'data_trunk2.jsonl'
    """

    def __init__(self, path="./dataset_db") -> None:
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def append(self, dataset_path):
        Dataset(save_dir=self.path, jsonl_folder=dataset_path)

    def clear(self):
        try:
            shutil.rmtree(self.path)
            os.mkdir(self.path)
            print(f"训练数据库清空。")
        except Exception as e:
            print(f"训练数据库清空失败：{str(e)}")

    def get_batch(self):
        datasets = os.listdir(self.path)

        datasets = [os.path.join(self.path, f) for f in datasets if ":" in f]
        batch = []
        for dataset in datasets:
            sample = []
            p_sample = float(dataset.split(":")[1])
            if p_sample < 1:
                if random.random() > p_sample:
                    continue
                p_sample = 1
            n_sample = int(p_sample)
            progress_bar = tqdm.tqdm(
                total=n_sample, desc=f"collecting [{dataset.split('/')[-1]}]"
            )
            jsonls = [
                os.path.join(dataset, f)
                for f in os.listdir(dataset)
                if f.endswith(".jsonl")
            ]
            while len(sample) < n_sample:
                n_select = min(1 + len(jsonls) // 12, 16)
                choises = random.sample(jsonls, n_select)
                for f in choises:
                    lines = []
                    with open(f, "r", encoding="utf-8") as file:
                        for line in file:
                            if line.strip():
                                json_line = json.loads(line)
                                lines.append(json_line)
                    nn = n_sample // n_select
                    patch = random.sample(lines, min(len(lines), nn if nn > 0 else 1))
                    sample += patch
                    progress_bar.update(1)  # 更新进度条
                    progress_bar.set_postfix(n=len(sample))
                    if len(sample) >= n_sample:
                        break
            batch += sample
        random.shuffle(batch)
        return batch


if __name__ == "__main__":
    db = DataDB("./dataset_db")
    db.append("./data_sample")
    batch = db.get_batch()
    print(batch[:3])
