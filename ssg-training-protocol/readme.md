# repretraining by ssg-training-protocol for RWKV-Ouroboros
(临时readme)
---
# 配置

1. 学习率很高， lr_init 与lr_init 建议修改为1e-5
2. （可选）由于当前config文件不支持选择，可以前往utils.py将config改为当前目录的config


# 数据集准备
数据集协议与 https://github.com/ssg-qwq/special_token_json2binidx_tool 中一致
请打开rwkv_vocab_ssg_train.txt词表观看末尾的special tokens（但是因为项目还没有更词表config，目前暂时没有使用这个词表。）

# 加载数据集
```python dataset_manager.py```
将加载demo数据集

# 训练
```pyhon training.py```
将训练demo数据集

想要具体修改，请参照 dataset_manager.py 和 training.py 脚本。