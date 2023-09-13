# RWKV-Ouroboros
This project is established for real-time training of the RWKV model.


```
The snake that eats its own tail, finds sustenance in infinity. 
```



鉴于大语言模型的发展，人与大语言模型之间的关系和未来产生了很多困扰和猜测。 本项目根据一个基本假设建立，即“人在回路”是人类最好的未来。 通过建立低成本的人在回路实现大语言模型与人类的和谐发展

In light of the development of large language models, the relationship between humans and large language models and the future have caused much distress and speculation. This project is established based on a fundamental assumption that "human in the loop" is the best future for humankind. By establishing low-cost human-in-the-loop systems, we can achieve harmonious development between large language models and humans.


## 项目组成

- app.py # 用于模型启动，建立一个基于bottle的服务，通过api暴露训练和数据加载接口
- scripts # 交互脚本 代码很简单， 
  - inference.py 
  - train.py 
- src
  - model.py 训练模型类
  - model_run.py 推理类


项目启动代码
```bash
deepspeed --num_nodes=1 --num_gpus=1 app.py --deepspeed --deepspeed_config ds_config.config
```

在3000端口建立服务

资源需求
3B 显存 16gb， 内存128
7b 显存 20gb， 内存150g以上


## 路线图
- 推理结果存档、修正、训练、存储。
- 提供训练与推理实时同步机制（cpu base）
- 提供实时图数据库基础的训练数据管理（基于datomic或pydatomic）
- 增加lora支持
- 减少推理预训练模型间切换时间 （30s-> 5s）
- 增加llama系模型支持。


