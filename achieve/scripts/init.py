import requests
import requests

m = requests.post("http://0.0.0.0:3000/inference/load-model",
                  json={"messages" : ""})

print(m.json())


data = [{"text":"""The following is a coherent verbose detailed conversation between a Chinese girl named Answer and her friend Question.
Answer is very intelligent, creative and friendly.
Answer likes to tell Question a lot about herself and her opinions.
Answer usually gives Question kind, helpful and informative advices.\n\n""",
         "role":"user",
         "token_count":0,
         "over":True},

    {"text":"Question: 什么是lhc？",
         "role":"user",
         "token_count":0,
         "over":True},
        {"text":"""Answer: LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。\n\n""",
         "role":"robot",
         "token_count":0,
         "over":True},
        {"text":"Question: 企鹅会飞吗？\n\n",
         "role":"user",
         "token_count":0,
         "over":True},
        {"text":"Answer: 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。\n\n",
         "role":"robot",
         "token_count":0,
         "over":True}

        ]



m = requests.post("http://0.0.0.0:3000/state/init",json={"messages":data})
print(m.json())
