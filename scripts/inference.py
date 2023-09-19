import requests


data = [{"text":"Question: 根据今天 下雨 日子 不好 写一千字短文\n\n",
         "role":"user",
         "token_count":0,
         "over":True},
        {"text":"Answer: ",
         "role":"robot",
         "token_count":2048,
         "temperature":2.0,
         "top_p":0.75,
         "token_stop": [65530,65531,65532,65533,65534,65535],
         "over":False},



        ]

m = requests.post("http://0.0.0.0:3000/inference/generate",
                  json={"messages" : data,

                        })

print(m.json())
