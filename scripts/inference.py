import requests


data = [{"text":"User: 你好啊","role":"user","token_count":0},
        {"text":"Assistant: ",
                         "role":"robot",
                         "token_count":256,
                         "over":False},
        {"text":"User: 我想见你","role":"user","token_count":0},
        {"text":"Assistant: ",
                         "role":"robot",
                         "token_count":256,
                         "over":False}

        ]

m = requests.post("http://0.0.0.0:7000/inference/generate",
                  json={"messages" : data,

                        })

print(m.json())
