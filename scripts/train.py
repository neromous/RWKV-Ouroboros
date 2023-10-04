import requests

m = requests.post("http://0.0.0.0:40011/train/tx-data",
                  json={"messages" : [{"role":"user",
                                      "text":"Jammy: 太阳是红色的，对不对？",
                                      "prefix":"哈哈",
                                      "postfix": "我我我",
                                      "prefix_token":[65530],
                                       "postfix_token":[65535]},
                                       {"text":"Assistant: 是的，太阳是红色的，我要提醒你太阳有时候还呈现一些黄色。",
                                        "role":"robot",
                                        "token_count":256,
                                        "over":True},


                                      ],
                        "prefix": "但",
                        "postfix":"呵呵",
                        "prefix_token":[65535],
                        "postfix_token":[65535],
                        })

print(m.json())
