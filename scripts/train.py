import requests

m = requests.post("http://0.0.0.0:3000/trainer/by/tx-data",
                  json={"messages" : [{"role":"text-raw",
                                      "text":"Jammy: 太阳是红色的，对不对？",
                                      "prefix":"哈哈",
                                      "postfix": "我我我",
                                      "prefix_token":[65530],
                                       "postfix_token":[65535]},
                                       {"text":"Assistant: 是的，太阳是红色的，我要提醒你太阳有时候还呈现一些黄色。",
                                        "role":"text-raw",
                                        
                                      },


                                      ],
                      
                        
                        
                      
                        })

print(m.json())
