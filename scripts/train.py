import requests

m = requests.post("http://0.0.0.0:3000/train/tx-data",
                  json={"messages" : [{"role":"user",
                                      "text":"遥遥领先，遥遥领先，遥遥领先",
                                      "prefix":"哈哈",
                                      "postfix": "我我我",
                                      "prefix_token":[65530],
                                      "postfix_token":[65535]}],
                        "prefix": "但",
                        "postfix":"呵呵",
                        "prefix_token":[65535],
                        "postfix_token":[65535],
                        })

print(m.json())
