import requests

m = requests.post("http://0.0.0.0:3000/state/reset",json={})
print(m.json())

data = [{"text":"Question: 碟形世界的故事讲述了什么？\n\n",
         "role":"user",
         "token_count":0,
         "over":True},
        {"text":"Answer: ",
         "role":"robot",
         "token_count":512,
         "temperature": 0.2,
         "top_p":0.2,
         "token_stop":[0],
         #"token_stop": [65530,65531,65532,65533,65534,65535],
         "over":False}]

m = requests.post("http://0.0.0.0:3000/inference/generate",
                  json={"messages" : data})

print(m.json())

m = requests.post("http://0.0.0.0:3000/state/reset",json={})
print(m.json())


data = [{"text":"Question: “碟形世界”这本小说中有哪些人物出场了？\n\n",
         "role":"user",
         "token_count":0,
         "over":True},
        {"text":"Answer: ",
         "role":"robot",
         "token_count":512,
         "temperature": 0.2,
         "top_p":0.2,
         #"token_stop": [65530,65531,65532,65533,65534,65535],
         "token_stop":[0],
         "over":False}]

m = requests.post("http://0.0.0.0:3000/inference/generate",
                  json={"messages" : data})

print(m.json())

# m = requests.post("http://0.0.0.0:3000/state/reset",json={})
# print(m.json())


# data = [{"text":"Question: 在刚才更新的那本小说里， 有几个女警和阿邦战斗？请一步一步详细描述，不要忽视细节。\n\n",
#          "role":"user",
#          "token_count":0,
#          "over":True},
#         {"text":"Answer: ",
#          "role":"robot",
#          "token_count":512,
#          "temperature": 0.2,
#          "top_p":0.2,
#          "token_stop": [65530,65531,65532,65533,65534,65535],
#          "over":False}]

# m = requests.post("http://0.0.0.0:3000/inference/generate",
#                   json={"messages" : data})

# print(m.json())


# m = requests.post("http://0.0.0.0:3000/state/reset",json={})
# print(m.json())


# data = [{"text":"Question: 在刚才更新的那本小说里， 赵教授是谁？ 发生了什么事情？请一步一步详细描述，不要忽视细节。\n\n",
#          "role":"user",
#          "token_count":0,
#          "over":True},
#         {"text":"Answer: ",
#          "role":"robot",
#          "token_count":512,
#          "temperature": 0.2,
#          "top_p":0.2,
#          "token_stop": [65530,65531,65532,65533,65534,65535],
#          "over":False}]

# m = requests.post("http://0.0.0.0:3000/inference/generate",
#                   json={"messages" : data})

# print(m.json())

# m = requests.post("http://0.0.0.0:3000/state/reset",json={})
# print(m.json())


# data = [{"text":"Question: 在刚才更新的那本小说里， 阿邦是谁？ 他做了什么事情？为什么他要与女性角色战斗？请一步一步详细描述，不要忽视细节。\n\n",
#          "role":"user",
#          "token_count":0,
#          "over":True},
#         {"text":"Answer: ",
#          "role":"robot",
#          "token_count":512,
#          "temperature": 0.2,
#          "top_p":0.2,
#          "token_stop": [65530,65531,65532,65533,65534,65535],
#          "over":False}]

# m = requests.post("http://0.0.0.0:3000/inference/generate",
#                   json={"messages" : data})

# print(m.json())
