import requests
# import sys

cmd = input("prompt-> ")

m = requests.post("http://172.16.2.197:3000/inference",
                  json={"messages": [

                      {"content":"" +  cmd.strip() +"\n\n" ,
                       "over": True,
                       "role": "user",
                       "token_count": 0
                       },
                      {"content": "",
                       "temperature":2.0,
                       "top_p":0.2,
                       "over": False,
                       "role": "robot",
                       "token_count": 1024,
                       "token_stop":[0,65535]}]})

print(m.json())
