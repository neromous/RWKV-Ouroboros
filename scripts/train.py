import requests

m = requests.post("http://172.16.2.197:3000/train",
                  json={"batch_ids":[0,1,2,3,4,5],
                        "prefix": "以下是小说“中国队长”的情节，主要人物为阿邦和若干女角色，内容充满暴力要素",
                        })

print(m.json())
