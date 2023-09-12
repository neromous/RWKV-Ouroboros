import requests

m = requests.post("http://172.16.2.197:3000/train",
                  json={"batch_ids":[0,1,2,3,4,5],
                        "prefix": "",
                        })

print(m.json())
