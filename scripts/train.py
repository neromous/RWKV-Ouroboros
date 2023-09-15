import requests

m = requests.post("http://0.0.0.0:3000/train",
                  json={"batch_ids":[0,1,2,3,4,5],
                        "prefix": "",
                        })

print(m.json())
