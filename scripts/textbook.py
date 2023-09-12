import requests

m = requests.post("http://172.16.2.197:3000/textbook",
                  json={"batch_ids": [x for x in range(2070,2140)],
                        "step": 70
                        })

print(m.json())
