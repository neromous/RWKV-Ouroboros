import requests



m = requests.post("http://0.0.0.0:40011/inference/load-model",
                  json={"messages" : ""})

print(m.json())
