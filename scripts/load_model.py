import requests


m = requests.post("http://0.0.0.0:3000/train/weight-to-cpu",
                  json={"messages" : ""})

m = requests.post("http://0.0.0.0:3000/inference/load-model",
                  json={"messages" : ""})

print(m.json())
