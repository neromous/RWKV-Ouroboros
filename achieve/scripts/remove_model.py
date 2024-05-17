import requests

m = requests.post("http://0.0.0.0:3000/inference/remove-model",
                  json={"messages" : ""})

print(m.json())
