import requests

m = requests.post("http://0.0.0.0:3000/trainer/model/save-to-disk",
                  json={'model_name':'default'})
print(m.json())
