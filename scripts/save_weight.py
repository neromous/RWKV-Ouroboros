import requests

m = requests.post("http://0.0.0.0:40011/train/save-weight",json={'model_name':'3b-stage1'})
print(m.json())
