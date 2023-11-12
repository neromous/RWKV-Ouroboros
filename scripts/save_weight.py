import requests

m = requests.post("http://0.0.0.0:3000/train/save-weight",json={'model_name':'3b-stage-infctx'})
print(m.json())
