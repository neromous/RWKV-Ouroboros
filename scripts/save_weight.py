import requests

m = requests.post("http://0.0.0.0:3000/train/save-weight",json={})
print(m.json())
