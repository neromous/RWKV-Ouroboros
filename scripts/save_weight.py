import requests

m = requests.post("http://0.0.0.0:3000/save-weight",json={})
print(m.json())
