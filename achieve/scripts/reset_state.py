import requests

m = requests.post("http://0.0.0.0:3000/state/reset",json={})
print(m.json())
