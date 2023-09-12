import requests

m = requests.get("http://172.16.2.197:3000/reset-state")
print(m.json())
