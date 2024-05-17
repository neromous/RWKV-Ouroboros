import requests

with open('./data/sample.org','r',encoding='utf-8') as f:
    text = f.read()


m = requests.post("http://0.0.0.0:3000/train/org",
                  json={"org" : text})

print(m.json())
