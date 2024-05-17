import requests


result = {}
while result.get('loss', 2.0) > 0.5:
    m = requests.post("http://0.0.0.0:3000/train/sft",
                      json={"message":"sample"})
    print(m.json())
    result = m.json()



m = requests.post("http://0.0.0.0:3000/train/save-weight",json={})
print(m.json())
