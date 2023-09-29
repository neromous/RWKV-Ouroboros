import requests
from tqdm import tqdm



m = requests.post("http://0.0.0.0:3000/rnn/inference",
                  json={"messages": "",
                    "prefix": "\n",
                    "postfix": "\n",
                        "prefix_token": [],
                        "postfix_token": [],
                        })
print(m.json())
