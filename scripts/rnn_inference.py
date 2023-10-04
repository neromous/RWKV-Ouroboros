import requests
from tqdm import tqdm



m = requests.post("http://0.0.0.0:40011/rnn/inference",
                  json={"messages": "",
                    "prefix": "\n",
                    "postfix": "\n",
                        "prefix_token": [],
                        "postfix_token": [],
                        })
print(m.json())
