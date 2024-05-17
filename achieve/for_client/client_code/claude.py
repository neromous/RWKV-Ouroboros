import requests
import sys

prompts = sys.stdin.readlines()
prompts = [x.strip() for x in prompts if len(x.strip()) != 0]
prompts = "\n".join(prompts)


url = "https://xqtd520qidong.com/v1/chat/completions"

# proxies = {
#     "http": "http://127.0.0.1:33229",
#     "https": "http://127.0.0.1:33229"
# }


headers = {
    "Authorization": "sk-MxpiqiX7zsNCW5lC8d1aEb6eEe5548AeBbE2EbBc82E24287",
    "content-type": "application/json"
}
data = {
    "messages": [
        {
            "role": "user",
            "content": prompts,
        }
    ],
    "model": "claude-instant-1-100k",
    "max_tokens_to_sample": 300,
}

response = requests.post(url, headers=headers, json=data)
res = response.json()
res = res['choices'][0]['message']['content']
res = res.split("\n\n")
res = "\n".join(res)
print(f"** CLAUDE {res}")
