import requests
import sys
import json
# 用于存储API返回的上下文
class gpt:
    def send_request(self,messages):
        # 设置代理服务器的地址和端口
        proxies = {
            "http": "http://127.0.0.1:56598",
            "https": "http://127.0.0.1:56598"
        }
        # ChatGPT API的URL
        url = "https://api.openai-sb.com/v1/chat/completions"
        # ChatGPT API的访问密钥
        # api_key = "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"
        # api_key= "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"
        # api_key="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"
        api_key="sb-0421008571f399caf996743f1d884b0fcc8034bf83e30359"
        # api_key = "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"
        # 请求参数
        parameters = {
                      "model": "gpt-3.5-turbo-0301",
            #gpt-3.5-turbo-0301
                      "messages":messages
            # [{"role": "user", "content": context}]
                    }
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 发送请求
        response = requests.post(url,
                                 headers=headers,
                                 json=parameters,
                                 #proxies=proxies
                                 )

        # 解析响应
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]

            return text
        else:
            print(response)
            return "Sorry, something went wrong."

    def start_conversation(self,messages):
        print("\n=====chatgpt=====")
        user_input =  sys.stdin.readlines()
        user_input = "\n".join(user_input)
        user_input = user_input.strip()
        user_message={"role": "user", "content": user_input}
        # 初始化系统
        system_message = {"role": "system", "content": "你是一个聪明的助手，会努力一步步的分析和思考问题，给出你经过深思熟虑得出的结论。"}
        messages.append(system_message)
        # 将用户输入添加到messages中
        messages.append(user_message)
        print("\n=====chatgpt=====")
        # 发送API请求
        response = self.send_request(messages)
        # 输出API返回内容
        print(response["content"].strip())
        # text = json.dumps({"system":system_message['content'],
        #            "question": user_message['content'],
        #            "answer":response["content"].strip()},
        #           ensure_ascii=False)

        # with open('/home/neromous/Scripts/output/chatgpt.jsonl',
        #           'a',encoding='utf-8') as f:
        #     f.write(text +"\n" )

        #将API接口返回的内容添加至messages，以用作多轮对话

if __name__ == '__main__':
    messages=[] #初始化prompt
    gpt().start_conversation(messages)
