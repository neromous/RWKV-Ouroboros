import requests
import sys
import orgparse
from orgparse import loads
import sys
import re
from model import Inference

# host = "172.16.2.197"

prompt = sys.stdin.readlines()
prompt = [x.strip() for x in prompt]
prompt = [x for x in prompt if x != ""]
prompt = "\n".join(prompt)
prompt = prompt.strip()

env = Inference('3b')
messages = env.text2message(prompt)


item = messages[0]

if 'reset' in item['shallow_tags']:
    env.reset_state()


if 'train' in item['shallow_tags']:
    resp = env.teach(messages)
    loss = resp["loss"]
    res = ""
    if loss > 2:
        res = "太难了，我没怎么学会，需要更多的参考资料看看。"
    elif loss < 0.1:
        res = "我想我学了太多遍这些东西了，已经能够精确的表达这些东西，应该换换口味学点别的"
    elif loss <0.5:
        res = "这个知识我掌握的差不多了。回答的准确率应该还行"
    elif loss < 1.0:
        res = "我感觉学得还算明白， 应该能流畅的表达。"
    elif loss < 1.5:
        res = "马马虎虎吧，好像我表达的不够清晰和准确。"
    else:
        res = "完全没学会，我会努力再试试。 "
    print(f"** ROBOT 琉璃： 我刚刚试着学习了这些知识。结果{res}")

elif 'lima' in item['shallow_tags']:
    resp = env.teach_by_org('/home/neromous/Documents/Workspace/projects/sft/lima.org')
    loss = resp["loss"]
    res = ""
    if loss > 2:
        res = "太难了，我没怎么学会，需要更多的参考资料看看。"
    elif loss < 0.1:
        res = "我想我学了太多遍这些东西了，已经能够精确的表达这些东西，应该换换口味学点别的"
    elif loss <0.5:
        res = "这个知识我掌握的差不多了。回答的准确率应该还行"
    elif loss < 1.0:
        res = "我感觉学得还算明白， 应该能流畅的表达。"
    elif loss < 1.5:
        res = "马马虎虎吧，好像我表达的不够清晰和准确。"
    else:
        res = "完全没学会，我会努力再试试。 "
    print(f"** ROBOT 琉璃： 我刚刚试着学习了这些知识。结果{res}")

elif 'benchmark' in item['shallow_tags']:
    with open('/home/neromous/Documents/Workspace/projects/sft/benchmark.org','r',encoding='utf-8') as f:
        text = f.read()
    messages = env.text2message(text)
    for message in messages:
        env.reset_state()
        print(f"\n** REQUEST {message['text']}")
        msg = [message]
        msg = env._question(msg)
        msg = [env.add_role(x) for x in msg]
        resp = env.question(msg)
        msg = resp['messages']

        for item in msg:
            if item['response'].__len__() != 0:
                role = item['role']
                prefix = item['prefix']
                text = item['text']
                result_text = item['response']
                result_text = result_text.replace('<|me|><|system|>', '\n*** SYSTEM ')
                result_text = result_text.replace('<|me|><|response|>', '\n*** RESPONSE ')
                result_text = result_text.replace('<|me|><|request|>', '\n*** REQUEST ')
                result_text = result_text.replace('<|me|><|think|>', '\n*** ANALYSIS ')
                result_text = result_text.replace('<|over|>', '')
                result_text = result_text.replace('<|me|>', '')
                result_text = re.sub('\n+', '\n',result_text)
                print( f"\n** {role.upper()} {prefix + text}" + result_text + "\n", end="")



else:
    messages = env.middleware_for_question(messages)
    messages = env._question(messages)
    messages = [env.add_role(x) for x in messages]
    resp = env.question(messages)
    messages = resp['messages']

    for item in messages:
        if item['response'].__len__() != 0:
            role = item['role']
            prefix = item['prefix']
            text = item['text']
            result_text = item['response']
            result_text = result_text.replace('<|me|><|system|>', '\n** SYSTEM ')
            result_text = result_text.replace('<|me|><|response|>', '\n** RESPONSE ')
            result_text = result_text.replace('<|me|><|request|>', '\n** REQUEST ')
            result_text = result_text.replace('<|me|><|think|>', '\n** ANALYSIS ')
            result_text = result_text.replace('<|over|>', '')
            result_text = result_text.replace('<|me|>', '')
            result_text = re.sub('\n+', '\n',result_text)
            print( f"\n** {role.upper()} {prefix + text}" + result_text + "\n", end="")
