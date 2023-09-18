from llm_datasets.sft import Sft
from models.scene import Scene
from models.message import Message
import requests

# data = Sft
# data.load()

# print()
# messages = []
# for x in data.find_all(section_id=1):
#     item = x.__dict__
#     item.update({"token_count":256,"over": False})
#     messages.append(item)



# train_data = Scene.new({"messages": messages})


# m = requests.post("http://0.0.0.0:7000/inference/generate",
#                   json={"messages" : messages,

#                         })



# print(train_data)
