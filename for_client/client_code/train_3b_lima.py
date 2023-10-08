import requests
import sys
import orgparse
from orgparse import loads
import sys
import re
from model import Inference

env = Inference('3b')

resp = env.teach_by_org('/home/neromous/Documents/Workspace/projects/sft/lima.org')
for x in resp:
    print(x)
