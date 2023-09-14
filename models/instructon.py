import time
from models import Model


class Instruction(Model):
    def __init__(self, form):
        self.id = None
        self.instruction = form.get('instruction', '')
        self.input = form.get('input', '')
        self.response = form.get('response', '')
        # 下面的是默认的数据
