import time
from models import Model


class Text(Model):
    def __init__(self, form):
        self.id = None
        self.title = form.get('text', '')
        self.text = form.get('text', '')
        # 下面的是默认的数据
