import json
import requests
import random
from tqdm import tqdm


def train(data):
    m = requests.post("http://192.168.0.252:3000/read",
                      json={})
    return m

if __name__ == '__main__':
    train({})
