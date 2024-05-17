import requests

host = "http://127.0.0.1:9001"

api = ""


def remark(url, domain, title, content):
    query = {}
    resp = requests.post(url, json=query)
    resp = resp.json()
    return resp


def recall(url, domain, title):
    query = {}
    resp = requests.post(url, json=query)
    resp = resp.json()
    return resp
