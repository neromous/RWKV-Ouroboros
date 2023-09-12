import json
import requests

host = "http://172.16.0.62:3000"

all_data = {}

# :data.qa/answer
def find_eids(ek):
    query = f"[:find [?e ...] :where [?e {ek}]]"
    resp = requests.post(url=f"{host}/api/datomic/query",
                         json={"query-string": query})
    return resp.json()['total']

def search_eids(ek,text):
    query = f"[:find [?e ...] :in $ :where [(fulltext $ {ek} {text}) [[?e ?name ?tx ?score]]]]"
    resp = requests.post(url=f"{host}/api/datomic/query",
                         json={"query-string": query})
    return resp.json()['total']

print("====",find_eids(":data.qa/answer")[0:10])

print("====",search_eids(":data.qa/answer", "\"太阳\"")[0:10])


# all_data['qa'] = resp.json()['total']
# resp = requests.post(url=f"{host}/api/datomic/query",
#                      json={"query-string": "[:find [?e ...] :where [?e :facts/content]]"})

# all_data['facts'] = resp.json()['total']
# with open('../datomic.json','w', encoding='utf-8') as f:
#     text = json.dump(all_data, f, ensure_ascii=False)

#item = requests.post(url=f"{host}/api/datomic/pull",
#                     json={"eids": eids})
#print(item.json())
