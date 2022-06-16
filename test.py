import os
import json
from tqdm import tqdm
from typing import List, Any

def load_jsonl(path: str):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list

def dump_jsonl(data_list: List[Any], path: str):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj) + "\n")

load_data_path = "/home/huyuxuan/projects/SubgraphRetrievalKBQA-main/code/tmp/reader_origin"
dump_data_path = "/home/huyuxuan/projects/SubgraphRetrievalKBQA-main/code/tmp/reader"

train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))    
dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple.json"))

entities = []
rels = []

with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
    for line in f.readlines():
        entities.append(line.strip())

with open(os.path.join(load_data_path, "relations.txt"), "r") as f:
    for line in f.readlines():
        rels.append(line.strip())

ent2id = {}
for word in entities:
    ent2id[word] = len(ent2id)

rel2id = {}
for word in rels:
    rel2id[word] = len(rels)

for dataset, dataset_name in zip(
    [train_dataset, test_dataset, dev_dataset], 
    ["train_simple.json", "test_simple.json", "dev_simple.json"]
):
    for json_obj in tqdm(dataset):
        entities = [ent2id[e] for e in json_obj["entities"]]
        subgraph_entities = [ent2id[e] for e in json_obj["subgraph"]["entities"]]
        subgraph_tuples = [(ent2id[h], r, ent2id[t]) for h, r, t in json_obj["subgraph"]["tuples"]]
        
        json_obj["entities"] = entities
        json_obj["subgraph"]["entities"] = subgraph_entities
        json_obj["subgraph"]["tuples"] = subgraph_tuples
    dump_jsonl(dataset, os.path.join(dump_data_path, dataset_name))
