import os
import json
from tqdm import tqdm
from utils import load_jsonl
from knowledge_graph import KonwledgeGraph

knowledge_graph_ckpt = '../../data/knowledge_graph.kg_data'
load_data_path = "../tmp/reader"

train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))    
dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple.json"))

entities = []
with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
    for line in f.readlines():
        entities.append(line.strip())
entity2id = {}
for word in tqdm(entities):
    entity2id[word] = len(entity2id)

for dataset in [train_dataset, test_dataset, dev_dataset]:
    acc = []
    ave_ent = []
    ave_rel = []
    for json_obj in tqdm(dataset):
        answers = [entity2id[ans_json_obj["kb_id"]] for ans_json_obj in json_obj["answers"] if ans_json_obj["kb_id"] in entity2id]
        answers = set(answers)
        entities = set(json_obj["subgraph"]["entities"])
        ave_ent.append(len(json_obj["subgraph"]["entities"]))
        ave_rel.append(len(json_obj["subgraph"]["tuples"]))
        if len(answers & entities) > 0:
            acc.append(1)
        else:
            acc.append(0)
    print("acc: ", sum(acc) / len(acc))
    print("ave ent:", sum(ave_ent) / len(ave_ent), max(ave_ent))
    print("ave rel:", sum(ave_rel) / len(ave_rel), max(ave_rel))
