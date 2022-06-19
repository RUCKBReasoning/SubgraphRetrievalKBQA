import os
import json
from tqdm import tqdm
from utils import load_jsonl
from loguru import logger

# logger.add("eval.log")

knowledge_graph_ckpt = '../tmp/knowledge_graph.kg_data'
load_data_path = "../tmp/reader"

# train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
# test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))    
dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple.json"))

# for dataset in [train_dataset, test_dataset, dev_dataset]:
for dataset in [dev_dataset]:
    acc = []
    ave_ent = []
    ave_rel = []
    for json_obj in tqdm(dataset):
        answers = [ans_json_obj["kb_id"] for ans_json_obj in json_obj["answers"]]
        answers = set(answers)
        subgraph_entities = set(json_obj["subgraph"]["entities"])
                        
        ave_ent.append(len(json_obj["subgraph"]["entities"]))
        ave_rel.append(len(json_obj["subgraph"]["tuples"]))
        # logger.info("entities: {}".format(json_obj["subgraph"]["entities"]))
        # logger.info("answers: {}".format(answers))
        if len(answers & subgraph_entities) > 0:
            acc.append(1)
        else:
            acc.append(0)
    print("acc: {:.5f}".format(sum(acc) / len(acc)))
    print("ave ent: {:.5f}, {}".format(sum(ave_ent) / len(ave_ent), max(ave_ent)))
    print("ave rel: {:.5f}, {}".format(sum(ave_rel) / len(ave_rel), max(ave_rel)))
