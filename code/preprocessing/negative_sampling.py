from copy import deepcopy
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from knowledge_graph import KonwledgeGraph
from utils import load_jsonl, dump_jsonl
from func_timeout import func_set_timeout, FunctionTimedOut

@func_set_timeout(30)
def generate_data_list(path_json_obj, json_obj, kg):
    new_data_list = []
    neg_num = 15

    path = path_json_obj["path"]
    path.append("END OF HOP")
    question = json_obj["question"] + " [SEP]"
    topic_entities = json_obj["topic_entities"]
    
    candidate_entities = deepcopy(topic_entities)
    candidate_neg_rels = []
    for rel in path[:-1]:
        candidate_neg_rels.extend(list(set(sum([kg.get_relation(h) for h in candidate_entities], ()))))
        
        next_topic_entities = []
        for h in topic_entities:
            next_topic_entities.extend(kg.get_tail(h, rel))
        next_topic_entities = list(set(next_topic_entities))
        candidate_entities = next_topic_entities
    candidate_neg_rels = list(set(candidate_neg_rels))
    
    if len(candidate_neg_rels) < neg_num + 1:
        return []
    
    for rel in path:
        data_row = []
        data_row.append(question)
        data_row.append(rel)
        neg_rels = list(set(sum([kg.get_relation(h) for h in topic_entities], ()))) + ["END OF HOP"]
        neg_rels = [r for r in neg_rels if r != rel]
        if len(neg_rels) < neg_num:
            ext_rels = [r for r in candidate_neg_rels if r != rel and r not in neg_rels]
            neg_rels.extend(random.sample(ext_rels, neg_num - len(neg_rels)))
        
        neg_rels = random.sample(neg_rels, neg_num)
        data_row.extend(neg_rels)
        
        new_data_list.append(data_row)
        
        # update for next step
        if rel != "END OF HOP":
            next_question = question + f" {rel} #"
            next_topic_entities = []
            for h in topic_entities:
                next_topic_entities.extend(kg.get_tail(h, rel))
            next_topic_entities = list(set(next_topic_entities))
            
            question = next_question
            topic_entities = next_topic_entities
    return new_data_list


def run_negative_sampling():
    threshold = 0.5
    kg = KonwledgeGraph.load_from_ckpt("../../data/knowledge_graph.kg_data")
    data_list = load_jsonl("../tmp/train_with_path_score.jsonl")
    new_data_list = []
    timeout_count = 0
    for json_obj in tqdm(data_list, desc="negative-sampling"):
        path_and_score_list = json_obj["path_and_score_list"]
        for path_json_obj in path_and_score_list:
            score = path_json_obj["score"]
            if score < threshold:
                continue
            try:
                data = generate_data_list(path_json_obj, json_obj, kg)
            except FunctionTimedOut as e:
                timeout_count += 1
                print("timeout: {}".format(e.msg))
                continue
            new_data_list.extend(data)            
    
    print("timeout_count:", timeout_count)
    
    new_data_list = np.array(new_data_list)
    df = pd.DataFrame(new_data_list)
    df.to_csv("../tmp/retriever/multi_hop_train.csv", header=False, index=False)
