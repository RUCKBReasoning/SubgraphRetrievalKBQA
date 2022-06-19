from copy import deepcopy
import json
import os
import random
from time import time
from tkinter import ALL
import numpy as np
import pandas as pd
from tqdm import tqdm
from knowledge_graph import KonwledgeGraph
from utils import load_jsonl, load_json
from func_timeout import func_set_timeout, FunctionTimedOut

END_REL = "END OF HOP"

ALL_relations = set()

# @func_set_timeout(5)
def generate_data_list(path_json_obj, json_obj, pos_rels, hard_neg_rels, kg):
    new_data_list = []
    neg_num = 15

    path = path_json_obj["path"]
    path = path + [END_REL]
    question = json_obj["question"] + " [SEP]"
    topic_entities = json_obj["topic_entities"]
    
    filter_threshold = 5
    current_filter_threshold = 1
    filter_flag = False
    
    candidate_entities = deepcopy(topic_entities)
    # candidate_neg_rels = []

    for rel in path[:-1]:
        current_filter_threshold *= filter_threshold
        # candidate_neg_rels.extend(list(set(sum([kg.get_relation(h) for h in candidate_entities], ()))))
        
        next_topic_entities = []
        for h in topic_entities:
            next_topic_entities.extend(kg.get_tail(h, rel))
        next_topic_entities = list(set(next_topic_entities))
        # candidate_entities = next_topic_entities
        
        if len(candidate_entities) > current_filter_threshold:
            filter_flag = True
    
    # candidate_neg_rels = list(set(candidate_neg_rels))
    candidate_neg_rels = list(ALL_relations)
    
    # print(len(candidate_neg_rels))
    
    if filter_flag:
        return None

    # print("A1:", time())

    prefix_list = []
    for rel in path:
        # print("B0:", time())
        prefix = ",".join(prefix_list)
        prefix_list.append(rel)
        
        data_row = []
        data_row.append(question)
        data_row.append(rel)

        # print("B1:", time())
        neg_rels = []
        for h in topic_entities:
            h_rels = kg.get_relation(h)
            neg_rels.extend(h_rels)
            if len(neg_rels) > 100:
                break
        neg_rels = list(set(neg_rels))
        neg_rels.append(END_REL)
        neg_rels = [r for r in neg_rels if r not in pos_rels[prefix]]
        if len(neg_rels) < neg_num:
            neg_rels_set = set(neg_rels)
            ext_rels = [r for r in candidate_neg_rels if r not in pos_rels[prefix] and r not in neg_rels_set]
            assert len(ext_rels) > 0
            while len(neg_rels) < neg_num:
                neg_rels.extend(ext_rels)
        neg_rels = random.sample(neg_rels, neg_num)
        # print("B2:", time())        
        
        if prefix in hard_neg_rels:
            neg_rels = list(hard_neg_rels[prefix]) + neg_rels
        neg_rels = neg_rels[:neg_num]
        
        data_row.extend(neg_rels)
        new_data_list.append(data_row)
        
        # update for next step
        if rel != END_REL:
            # print("B5:", time())
            next_question = question + f" {rel} #"
            next_topic_entities = []
            for h in topic_entities:
                next_topic_entities.extend(kg.get_tail(h, rel))
            next_topic_entities = list(set(next_topic_entities))
            # print("B6:", time())            
            question = next_question
            topic_entities = next_topic_entities
    return new_data_list


def run_negative_sampling():
    threshold = 0.5
    kg = KonwledgeGraph.load_from_ckpt("../tmp/knowledge_graph.kg_data")
    data_list = load_jsonl("../tmp/retriever/train_with_path_score.jsonl")
    update_paths = {}
    if os.path.exists("../tmp/retriever/update_paths.json"):
        update_paths = load_json("../tmp/retriever/update_paths.json")

    global ALL_relations
    for _, rels in kg.head2relation.items():
        for rel in rels:
            ALL_relations.add(rel)

    new_data_list = []
    timeout_count = 0
    for json_obj in tqdm(data_list, desc="negative-sampling"):
        question = json_obj["question"]
        path_and_score_list = json_obj["path_and_score_list"]
        path_and_score_list = [path_json_obj for path_json_obj in path_and_score_list if path_json_obj["score"] >= threshold]
        pos_rels = {}  # 1-hop positive, 2-hop positive, ...
        for path_json_obj in path_and_score_list:
            path = path_json_obj["path"]
            path = path + [END_REL]
            prefix_list = []
            for rel in path:
                prefix = ",".join(prefix_list)
                if prefix not in pos_rels:
                    pos_rels[prefix] = set()
                pos_rels[prefix].add(rel)
                prefix_list.append(rel)
        
        hard_neg_rels = {}
        if question in update_paths:
            for neg_rels in update_paths[question]["neg_paths"]:
                prefix_list = []
                for rel in neg_rels:
                    prefix = ",".join(prefix_list)
                    if prefix in pos_rels and rel in pos_rels[prefix]:
                        continue
                    if prefix not in hard_neg_rels:
                        hard_neg_rels[prefix] = set()
                    hard_neg_rels[prefix].add(rel)
        
        for path_json_obj in path_and_score_list:
            data = generate_data_list(path_json_obj, json_obj, pos_rels, hard_neg_rels, kg)
            if data is not None:
                new_data_list.extend(data)

    print("timeout_count:", timeout_count)
    
    new_data_list = np.array(new_data_list)
    df = pd.DataFrame(new_data_list)
    df.to_csv("../tmp/retriever/weak_supervised_train_test.csv", header=False, index=False)

