from copy import deepcopy
import json
import os
import random
from time import time
from tkinter import ALL
import numpy as np
import pandas as pd
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut

from knowledge_graph.knowledge_graph import KnowledgeGraph
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
from utils import load_jsonl
from config import cfg

END_REL = "END OF HOP"

@func_set_timeout(10)
def generate_data_list(path_json_obj, json_obj, pos_rels, kg: KnowledgeGraphFreebase):
    new_data_list = []
    neg_num = 15

    path = path_json_obj["path"]
    path = path + [END_REL]
    question = json_obj["question"] + " [SEP]"
    topic_entities = json_obj["topic_entities"]
    
    filter_threshold = 5
    current_filter_threshold = 1
    filter_flag = False
    

    for rel in path[:-1]:
        current_filter_threshold *= filter_threshold
        
        candidate_entities = set()
        for h in topic_entities:
            candidate_entities |= set(kg.get_hr2t_with_limit(h, rel, current_filter_threshold + 1))
            if len(candidate_entities) > current_filter_threshold:
                break

        if len(candidate_entities) > current_filter_threshold:
            filter_flag = True
            break
    
    if filter_flag:
        return None

    prefix_list = []
    for rel in path:
        prefix = ",".join(prefix_list)
        prefix_list.append(rel)
        
        data_row = []
        data_row.append(question)
        data_row.append(rel)

        neg_rels = set()
        for h in topic_entities:
            neg_rels |= set(kg.get_relation(h, limit=100))
            if len(neg_rels) > 100:
                break
        neg_rels = list(neg_rels)
        neg_rels.append(END_REL)
        neg_rels = [r for r in neg_rels if r not in pos_rels[prefix]]

        if len(neg_rels) > 0:
            sample_rels = []
            while len(sample_rels) < neg_num:
                sample_rels.extend(neg_rels)
            
            neg_rels = random.sample(sample_rels, neg_num)            
            neg_rels = neg_rels[:neg_num]
            
            data_row.extend(neg_rels)
            new_data_list.append(data_row)
        
        # update for next step
        if rel != END_REL:
            next_question = question + f" {rel} #"
            question = next_question
            topic_entities = kg.deduce_leaves_from_src_list_and_relation(topic_entities, rel)

    return new_data_list


def run_negative_sampling():
    load_data_path = cfg.preprocessing["step3"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step3"]["dump_data_path"]
    folder_path = cfg.preprocessing["step3"]["dump_data_folder"]
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    kg = KnowledgeGraphFreebase()
    threshold = 0.5
    
    data_list = load_jsonl(load_data_path)
    update_paths = {}

    new_data_list = []
    timeout_count = 0
    for json_obj in tqdm(data_list, desc="negative-sampling"):
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
                
        for path_json_obj in path_and_score_list:
            try:
                data = generate_data_list(path_json_obj, json_obj, pos_rels, kg)
            except FunctionTimedOut:
                continue
            if data is not None:
                new_data_list.extend(data)

    print("timeout_count:", timeout_count)
    
    new_data_list = np.array(new_data_list)
    df = pd.DataFrame(new_data_list)
    df.to_csv(dump_data_path, header=False, index=False)
