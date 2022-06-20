"""
枚举头实体到答案的所有简单路径，（限制这些路径长度不超过最短路径长度+1)
每个进程将结果写入自己对应的文件中
"""
import multiprocessing
import time
import math
import networkx as nx
import os
import json
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut

from utils import load_jsonl
from knowledge_graph import KonwledgeGraph
from knowledge_graph_freebase import KonwledgeGraphFreebase

def generate_paths(item, kg: KonwledgeGraphFreebase, pair_max: int = 20, path_max: int = 100):
    paths = []
    entities = [entity for entity in item['topic_entities']]
    answers = [answer for answer in item['answers']]
    for src in entities:
        for tgt in answers:
            if len(paths) > path_max:
                break
            n_paths = kg.get_all_path(src, tgt)
            paths.extend(n_paths)
    return paths[:path_max]


def run_sequential(kg, item_list):

    filename = '3hop_datalist.jsonl'
    
    if not os.path.exists("../tmp/3hop"):
        os.makedirs("../tmp/3hop")

    filepath = os.path.join("../tmp/3hop", filename)
    outf = open(filepath, 'w')
    for item in tqdm(item_list):
        paths = generate_paths(item, kg)
        outline = json.dumps([item, paths], ensure_ascii=False)
        print(outline, file=outf)
        outf.flush()
    outf.close()


def run_search_to_get_path():
    kg = KonwledgeGraphFreebase()
    train_dataset = load_jsonl('../tmp/retriever/train.jsonl')    
    run_sequential(kg, train_dataset)

run_search_to_get_path()
