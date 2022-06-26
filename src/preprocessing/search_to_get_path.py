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
from knowledge_graph.knowledge_graph import KnowledgeGraph
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from config import cfg


@func_set_timeout(10)
def generate_paths(item, kg: KnowledgeGraphFreebase, pair_max: int = 20, path_max: int = 100):
    paths = []
    entities = [entity for entity in item['topic_entities']]
    answers = [answer for answer in item['answers']]
    for src in entities:
        for tgt in answers:
            if len(paths) > path_max:
                break
            n_paths = []
            n_paths.extend(kg.search_one_hop_relaiotn(src, tgt))
            n_paths.extend(kg.search_two_hop_relation(src, tgt))
            paths.extend(n_paths)
    return paths[:path_max]


def run_search_to_get_path():
    load_data_path = cfg.preprocessing["step1"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step1"]["dump_data_path"]
    kg = KnowledgeGraphFreebase()
    train_dataset = load_jsonl(load_data_path)

    outf = open(dump_data_path, 'w')
    for item in tqdm(train_dataset):
        try:
            paths = generate_paths(item, kg)
        except FunctionTimedOut:
            continue
        outline = json.dumps([item, paths], ensure_ascii=False)
        print(outline, file=outf)
        outf.flush()
    outf.close()
