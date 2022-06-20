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

@func_set_timeout(30)
def generate_paths(item, kg, pair_max: int = 20, path_max: int = 100):
    paths = []
    entities = [entity for entity in item['topic_entities'] if entity in kg.G]
    answers = [answer for answer in item['answers'] if answer in kg.G]
    for src in entities:
        for tgt in answers:
            if len(paths) > path_max:
                break
            if not nx.has_path(kg.G, src, tgt):
                continue
            min_length = nx.shortest_path_length(kg.G, src, tgt)
            if min_length > 2:
                continue
            cutoff = min(2, min_length+1)
            cnt = 0
            # search shortest path first, then search shortest+1
            n_paths = []
            for p in nx.all_simple_edge_paths(kg.G, src, tgt, min_length):
                n_paths.append(p)
                cnt += 1
                if cnt > pair_max:
                    break
            if cutoff > min_length:
                for p in nx.all_simple_edge_paths(kg.G, src, tgt, cutoff):
                    if p in n_paths:
                        continue
                    else:
                        n_paths.append(p)
                        cnt += 1
                    if cnt > pair_max:
                        break
            paths.extend(n_paths)
    return paths[:path_max]


def run_sequential(args):
    seq_id = args["seq_id"]
    item_list = args["item_list"]
    G = args["G"]
    
    timeout_count = 0

    t = time.time()
    print(f"[id: {seq_id}] start time: {t}")
    filename = str(seq_id)+'_datalist.jsonl'
    
    if not os.path.exists("../tmp/preprocessing"):
        os.makedirs("../tmp/preprocessing")

    filepath = os.path.join("../tmp/preprocessing", filename)
    outf = open(filepath, 'w')
    for item in tqdm(item_list):
        try:
            paths = generate_paths(item, G)
        except FunctionTimedOut as e:
            timeout_count += 1
            print("timeout: {}".format(e.msg))
            continue
        outline = json.dumps([item, paths], ensure_ascii=False)
        print(outline, file=outf)
        outf.flush()
    outf.close()
    e = time.time()
    print(f"[id: {seq_id}] time cost: {e-t}")
    print("timeout count: {}".format(timeout_count))

def run_search_to_get_path(args):
    if args.KG_name == "EmbedKGQA":
        kg = KonwledgeGraph.load_from_ckpt("../tmp/knowledge_graph.kg_data")
    else:
        kg = KonwledgeGraphFreebase()
    G = kg.G

    train_dataset = load_jsonl('../tmp/retriever/train.jsonl')

    valid_list = []
    for item in tqdm(train_dataset, desc="generate valid list"):
        entities = [entity for entity in item['topic_entities'] if kg.have_entity(entity)]
        answers = [answer for answer in item['answers'] if kg.have_entity(answer)]
        if any((nx.has_path(G, src, tgt) for src in entities for tgt in answers)):
            valid_list.append(item)
    
    run_sequential({"seq_id": 0, "item_list": valid_list, "G": G})
