import argparse
import json
import os
import torch
import subprocess

from typing import Tuple, List, Any, Dict, Set

from utils import dump_jsonl, load_jsonl
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut
from transformers import AutoModel, AutoTokenizer

from loguru import logger

# from knowledge_graph.knowledge_graph import KnowledgeGraph
# from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
# from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from knowledge_graph.knowledge_graph_base import KnowledgeGraphBase
from config import cfg

END_REL = "END OF HOP"

TOP_K = 10

_min_score = 1e5

retrieval_model_ckpt = cfg.retriever_model_ckpt
device = 'cuda'

print("[load model begin]")

# kg = KnowledgeGraphCache()
kg = KnowledgeGraphBase(triple='./tmp/subgraph_2hop_triple.npy',
                        ent_type='./tmp/ent_type_ary.npy')
tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)

print("[load model end]")

def path_to_subgraph(topic_entity: str, path: List[str]):
    """输入topic_entity, path, 得到对应的实例化子图——节点集合、三元组集合"""
    return kg.deduce_subgraph_by_path(topic_entity, path)

def path_to_candidate_relations(topic_entity: str, path: List[str]) -> List[str]:
    """输入topic_entity, 得到叶子结点能提供的候选relation的集合"""
    new_relations = kg.deduce_leaves_relation_by_path(topic_entity, path)
    # filter relation
    candidate_relations = [r for r in new_relations if r.split(".")[0] not in ["kg", "common"]]
    
    # limit = 10
    # candidate_relations = [r for r in candidate_relations if 
    #                        kg.deduce_leaves_count_by_path(topic_entity, path + [r]) <= limit ** (len(path) + 1)]
    
    return list(candidate_relations)

@torch.no_grad()
def get_texts_embeddings(texts):
    inputs = tokenizer(texts, padding=True,
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, output_hidden_states=True,
                       return_dict=True).pooler_output
    return embeddings


def score_path_list_and_relation_list(question: str, path_list: List[List[str]], path_score_list: List[float], relation_list_list: List[List[str]], theta: float = 0.07) -> List[Tuple[List[str], float]]:
    """计算path和其对应的候选的relation的得分"""
    results = []
    
    query_lined_list = ['#'.join([question] + path) for path in path_list]
    
    # SR w/o QU # TMP
    # query_lined_list = [question for _ in path_list]
    
    all_relation_list = list(set(sum(relation_list_list, [])))
    # END_relation_index = all_relation_list.index(END_REL)
    q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)  # [B, 1, D]
    target_emb = get_texts_embeddings(all_relation_list).unsqueeze(0)  # [1, L, D]
    sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2) / theta  # [B, L]
    # end_score = sim_score[:, END_relation_index].unsqueeze(1)  # [B, 1]
    # sigmoid_score = torch.sigmoid((sim_score))  # 1 / (1 + exp(-(f(r) - f(end))))
    for i, (path, path_score, relation_list) in enumerate(zip(path_list, path_score_list, relation_list_list)):
        for relation in relation_list:
            j = all_relation_list.index(relation)
            new_path = path + [relation]
            score = float(sim_score[i, j]) + path_score
            results.append((new_path, score))
    return results

@torch.no_grad()
def infer_paths_from_kb(question: str, topic_entity: str, num_beams: int, num_return_paths: int, max_hop: int) -> List[Tuple[List[str], float]]:
    """从KB中进行宽为num_beams的搜索,得到num_return_paths条路径并提供对应得分"""
    candidate_paths = [[[], 0]]  # path and its score
    result_paths = []
    n_hop = 0
    while candidate_paths and len(result_paths) < num_return_paths and n_hop < max_hop:
        search_list = []
        # try every possible next_hop
        relation_list_list = []
        path_list = []
        path_score_list = []
        # logger.info(f'candidate_paths: {candidate_paths}')
        for path, path_score in candidate_paths:
            path_list.append(path)
            path_score_list.append(path_score)
            # logger.info(f'path_to_candidate_relations: {topic_entity}, {path}')
            candidate_relations = path_to_candidate_relations(
                topic_entity, path)
            candidate_relations = candidate_relations + [END_REL]
            relation_list_list.append(candidate_relations)
        # logger.info(f'path_list and relation_list_list:{path_list} {relation_list_list}')
        search_list = score_path_list_and_relation_list(
            question, path_list, path_score_list, relation_list_list)

        search_list = sorted(search_list, key=lambda x: x[1], reverse=True)[
            :num_beams]
        # Update candidate_paths and result_paths
        n_hop = n_hop + 1
        candidate_paths = []
        for path, score in search_list:
            if path[-1] == END_REL:
                result_paths.append([path, score])
            else:
                candidate_paths.append([path, score])
    # Force early stop
    if n_hop == max_hop and candidate_paths:
        for path, score in candidate_paths:
            path = path + [END_REL]
            result_paths.append([path, score])
    result_paths = sorted(result_paths, key=lambda x: x[1], reverse=True)[
        :num_return_paths]
    return result_paths


def _reverse_graph(G:Dict[str, List[str]]):
    r_G:Dict[str,List[str]] = dict()
    for u in G:
        for v in G[u]:
            r_G.setdefault(v, []).append(u)
    return r_G

def bfs_graph(G:Dict[str, List[str]],root):
    """
    G: a adjacency list in Dict
    return: all bfs nodes
    """
    visited = set()
    currentLevel = [root]
    while currentLevel:
        for v in currentLevel:
            visited.add(v)
        nextLevel = set()
        # levelGraph = {v:set() for v in currentLevel}
        for v in currentLevel:
            for w in G.get(v,[]):
                if w not in visited:
                    # levelGraph[v].add(w)
                    nextLevel.add(w)
        # yield levelGraph
        currentLevel = nextLevel
    return visited


def merge_graph(graph_l, root_l, graph_r, root_r):
    assert root_l != root_r
    all_nodes = set()
    common_nodes = set(graph_l) & set(graph_r)
    all_nodes |= common_nodes
    reverse_graph_l, reverse_graph_r = _reverse_graph(graph_l), _reverse_graph(graph_r)
    for node in common_nodes:
        ancestors_l = bfs_graph(reverse_graph_l, node)
        ancestors_r = bfs_graph(reverse_graph_r, node)
        descendants_l = bfs_graph(graph_l, node)
        descendants_r = bfs_graph(graph_r, node)
        all_nodes.update(ancestors_l)
        all_nodes.update(ancestors_r)
        all_nodes.update(descendants_l)
        all_nodes.update(descendants_r)
    return all_nodes


def filter_by_graph(nodes: List[str], triples: List[str], G: Dict[str, List[str]]):
    entities = set(G.keys())
    nodes = [e for e in nodes if e in entities]
    triples = [(h, r, t) for h, r, t in triples if h in entities and t in entities]
    return nodes, triples


def build_graph(nodes: List[str], triples: List[str]):
    G = {}
    for e in nodes:
        G[e] = []
    for h, _, t in triples:
        G.setdefault(h, []).append(t)
    return G


def retrieve_subgraph(json_obj: Dict[str, Any], entities: Set, rels: Set):
    data_list = []

    question = json_obj["question"]
    if len(json_obj["entities"]) == 0:
        return
    
    answers = set([ans_obj["kb_id"] for ans_obj in json_obj["answers"]])
        
    for topic_entity in json_obj["entities"]:

        path_score_list = infer_paths_from_kb(question, topic_entity, TOP_K, TOP_K, 2)
        nodes = []
        triples = []

        min_score = 1e5
    
        threshold_ent_size = 1000
        for path, score in path_score_list:
            partial_nodes, partial_triples = path_to_subgraph(topic_entity, path)
            if len(partial_nodes) > 1000:
                continue
            
            partial_nodes = [e for e in partial_nodes if e in entities]
            partial_triples = [(h, r, t) for h, r, t in partial_triples 
                               if h in entities and t in entities and r in rels]
            
            nodes.extend(partial_nodes)
            triples.extend(partial_triples)
            
            data_list.append({
                "id": json_obj["id"],
                "question": json_obj["question"],
                "entities": json_obj["entities"],
                "answers": json_obj["answers"],
                "paths": [(topic_entity, path)],
                "subgraph": {
                    "entities": partial_nodes,
                    "tuples": partial_triples
                }
            })
            
            if len(answers & set(partial_nodes)) > 0:
                min_score = min(min_score, score)
            if len(nodes) > threshold_ent_size:
                break
    return data_list


def build_entities(load_data_path):
    entities = []
    with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
        for line in f.readlines():
            entities.append(line.strip())
    return entities

def build_relations(load_data_path):
    rels = []
    with open(os.path.join(load_data_path, "relations.txt"), "r") as f:
        for line in f.readlines():
            rels.append(line.strip())
    return rels


def run():
    load_data_folder = cfg.retriever_finetune["load_data_path"]
    
    train_dataset = load_jsonl(os.path.join(load_data_folder, "train_simple.json"))

    entities = build_entities(load_data_folder)
    rels = build_relations(load_data_folder)
    
    entities = set(entities)
    rels = set(rels)
    
    new_dataset = []
    for json_obj in tqdm(train_dataset, desc="retrieve:train"):
        data_list = retrieve_subgraph(json_obj, entities, rels)
        new_dataset.extend(data_list)
    
    dump_jsonl(new_dataset, os.path.join(load_data_folder, "finetune_simple.json"))
