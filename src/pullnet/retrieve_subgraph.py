import argparse
import os
from numpy import dtype
import torch
import networkx

from typing import Tuple, List, Any, Dict

from tqdm import tqdm
from knowledge_graph_freebase import KonwledgeGraphFreebase
from func_timeout import func_set_timeout, FunctionTimedOut
from transformers import AutoModel, AutoTokenizer

from utils import load_jsonl, dump_jsonl
from loguru import logger

END_REL = "END OF HOP"

knowledge_graph_ckpt = '../tmp/knowledge_graph.kg_data'
retrieval_model_ckpt = '../tmp/model_ckpt/SimBERT'
device = 'cuda'

print("[load model begin]")

# kg = KonwledgeGraph.load_from_ckpt(knowledge_graph_ckpt)
kg = KonwledgeGraphFreebase()
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


def score_path_list_and_relation_list(question: str, entity_list: List[str], path_list: List[List[str]], path_score_list: List[float], relation_list_list: List[List[str]], theta: float = 0.07) -> List[Tuple[List[str], float]]:
    """计算path和其对应的候选的relation的得分"""
    results = []
    query_lined_list = ['#'.join([question] + path) for path in path_list]
    all_relation_list = list(set(sum(relation_list_list, [])))
    # END_relation_index = all_relation_list.index(END_REL)
    q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)  # [B, 1, D]
    target_emb = get_texts_embeddings(all_relation_list).unsqueeze(0)  # [1, L, D]
    sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2) / theta  # [B, L]
    # end_score = sim_score[:, END_relation_index].unsqueeze(1)  # [B, 1]
    # sigmoid_score = torch.sigmoid((sim_score))  # 1 / (1 + exp(-(f(r) - f(end))))
    for i, (entity, path, path_score, relation_list) in enumerate(zip(entity_list, path_list, path_score_list, relation_list_list)):
        for relation in relation_list:
            j = all_relation_list.index(relation)
            score = float(sim_score[i, j]) + path_score
            results.append((entity, relation, score))
    return results


def retrieve_subgraph(
    G: networkx.DiGraph, 
    topic_entity: str, 
    src_entities: List[str], 
    question: str, 
    relation2id: Dict[str, int]
):
    top_k = 20
    
    entity_list = []
    path_list = []
    path_score_list = []
    relation_list = []
    for entity in src_entities:
        paths = list(networkx.all_simple_edge_paths(G, topic_entity, entity))
        if topic_entity == entity:
            paths.append([])
        if len(paths) == 0:
            continue
        entity_list.append(entity)
        path_list.append(paths[0])
        path_score_list.append(0.)
        relation_list.append(kg.get_relation(entity))
        
    if len(entity_list) == 0:
        return [], []
    
    
    results = score_path_list_and_relation_list(question, entity_list, path_list, path_score_list, relation_list)
    results = sorted(results, key=lambda T: T[2], reverse=True)[:top_k]
    new_entities = []
    new_tuples = []
    for h, r, _ in results:
        if r not in relation2id:
            continue
        t_list = kg.get_hr2t_with_limit(h, r, limit=top_k)
        for t in t_list:
            new_entities.append(t)
            new_tuples.append((h, relation2id[r], t))
    return new_entities, new_tuples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    return args


def run(args):
    relation2id = {}
    with open(os.path.join(args.data_folder, "relations.txt")) as f:
        for line in f.readlines():
            line = line.strip()
            relation2id[line] = len(relation2id)
    
    if args.hop == 0:
        test_dataset = load_jsonl(os.path.join(args.data_folder, "test_simple.json"))
        for json_obj in test_dataset:
            entities = json_obj["entities"]
            json_obj["subgraph"] = {
                "tuples": [],
                "entities": entities
            }
        for json_obj in tqdm(test_dataset):
            topic_entity = json_obj["entities"][0]
            tuples = json_obj["subgraph"]["tuples"]
            entities = json_obj["subgraph"]["entities"]
            G = networkx.DiGraph()
            for e in entities:
                G.add_node(e)
            for h, r, t in tuples:
                G.add_edge(h, t, keyword=r)
            src_entities = [topic_entity]

            new_entities, new_tuples = retrieve_subgraph(
                G, topic_entity, src_entities, json_obj["question"], relation2id)
            
            entities = list(set(entities) | set(new_entities))
            tuples = list(set(tuples) | set(new_tuples))
            
            json_obj["subgraph"] = {
                "tuples": tuples,
                "entities": entities
            }
        dump_jsonl(test_dataset, os.path.join(args.data_folder, "test_simple.json"))
    else:
        test_dataset = load_jsonl(os.path.join(args.data_folder, "test_simple.json"))
        dist_dataset = load_jsonl(os.path.join(args.data_folder, "dist.info"))
        for json_obj, dist_dict in tqdm(zip(test_dataset, dist_dataset), total=len(test_dataset)):
            topic_entity = json_obj["entities"][0]
            tuples = json_obj["subgraph"]["tuples"]
            entities = json_obj["subgraph"]["entities"]
            G = networkx.DiGraph()
            for e in entities:
                G.add_node(e)
            for h, r, t in tuples:
                G.add_edge(h, t, keyword=r)
            src_entities = [k for k, v in dist_dict.items()]

            new_entities, new_tuples = retrieve_subgraph(
                G, topic_entity, src_entities, json_obj["question"], relation2id)
            
            entities = list(set(entities) | set(new_entities))
            tuples = list(set(tuples) | set(new_tuples))
            
            json_obj["subgraph"] = {
                "tuples": tuples,
                "entities": entities
            }
            
        dump_jsonl(test_dataset, os.path.join(args.data_folder, "test_simple.json"))

if __name__ == '__main__':
    run(parse_args())
