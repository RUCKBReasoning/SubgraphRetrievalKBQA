import argparse
import os
import json
import torch

from typing import Tuple, List, Any, Dict

from yaml import dump
from utils import dump_jsonl, load_jsonl
from tqdm import tqdm
from knowledge_graph import KonwledgeGraph
from transformers import AutoModel, AutoTokenizer

from loguru import logger
logger.remove(handler_id=None)
logger.add("log.txt")

_min_score = 100.0

END_REL = "END OF HOP"

knowledge_graph_ckpt = '../tmp/knowledge_graph.kg_data'
retrieval_model_ckpt = '../tmp/model_ckpt/SimBERT'
device = 'cuda'

print("[load model begin]")

kg = KonwledgeGraph.load_from_ckpt(knowledge_graph_ckpt)
tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)

print("[load model end]")

def path_to_subgraph(topic_entity: str, path: List[str]):
    """输入topic_entity, path, 得到对应的实例化子图——节点集合、三元组集合"""
    return kg.deduce_subgraph_by_path(topic_entity, path)

def path_to_leaves(topic_entity: str, path: List[str]):
    return kg.deduce_leaves_by_path(topic_entity, path, no_hop_flag=END_REL)

def path_to_candidate_relations(topic_entity: str, path: List[str]) -> List[str]:
    """输入topic_entity, 得到叶子结点能提供的候选relation的集合"""
    candidate_relations = set()
    nodes = kg.deduce_leaves_by_path(topic_entity, path)
    for node in nodes:
        new_relations = kg.get_relation(node)
        candidate_relations = candidate_relations | set(new_relations)
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
    all_relation_list = list(set(sum(relation_list_list, [])))
    q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)
    target_emb = get_texts_embeddings(all_relation_list).unsqueeze(0)
    sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2) / theta
    for i, (path, path_score, relation_list) in enumerate(zip(path_list, path_score_list, relation_list_list)):
        for relation in relation_list:
            j = all_relation_list.index(relation)
            new_path = path + [relation]
            score = float(sim_score[i, j]) + path_score
            results.append((new_path, score))
    return results


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
        for path, path_score in candidate_paths:
            path_list.append(path)
            path_score_list.append(path_score)
            candidate_relations = path_to_candidate_relations(
                topic_entity, path) + [END_REL]
            relation_list_list.append(candidate_relations)
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


# TODO
def merge_paths(paths: List[Tuple[List[str], float]]):
    pass

scores = []

def retrieve_subgraph(json_obj: Dict[str, Any], entity2id, relation2id, entities, rels):
    question = json_obj["question"]
    if len(json_obj["entities"]) == 0:
        return None, None
    
    answers = set([ans_obj["kb_id"] for ans_obj in json_obj["answers"]])
    
    topic_entity = entities[json_obj["entities"][0]]
    path_score_list = infer_paths_from_kb(question, topic_entity, 10, 10, 3)
    
    new_obj = {
        "pos_paths": [],
        "neg_paths": [],
    }
    
    for path, score in path_score_list:
        partial_nodes = path_to_leaves(topic_entity, path)
        
        if len(answers) / len(partial_nodes) >= 0.5 and len(answers & set(partial_nodes)) > 0:
            new_obj["pos_paths"].append(path)
            scores.append((1, score))
            logger.info("pos score: {}".format(score))
        else:
            new_obj["neg_paths"].append(path)
            logger.info("neg score: {}".format(score))
    
    return question, new_obj
    

def build_id_map(load_data_path):
    entities = []
    rels = []
    with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
        for line in f.readlines():
            entities.append(line.strip())
    entity2id = {}
    for word in entities:
        entity2id[word] = len(entity2id)
    relation2id = {}
    for entity in kg.head2relation:
        if entity not in entity2id:
            entities.append(entity)
            entity2id[entity] = len(entity2id)
        for rel in kg.head2relation[entity]:
            if rel not in relation2id:
                rels.append(rel)
                relation2id[rel] = len(relation2id)
    return entity2id, relation2id, entities, rels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data_path", type=str, required=True)
    args = parser.parse_args()
    
    
    load_data_path = args.load_data_path
        
    train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
    test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))    
    dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple.json"))    

    entity2id, relation2id, entities, rels = build_id_map(load_data_path)
    
    update_paths = {}
    for json_obj in tqdm(train_dataset, desc="retrieve:train"):
        question, path_obj = retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)
        if question is not None:
            update_paths[question] = path_obj
    
    for json_obj in tqdm(test_dataset, desc="retrieve:test"):
        question, path_obj = retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)
        if question is not None:
            update_paths[question] = path_obj

    for json_obj in tqdm(dev_dataset, desc="retrieve:dev"):
        question, path_obj = retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)
        if question is not None:
            update_paths[question] = path_obj

    # with open("../tmp/retriever/update_paths.json", "w") as f:
    #     f.write(json.dumps(update_paths) + "\n")

    # print("scores:", scores)
    print("min_score:", min(scores))
