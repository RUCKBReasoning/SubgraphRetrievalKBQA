import argparse
import os
import torch

from typing import Tuple, List, Any, Dict

from yaml import dump
from utils import dump_jsonl, load_jsonl
from tqdm import tqdm
from knowledge_graph import KonwledgeGraph
from transformers import AutoModel, AutoTokenizer

from loguru import logger

logger.remove(handler_id=None)
logger.add("retr.log")

END_REL = "END OF HOP"

_min_score = 1e5

knowledge_graph_ckpt = '../tmp/knowledge_graph.kg_data'
retrieval_model_ckpt = '../tmp/model_ckpt/SimBERT'
# retrieval_model_ckpt = '../tmp/model_ckpt/SimBERT_backup'
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


def retrieve_subgraph(json_obj: Dict[str, Any], entity2id, relation2id, entities, rels):
    logger.info("[sample]")

    question = json_obj["question"]
    if len(json_obj["entities"]) == 0:
        return
    
    answers = set([ans_obj["kb_id"] for ans_obj in json_obj["answers"]])
    
    topic_entity = entities[json_obj["entities"][0]]
    path_score_list = infer_paths_from_kb(question, topic_entity, 10, 10, 3)
    nodes = []
    triples = []

    min_score = 1e5
    
    threshold_ent_size = 100
    for path, score in path_score_list:
        partial_nodes, partial_triples = path_to_subgraph(topic_entity, path)
        if len(partial_nodes) > 1000:
            continue
        nodes.extend(partial_nodes)
        triples.extend(partial_triples)
        
        if len(answers & set(partial_nodes)) > 0:
            min_score = min(min_score, score)
            logger.info("[pos path] score: {}".format(score))
        else:
            logger.info("[neg path] score: {}".format(score))          
        if len(nodes) > threshold_ent_size:
            break

    global _min_score
    _min_score = min(_min_score, min_score)
    
    nodes = list(set(nodes))
    triples = list(set(triples))
    subgraph_entities = [e for e in nodes if e in entity2id]
    subgraph_tuples = [(h, relation2id[r], t) for h, r, t in triples 
                       if h in entity2id and t in entity2id and r in relation2id]
    json_obj["entities"] = [entities[e] for e in json_obj["entities"]]
    json_obj["subgraph"] = {
        "tuples": subgraph_tuples,
        "entities": subgraph_entities
    }

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
    parser.add_argument("--dump_data_path", type=str, required=True)
    args = parser.parse_args()
    
    
    load_data_path = args.load_data_path
    dump_data_path = args.dump_data_path
    
    if not os.path.exists(dump_data_path):
        os.makedirs(dump_data_path)
    
    train_dataset = load_jsonl(os.path.join(load_data_path, "train_simple.json"))
    test_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))    
    dev_dataset = load_jsonl(os.path.join(load_data_path, "dev_simple.json"))    

    entity2id, relation2id, entities, rels = build_id_map(load_data_path)

    for dataset in [train_dataset, test_dataset, dev_dataset]:
        count = 0
        for obj in train_dataset:
            for ans_obj in obj["answers"]:
                if ans_obj["kb_id"] not in entity2id:
                    count += 1
        print(count)

    with open(os.path.join(dump_data_path, "entities.txt"), "w") as f:
        for word in entities:
            f.write(word + "\n")
    
    with open(os.path.join(dump_data_path, "relations.txt"), "w") as f:
        for word in rels:
            f.write(word + "\n")
    for json_obj in tqdm(train_dataset, desc="retrieve:train"):
        retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)
    
    for json_obj in tqdm(test_dataset, desc="retrieve:test"):
        retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)

    for json_obj in tqdm(dev_dataset, desc="retrieve:dev"):
        retrieve_subgraph(json_obj, entity2id, relation2id, entities, rels)

    dump_jsonl(train_dataset, os.path.join(dump_data_path, "train_simple.json"))
    dump_jsonl(test_dataset, os.path.join(dump_data_path, "test_simple.json"))
    dump_jsonl(dev_dataset, os.path.join(dump_data_path, "dev_simple.json"))

    print("min score:", _min_score)
