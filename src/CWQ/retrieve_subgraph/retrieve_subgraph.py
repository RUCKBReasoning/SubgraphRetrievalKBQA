import argparse
import json
import os
import torch
import subprocess

from typing import Tuple, List, Any, Dict

from utils import dump_jsonl, load_jsonl, load_dict
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
relation_cache_filename = cfg.retrieve_subgraph['relation_cache_filename']
device = 'cuda'

print("[load model begin]")

# kg = KnowledgeGraphCache()
kg = KnowledgeGraphBase(triple='./tmp/subgraph_2hop_triple.npy',
                        ent_type='./tmp/ent_type_ary.npy')
tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
relation_list, relation2id = load_dict('./tmp/relations.txt')
relation_list.append(END_REL)
relation2id[END_REL] = -1
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)
relation_cache = torch.load(relation_cache_filename)
relation_cache = relation_cache.to(device)
print("[load model end]")
type_event_entities = {i for i in range(len(kg.ent_type)) if kg.ent_type[i]==3}
def path_to_subgraph(topic_entity: str, path: List[str]):
    """输入topic_entity, path, 得到对应的实例化子图——节点集合、三元组集合"""
    # TODO: add relation conversions(both conversions)
    path_in_id = [relation2id[rel] for rel in path if rel != END_REL]
    nodes, edge_ary = kg.deduce_subgraph_by_path(topic_entity, path_in_id)
    nodes = nodes.tolist()
    edges = []
    for h, r_id, t in edge_ary:
        rel = relation_list[r_id]
        edges.append((int(h), rel, int(t)))
    return nodes, edges


def path_to_candidate_relations(topic_entity: str, path: List[str]) -> List[str]:
    """输入topic_entity, 得到叶子结点能提供的候选relation的集合"""
    # TODO: change this method; when input, add relationLabel2id, when output, add id2relationLabel
    path_in_id = [relation2id[rel] for rel in path]
    new_relations_in_id = kg.deduce_relation_leaves_by_path(
        topic_entity, path_in_id)
    new_relations = [relation_list[rel_id] for rel_id in new_relations_in_id]
    # filter relation
    candidate_relations = [r for r in new_relations if r.split(".")[0] not in [
        "kg", "common"]]

    # limit = 10
    # candidate_relations = [r for r in candidate_relations if
    #                        kg.deduce_leaves_count_by_path(topic_entity, path + [r]) <= limit ** (len(path) + 1)]

    return list(candidate_relations)


def get_relation_embedding_from_cache(rel_list):
    idxs = [relation2id[rel] for rel in rel_list]
    return relation_cache[idxs]


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
    # TODO Add cache or pre-calculate relation embeding
    q_emb = get_texts_embeddings(query_lined_list).unsqueeze(1)  # [B, 1, D]
    target_emb = get_relation_embedding_from_cache(
        all_relation_list).unsqueeze(0)
    # target_emb = get_texts_embeddings(
    #     all_relation_list).unsqueeze(0)  # [1, L, D]
    sim_score = torch.cosine_similarity(
        q_emb, target_emb, dim=2) / theta  # [B, L]
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
            candidate_relations = path_to_candidate_relations(  # Return paths in STRING; KG-BASE return Int Indexs
                # TODO: Return Int Index back to Relation name/label
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


def _reverse_graph(G: Dict[str, List[str]]):
    r_G: Dict[str, List[str]] = dict()
    for u in G:
        for v in G[u]:
            r_G.setdefault(v, []).append(u)
    return r_G


def bfs_graph(G: Dict[str, List[str]], root):
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
            for w in G.get(v, []):
                if w not in visited:
                    # levelGraph[v].add(w)
                    nextLevel.add(w)
        # yield levelGraph
        currentLevel = nextLevel
    return visited


def merge_graph(graph_l, root_l, graph_r, root_r):
    assert root_l != root_r
    all_nodes = set()
    graph_l_set = set(graph_l)
    graph_r_set = set(graph_r)
    common_nodes = graph_l_set & graph_r_set
    if len(common_nodes)==0:
        return graph_l_set | graph_r_set
    # load all event nodes
    common_nodes |= type_event_entities & (graph_l_set | graph_r_set)
    all_nodes |= common_nodes
    reverse_graph_l, reverse_graph_r = _reverse_graph(
        graph_l), _reverse_graph(graph_r)
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
    triples = [(h, r, t)
               for h, r, t in triples if h in entities and t in entities]
    return nodes, triples


def build_graph(nodes: List[str], triples: List[str]):
    G = {}
    for e in nodes:
        G[e] = []
    for h, _, t in triples:
        G.setdefault(h, []).append(t)
    return G


def retrieve_subgraph(json_obj: Dict[str, Any], entities):
    # logger.info("[sample]")

    question = json_obj["question"]
    # if len(json_obj["entities"]) == 0:
    #     return

    # answers = set([ans_obj["kb_id"] for ans_obj in json_obj["answers"]])
    answers = set(json_obj['answers_cid'])
    if -1 in answers:
        answers.remove(-1)

    paths = []  # List[Tuple[str, List[relation]]]
    graphs = []

    # print("len entities:", len(json_obj["entities"]))
    # logger.info(f'question: {question}')
    # for entity_id in json_obj["entities"]:
    min_score = 1e5
    entities_cid = [c for c in json_obj['entities_cid'] if c > -1]
    if not entities_cid:
        print('No Entity')
    for entity_id in entities_cid:
        # topic_entity = entities[entity_id]
        topic_entity = entity_id
        path_score_list = infer_paths_from_kb(
            question, topic_entity, TOP_K, TOP_K, 4)
        nodes = []
        triples = []
        threshold_ent_size = 1000
        for path, score in path_score_list:
            partial_nodes, partial_triples = path_to_subgraph(
                topic_entity, path)
            if len(partial_nodes) > 1000:
                continue
            paths.append((topic_entity, path))
            nodes.extend(partial_nodes)
            triples.extend(partial_triples)

            if len(answers & set(partial_nodes)) > 0:
                min_score = min(min_score, score)
            if len(nodes) > threshold_ent_size:
                break
        #graphs.append((nodes, triples))
        graphs.append((topic_entity, nodes, triples,
                      build_graph(nodes, triples)))

    n = len(graphs)
    if n < 1:
        return
    for i in range(1, n):
        g0 = graphs[0]
        gi = graphs[i]
        topic_entity = g0[0]
        nodes = merge_graph(g0[3], g0[0], gi[3], gi[0])
        triples = [(h, r, t) for h, r, t in list(
            set(g0[2]) | set(gi[2])) if h in nodes and t in nodes]
        graph = build_graph(nodes, triples)
        graphs[0] = (topic_entity, nodes, triples, graph)
    #nodes = sum([g[0] for g in graphs], [])
    nodes = graphs[0][1]
    #triples = sum([g[1] for g in graphs], [])
    triples = graphs[0][2]

    global _min_score
    _min_score = min(_min_score, min_score)

    nodes = list(set(nodes))
    triples = list(set(triples))
    subgraph_entities = [e for e in nodes]
    subgraph_tuples = [(h, r, t) for h, r, t in triples]
    json_obj["paths"] = paths
    json_obj["entities"] = [e for e in json_obj["entities"]]
    json_obj["subgraph"] = {
        "tuples": subgraph_tuples,
        "entities": subgraph_entities
    }


def build_entities(load_data_path):
    entities = []
    with open(os.path.join(load_data_path, "entities.txt"), "r") as f:
        for line in f.readlines():
            entities.append(line.strip())
    return entities


def run():
    #load_data_folder = cfg.retrieve_subgraph["load_data_folder"]
    dump_data_folder = cfg.retrieve_subgraph["dump_data_folder"]
    load_data_folder = "./tmp/CWQ_int/"
    if not os.path.exists(dump_data_folder):
        os.makedirs(dump_data_folder)

    subprocess.run(["cp", "-r", load_data_folder,
                   os.path.dirname(dump_data_folder)])
    # DEBUG_NUM = 100
    train_dataset = load_jsonl(os.path.join(
        load_data_folder, "train.jsonl"))#[:DEBUG_NUM]
    test_dataset = load_jsonl(os.path.join(
        load_data_folder, "test.jsonl"))#[:DEBUG_NUM]
    dev_dataset = load_jsonl(os.path.join(load_data_folder, "dev.jsonl")) # For this one, we use the full

    # entities = build_entities(load_data_folder)
    entities = None

    for json_obj in tqdm(train_dataset, desc="retrieve:train"):
        retrieve_subgraph(json_obj, entities)

    for json_obj in tqdm(test_dataset, desc="retrieve:test"):
        retrieve_subgraph(json_obj, entities)

    for json_obj in tqdm(dev_dataset, desc="retrieve:dev"):
        retrieve_subgraph(json_obj, entities)

    dump_jsonl(train_dataset, os.path.join(
        dump_data_folder, "train_simple.json"))
    dump_jsonl(test_dataset, os.path.join(
        dump_data_folder, "test_simple.json"))
    dump_jsonl(dev_dataset, os.path.join(dump_data_folder, "dev_simple.json"))

    print("min score:", _min_score)
