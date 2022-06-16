"""
获取简单路径所对应的 relation 路径, 并计算该 relation 路径的得分
relation 路径的得分定义为, 使用该路径实例化得到的预测结果相比较答案的 HIT 得分
"""
import os
from copy import deepcopy
import glob
from tqdm import tqdm

from knowledge_graph import KonwledgeGraph
from utils import load_jsonl, dump_jsonl

def run_path_to_relation():

    knowledge_graph_ckpt = '../tmp/knowledge_graph.kg_data'
    kg = KonwledgeGraph.load_from_ckpt(knowledge_graph_ckpt)
    G = kg.G

    all_jsonl = glob.glob('../tmp/preprocessing/*')
    data_list = sum([load_jsonl(filepath) for filepath in all_jsonl], [])

    data_with_path_list = []
    for (item, paths) in tqdm(data_list):
        m = set()
        for path in paths:
            relation_list = []
            for edge in path:
                r = G.get_edge_data(*edge)['keyword']
                relation_list.append(r)
            m.add('\t'.join(relation_list))
        data_with_path_list.append((item, tuple(m)))


    def cal_path_val(topic_entity, path, answers):
        preds = kg.deduce_leaves_by_path(topic_entity, path)
        preds = set(preds)
        hit = preds & answers
        full = preds
        if not full:
            return 1
        return len(hit) / len(full)

    m_list = []
    for item, p_strs in tqdm(data_with_path_list):
        answers = set(item['answers'])
        topic_entities = item['topic_entities']
        path_and_score_list = []
        for p_str in p_strs:
            path = p_str.split('\t')
            p_val_list = []
            for topic_entity in topic_entities:
                p_val_list.append(cal_path_val(topic_entity, path, answers))
            p_val = max(p_val_list)
            path_and_score_list.append(dict(path=path, score=p_val))
        m_item = deepcopy(item)
        m_item['path_and_score_list'] = path_and_score_list
        m_list.append(m_item)

    if not os.path.exists("../tmp/retriever"):
        os.makedirs("../tmp/retriever")
    
    dump_jsonl(m_list, '../tmp/retriever/train_with_path_score.jsonl')
