'''统计Reader给每个路径的打分,最终结果将输出到某个文件中'''
import argparse
from ast import arg
import os
import json
import os.path
import copy
from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--load_data_path", type=str, required=True)
args = parser.parse_args()


# load_data_path = 'finetune_data'
load_data_path = args.load_data_path
train_dataset = load_jsonl(os.path.join(
    load_data_path, "finetune_simple.json"))
dist_info = load_jsonl(os.path.join(load_data_path, "dist.info"))

train_with_val_list = []
for train_item, dist_item in zip(train_dataset, dist_info):
    val = 0
    for answer in train_item['answers']:
        answer_kb_id = answer['kb_id']
        val += dist_item.get(answer_kb_id, 0)
    new_item = copy.copy(train_item)
    new_item.pop('subgraph')
    new_item['topic_entity'], new_item['path'] = new_item['paths'][0][0], new_item['paths'][0][1]
    new_item.pop('paths')
    new_item['path_val'] = val
    train_with_val_list.append(new_item)

m_dict = dict()
for item in train_with_val_list:
    question = item['question']
    path = item['path']
    p_val = item['path_val']
    topic_entity = new_item['topic_entity']
    m_dict.setdefault(question, {'topic_entities': [], 'path': []})
    if topic_entity not in m_dict[question]['topic_entities']:
        m_dict[question]['topic_entities'].append(topic_entity)
    m_dict[question]['path'].append((path, p_val))

with open('m_dict.json', 'w') as f:
    json.dump(m_dict, f)
