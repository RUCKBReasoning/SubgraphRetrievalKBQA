'''统计Reader给每个路径的打分,最终结果将输出到某个文件中'''
from typing import Dict, List, Any
import copy
import os.path
import json
import sys
import os


def load_jsonl(path: str):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


load_data_path = 'finetune_data_of_cwq'
train_dataset = load_jsonl(os.path.join(
    load_data_path, "webqsp_nsm_test.info"))


train_with_val_list = []
for train_item in train_dataset:
    val = 0
    dist_item = train_item['dist']
    answers = train_item['original']['answers_cid']
    for answer in answers:
        answer_kb_id = str(answer)
        val += dist_item.get(answer_kb_id, 0)
    new_item = copy.copy(train_item['original'])
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
