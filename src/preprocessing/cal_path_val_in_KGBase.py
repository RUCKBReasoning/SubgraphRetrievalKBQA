from copy import deepcopy
import numpy as np
from tqdm import tqdm
import json
from utils import load_jsonl, dump_jsonl
from knowledgeGraphBase import KnowledgeGraphBase
kg = KnowledgeGraphBase('subgraph_2hop_triple.npy', 'ent_type_ary.npy')


# %%

q_with_path_datalist = load_jsonl('cwq_all_relation_path.jsonl')
# %%
q_with_path_and_val_datalist = []
for item in tqdm(q_with_path_datalist):
    ent_and_path = item['ent_and_path']
    answers = item['answers_cid']
    ent, path_list = ent_and_path['ent'], ent_and_path['path']
    path_and_val_list = []
    for p in path_list:
        leaves = kg.deduce_node_leaves_by_path(ent, p)
        if not leaves.size:
            continue
        common = np.intersect1d(answers, leaves)
        val = common.size / leaves.size
        path_and_val_list.append([p, val])
    new_item = deepcopy(item)
    new_item.pop('ent_and_path')
    new_item['ent'] = ent
    new_item['path_and_val_list'] = path_and_val_list
    q_with_path_and_val_datalist.append(new_item)

# %%
dump_jsonl(q_with_path_and_val_datalist, 'cwq_all_relation_path_and_val.jsonl')
# %%
