# %%
'''对结果进行重新的负采样'''
import sys

sys.path.append('..')
from utils import load_jsonl
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from copy import deepcopy
import argparse
import json
import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut

parser = argparse.ArgumentParser()
parser.add_argument("--dump_data_path", type=str, required=True)
args = parser.parse_args()

END_REL = "END OF HOP"

# kg = KnowledgeGraphFreebase()
# kg.deduce_relations_from_src_list()
# %%
infile_name = './search_tree_list.pkl'
with open(infile_name, 'rb') as f:
    search_tree_list = pickle.load(f)
# %%
# search_item = search_tree_list[88]

def preprocess_item(search_item, negative_num=15):
    question = search_item['question']
    # topic_entity = search_item['topic_entities'][0]  # Assume only one
    data_list = []
    for history in search_item['search_tree']:
        all_path = search_item['search_tree'][history]
        pos_rels, neg_rels = all_path['pos'], all_path['neg']
        query = question + ' [SEP] ' + ' # '.join(history)
        for pos_rel in pos_rels:
            if neg_rels:
                local_neg_list = random.choices(tuple(neg_rels), k=negative_num)
                data_list.append([query, pos_rel]+local_neg_list)
    return data_list

# %%
all_data_list = []
for search_item in search_tree_list:
    data_list = preprocess_item(search_item)
    all_data_list.extend(data_list)
# %%
import pandas as pd
# %%
df = pd.DataFrame(all_data_list)
# %%
# dump_data_path = './e2e_train_data.csv'
dump_data_path = os.path.join(args.dump_data_path, 'e2e_train_data.csv')
df.to_csv(dump_data_path, header=False, index=False)
