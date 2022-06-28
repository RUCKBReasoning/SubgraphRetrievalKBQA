# %%
# from func_timeout import func_set_timeout
# from fast_ppr import search_all_path
from copy import deepcopy
import numpy as np
# %%
import tqdm
import json
from knowledge_graph.knowledge_graph_base import KnowledgeGraphBase

kg = KnowledgeGraphBase('./tmp/subgraph_2hop_triple.npy', './tmp/ent_type_ary.npy')
# %%
# DATASET_IN_NAME = 'webqsp/webqsp_full_with_int_id.jsonl'
DATASET_IN_NAME = './tmp/CWQ_full_with_int_id.json'
dataset = []
f = open(DATASET_IN_NAME)
for line in f:
    dataset.append(json.loads(line))
f.close()
# %%
dis_list = []
from copy import deepcopy
outf = open('./tmp/cwq_all_relation_path.jsonl', 'w')
for item in tqdm.tqdm(dataset):
    ent_seed = item['entities_cid']
    ans_seed = item['answers_cid']
    ent_seed = [ent for ent in ent_seed if ent > -1]
    ans_seed = [ans for ans in ans_seed if ans > -1]
    if not ent_seed or not ans_seed:
        continue
    for ent in ent_seed:
        path_list = None
        dis = kg.get_shortest_path_length(ent, ans_seed)
        if dis == -1:
            continue
        if dis <= 2:
            path_list = kg.get_all_path_with_length_limit(ent, ans_seed, dis+1)
        else:
            path_list = kg.get_all_path_with_length_limit(ent, ans_seed, dis)
        out_list = []
        if path_list:
            for path in path_list:
                p = path['path']
                path_in_rel = p[:, range(1, p.shape[1], 2)]
                path_in_rel = np.unique(path_in_rel, axis=0)
                out_list.extend(path_in_rel.tolist())
        if out_list:
            new_item = deepcopy(item)
            new_item['ent_and_path'] = dict(ent=ent, path=out_list)
            outline = json.dumps(new_item)
            print(outline, file=outf)
outf.close()


# %%
