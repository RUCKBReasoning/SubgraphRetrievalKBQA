''' Generate all candidate relations for each path's step relation '''
# %%
from tqdm import tqdm
from torch import threshold
from utils import load_jsonl, dump_jsonl
from knowledgeGraphBase import KnowledgeGraphBase
kg = KnowledgeGraphBase('subgraph_2hop_triple.npy', 'ent_type_ary.npy')
# %%
dataset = load_jsonl('cwq_all_relation_path_and_val.jsonl')
# %%
item = dataset[0]
# %%
kg.deduce_relation_leaves_by_path(item['ent'], [25647, 25682])
# %%


def get_path_val_limit(path_and_val_list):
    '''分类判断,如果有不小于0.5的path, 选择0.5; 如果'''
    val_list = [v for p, v in path_and_val_list]
    max_val = max(val_list)
    if max_val >= 0.5:
        threshold = 0.5
    elif max_val >= 0.2:
        threshold = max_val - 0.1
    else:
        threshold = 100  # No result
    return threshold


# %%
all_search_state_list = []
for item in tqdm(dataset):
    q = item['question']
    ent = item['ent']
    path_and_val_list = item['path_and_val_list']
    if not path_and_val_list:
        continue
    val_limit = get_path_val_limit(path_and_val_list)
    for path, val in path_and_val_list:
        if val < val_limit:
            continue
        history = []
        for rel in path:
            candidates = kg.deduce_relation_leaves_by_path(
                ent, history).tolist()
            state = dict(question=q, ent=ent, positive_rel=rel, history=tuple(history),
                         candidates=candidates)
            history.append(rel)
            all_search_state_list.append(state)
        final_candidates = kg.deduce_relation_leaves_by_path(
            ent, path).tolist()
        state = dict(question=q, ent=ent, positive_rel='END OF HOP', history=tuple(path),
                     candidates=candidates)
        all_search_state_list.append(state)

# %%
dump_jsonl(all_search_state_list, 'cwq_full_with_search_state.jsonl')
# %%
