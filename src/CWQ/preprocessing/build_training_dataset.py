''' Trans to retriever raw data '''

# %%
from numpy import positive
from torch import negative
from utils import load_dict, load_jsonl, dump_jsonl
import random
# %%
NEGATIVE_SAMPLE_NUM = 15
# %%
cwq_list = load_jsonl('./tmp/cwq_full_with_search_state.jsonl')
# example_item = {"question": "What state is home to the university that is represented in sports by George Washington Colonials men's basketball?", "ent": 34939943, "positive_rel": 25630,
#                 "history": [], "candidates": [3791, 3793, 3794, 6963, 6964, 6965, 7896, 7897, 7899, 7900, 7901, 8379, 9390, 25565, 25566, 25630, 25634, 25643, 25645, 25647, 26154, 26155]}
# %%
relation_list, relation2id = load_dict('./tmp/relations.txt')
relation_list.append('END OF HOP')
relation2id['END OF HOP'] = -1
# %%


def question_templeate(question, idx_path):
    question = question.lower() + ' [SEP] '
    path = [relation_list[idx] for idx in idx_path]
    suffix = ' # '.join(path)
    return question + suffix


# %%
out_list = []
for item in cwq_list:
    query = question_templeate(item['question'], item['history'])
    positive_idx = - \
        1 if isinstance(item['positive_rel'], str) else int(item['positive_rel'])
    negative_idx_set = set(item['candidates'])
    if positive_idx in negative_idx_set:
        negative_idx_set.remove(positive_idx)
    if positive_idx != -1:
        negative_idx_set.add(-1)
    if not negative_idx_set:
        continue
    negative_idx_list = random.choices(tuple(negative_idx_set),k=NEGATIVE_SAMPLE_NUM)
    positive_relation = relation_list[positive_idx]
    negative_relation_iter = (relation_list[idx] for idx in negative_idx_list)
    out_list.append((query, positive_relation)+tuple(negative_relation_iter))


# %%
dump_jsonl(out_list, './tmp/cwq_full_data_raw_for_retriever.jsonl')
# %%
