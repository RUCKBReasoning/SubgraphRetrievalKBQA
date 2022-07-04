# 测试数据集的加载时间
# %%
import pickle
import re
with open('ent2id.pickle','rb') as f:
    ent2id = pickle.load(f)
# %%
import json
dataset = []
f = open('util/CWQ_step0_test_new.json')
for line in f:
    dataset.append(json.loads(line))
f.close()
# %%
len(dataset)
# %%
dataset[0]['answers']
# %%
cnt = 0
for item in dataset:
    answer_kb_list = [i['kb_id'] for i in item['answers']]
    if not answer_kb_list:
        cnt += 1
        continue
    if any((answer_kb in ent2id for answer_kb in answer_kb_list)):
        cnt += 1
cnt / len(dataset)

# %%
def load_jsonl(filename):
    dataset = []
    f = open(filename)
    for line in f:
        dataset.append(json.loads(line))
    f.close()
    return dataset
# %%
dataset = load_jsonl('util/CWQ_step0.json') + load_jsonl('util/CWQ_step0_test_new.json')
# %%
len(dataset)
# %%
# 新定义一下 cache_id 算了
from tqdm import tqdm 
bad_question = []
for item in tqdm(dataset):
    answers = item['answers']
    answers = [i['kb_id'] for i in answers]
    entities = item['entities']
    entities = [i['kb_id'] for i in entities]
    item['answers_cid'] = [ent2id.get(a,-1) for a in answers]
    item['entities_cid'] = [ent2id.get(e, -1) for e in entities]
    if -1 in item['answers_cid'] or -1 in item['entities_cid']:
        bad_question.append(item)
        print(item)
# %%
# Only a small number question are not covered, ignore it
f = open('CWQ_full_with_int_id.json','w')
for item in dataset:
    outline = json.dumps(item)
    print(outline, file=f)
f.close()

# %%
from util.deal_cvt import load_cvt
# %%
cvt_map = load_cvt()
# %%
from util.deal_cvt import is_cvt
# %%
for k in ent2id:
    break
# %%
is_cvt(k,cvt_map)
# %%
def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    return False
# %%
for _, k in zip(range(100),ent2id):
    continue
# %%
k
# %%
is_ent(k)
# %%
is_cvt(k,cvt_map)
# %%
# 我们将实体切分为三种类型, "值","普通实体","CVT实体",
# 我们将上述实体分别定为 1, 2, 3
import numpy as np
# %%
ent_type_ary = np.zeros(len(ent2id))
# %%
def judge_type(key):
    if is_ent(key):
        if is_cvt(key,cvt_map):
            return 3
        else:
            return 2
    else:
        return 1
# %%
judge_type(k)
# %%
for ent_key, ent_id in tqdm(ent2id.items()):
    ent_type = judge_type(ent_key)
    ent_type_ary[ent_id] = ent_type
# %%
ent_type_ary = ent_type_ary.astype(np.int8)
# %%
np.bincount(ent_type_ary)
# %%
with open('ent_type_ary.npy','wb') as f:
    np.save(f,ent_type_ary)
# %%
