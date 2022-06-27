# 将 test_set 的 entities 映射到 m. g. 等 kb_id

# %%
import json
dataset = []
f = open('util/CWQ_step0_test.json')
for line in f:
    dataset.append(json.loads(line))
f.close()
# %%


def load_dict(filename):
    id2word, word2id = list(), dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
            id2word.append(word)
    return id2word, word2id

# %%
id2ent, ent2id = load_dict('util/entities_ppr_cache.txt')
# %%
dataset[0]
# %%
for item in dataset:
    entities = item['entities']
    kb_id_list = [id2ent[eid] for eid in entities]
    ent_list = [{"kb_id":kb_id, "text":kb_id} for kb_id in kb_id_list]
    item['entities'] = ent_list
    answers = item['answers']
    ans_list = []
    for answer in answers:
        kb_id:str = answer['kb_id']
        if kb_id.startswith(':'):
            kb_id = kb_id.removeprefix(':')
        answer['kb_id'] = kb_id
        ans_list.append(answer)
    item['answers'] = ans_list

# %%
f = open('util/CWQ_step0_test_new.json','w')
for item in dataset:
    outline = json.dumps(item)
    print(outline, file=f)
f.close()
# %%
