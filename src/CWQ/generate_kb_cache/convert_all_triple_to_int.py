# 将全图上的字符串节点统一映射 Int 来

# %%
from matplotlib import spines
from tqdm import tqdm
f = open('/data/zhang2718/freebase_cache/cache/subgraph_hop2.txt')
entity_set = set()
relation_set = set()
for line in tqdm(f,total=200000000):
        spline = line.strip().split("\t")
        entity_set.add(spline[0])
        entity_set.add(spline[2])
        relation_set.add(spline[1])
# %%
f.close()
# %%
relation_list = sorted(relation_set)
with open('relations.txt','w') as f:
    for relation in relation_list:
        print(relation, file=f)
# %%
relation2id = {relation:idx for idx,relation in enumerate(relation_list)}
# %%
ent2id = {ent:idx for idx,ent in enumerate(entity_set)}
# %%
import pickle
# %%
with open('ent2id.pickle','wb') as f:
    pickle.dump(ent2id, f)
# %%
triple_list = []
f = open('/data/zhang2718/freebase_cache/cache/subgraph_hop2.txt')
for line in tqdm(f,total=200000000):
        spline = line.strip().split("\t")
        h, r, t = spline
        h, r, t = ent2id[h], relation2id[r], ent2id[t]
        triple_list.append([h, r, t])

# %%
import numpy as np
# %%
sp_g = np.array(triple_list)
# %%
sp_g.shape
# %%
sp_g.dtype
# %%
sp_g = sp_g.astype(np.int32)
# %%
with open('subgraph_2hop_triple.npy','wb') as f:
    np.save(f, sp_g)
# %%
