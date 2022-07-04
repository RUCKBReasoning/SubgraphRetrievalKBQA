# 生成relation集合对应的Embedding

# %%
import argparse
import json
import os
import torch

from typing import Tuple, List, Any, Dict

from utils import dump_jsonl, load_jsonl, load_dict
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from loguru import logger


END_REL = "END OF HOP"

retrieval_model_ckpt = '../tmp/model_ckpt/SimBERT-CWQ'
device = 'cuda:1'

print("[load model begin]")

tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
relation_list, relation2id = load_dict(
    '../tmp/relations.txt')
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)

print("[load model end]")


@torch.no_grad()
def get_texts_embeddings(texts):
    inputs = tokenizer(texts, padding=True,
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, output_hidden_states=True,
                       return_dict=True).pooler_output
    return embeddings


# %%
relation_list.append(END_REL)
relation2id[END_REL] = -1
# %%
emb_list = []
batch_size = 100
for i in range(0, len(relation_list), batch_size):
    part_list = relation_list[i:i+batch_size]
    emb = get_texts_embeddings(part_list)
    emb_list.append(emb)

# %%
relation_emb = torch.concat(emb_list)
# %%
relation_emb.shape
# %%
relation_emb = relation_emb.to('cpu')
# %%
torch.save(relation_emb, '../tmp/CWQ_int/relation_emb.tensor')
# %%
