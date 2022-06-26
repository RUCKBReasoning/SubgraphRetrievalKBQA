import argparse
from ast import keyword
import json
import os
import torch
import networkx as nx

from typing import Tuple, List, Any, Dict

from utils import load_jsonl
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut
from transformers import AutoModel, AutoTokenizer

from knowledge_graph.knowledge_graph import KnowledgeGraph as KnowledgeGraphMemory
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache
from config import cfg

from loguru import logger

END_REL = "END OF HOP"

TOP_K = 1

retrieval_model_ckpt = cfg.retriever_model_ckpt
device = 'cuda'

print("[load model begin]")

tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
model = AutoModel.from_pretrained(retrieval_model_ckpt)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print("[load model end]")


def get_texts_embeddings(texts):
    inputs = tokenizer(texts, padding=True,
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, output_hidden_states=True,
                       return_dict=True).pooler_output
    return embeddings


def get_path_pair(batch_question: List[str], batch_path_list: List[List[str]]):
    # 得到一个N*M的sentence map
        
    N, M = len(batch_question), max([len(path) for path in batch_path_list])
    query_list_list, rel_list_list, mask_list = [], [], []
    for question, path in zip(batch_question, batch_path_list):
        query_list = []
        rel_list = []
        mask = []
        query = question + ' [SEP] '
        for rel in path:
            query_list.append(query)
            rel_list.append(rel)
            mask.append(1)
            query = query + ' # ' + rel

        while len(query_list) < M:
            query_list.append('')
            rel_list.append('')
            mask.append(0)

        query_list_list.append(query_list)
        rel_list_list.append(rel_list)
        mask_list.append(mask)
        
        # print(len(query_list), len(mask), M)

    query_list = sum(query_list_list, [])
    tgt_list = sum(rel_list_list, [])
    mask_list = sum(mask_list, [])
    
    return N, M, query_list, tgt_list, mask_list


def finetune():

    load_data_path = cfg.retriever_finetune["load_data_path"]
    dump_model_path = cfg.retriever_finetune["dump_model_path"]
    
    train_dataset = load_jsonl(os.path.join(load_data_path, "finetune_simple.json"))
    dist_info = load_jsonl(os.path.join(load_data_path, "dist.info"))

    batch_size = 8
    n = len(train_dataset)
    batch_index = list(range(0, n, batch_size))
    for index in tqdm(batch_index, total=len(batch_index)):
        offset = min(n, index + batch_size) - index
        batch_json_obj = train_dataset[index:index + offset]
        batch_dist = dist_info[index:index + offset]
        batch_question = [json_obj["question"] for json_obj in batch_json_obj]
        batch_path_list = [json_obj["paths"][0][1] for json_obj in batch_json_obj]
        
        weights = []
        for json_obj, dist_json_obj in zip(batch_json_obj, batch_dist):
            answers = [x["kb_id"] for x in json_obj["answers"]]
            entities = json_obj["subgraph"]["entities"]
            
            if set(answers) & set(entities) == 0:
                for key in dist_json_obj.keys():
                    dist_json_obj[key] = 0.0
        
            answers_score = []
            for e in answers:
                answers_score.append(dist_json_obj[e] if e in dist_json_obj else 0.0)
            weight = max(answers_score + [0.])
            weights.append(weight)

        N, M, query_list, tgt_list, mask_list = get_path_pair(batch_question, batch_path_list)
        
        query_emb = get_texts_embeddings(query_list).view(N, M, -1)
        target_emb = get_texts_embeddings(tgt_list).view(N, M, -1)
        item_score = torch.cosine_similarity(query_emb, target_emb, dim=-1)
        mask = torch.tensor(mask_list).to(item_score.device).view(N, M)
        
        labels = torch.tensor(weights).to(item_score.device).view(N)

        path_score = (item_score * mask).sum(-1) / mask.sum(-1)
        kld_loss = torch.nn.KLDivLoss()
        loss: torch.Tensor = kld_loss(path_score, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save model
    torch.save(model.state_dict(), os.path.join(dump_model_path, "pytorch_model_finetune.bin"))
