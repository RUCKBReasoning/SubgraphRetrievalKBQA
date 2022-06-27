# %%
from copy import deepcopy
import json
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from util.ppr_util import personalized_pagerank
from sklearn.preprocessing import normalize
import pickle
import time


def load_npy(filename):
    with open(filename, 'rb') as f:
        return np.load(f)


MAX_CNT = 10**8

ENTITY_TYPE_FILE = 'ent_type_ary.npy'
TRIPLE_FILE = 'subgraph_2hop_triple.npy'
ENT_TYPE, CVT_TYPE = 2, 3

triple = load_npy(TRIPLE_FILE)
entity_type = load_npy(ENTITY_TYPE_FILE)
E = triple.shape[0]
csr_mat = csr_matrix((np.ones(E), (triple[:, 0], np.arange(E)))).astype('bool')
tail_csr_mat = csr_matrix(
    (np.ones(E), (triple[:, 2], np.arange(E)))).astype('bool')
bin_lookup = np.zeros(MAX_CNT, dtype=np.int32)
rel_csr_mat = csr_matrix(
    (np.ones(E), (triple[:, 1], np.arange(E)))).astype('bool')
# %%


def convert_edge_index_to_norm_graph(row, col):
    # edge_index shape is [2, E]
    node_unique = np.unique(np.concatenate([row, col]))
    bin_lookup[node_unique] = np.arange(node_unique.shape[0])
    size = node_unique.shape[0]
    u = bin_lookup[row]
    v = bin_lookup[col]
    return size, node_unique, u, v
# %%


def filter_cvt_nodes(seed_ary):
    seed_type = entity_type[seed_ary]
    return seed_ary[seed_type == CVT_TYPE]


def filter_ent_nodes(seed_ary):
    seed_type = entity_type[seed_ary]
    return seed_ary[seed_type == ENT_TYPE]


def fetch_triple(src_seed):
    indices = csr_mat[src_seed].indices
    return triple[indices]
# %%


def fetch_triple_with_tail(tgt_seed):
    indices = tail_csr_mat[tgt_seed].indices
    return triple[indices]


def search_one_hop(ent_seed):
    first_triple = fetch_triple(ent_seed)
    cvt_nodes = filter_cvt_nodes(np.unique(first_triple[:, 2]))
    second_triple = fetch_triple(cvt_nodes)
    return np.concatenate([first_triple, second_triple], axis=0)


def search_two_hop(ent_seed):
    one_hop_triple = search_one_hop(ent_seed)
    one_hop_nodes = np.unique(one_hop_triple[:, 2])
    one_hop_ents = filter_ent_nodes(one_hop_nodes)
    new_ent_seed = np.setdiff1d(one_hop_ents, ent_seed)
    two_hop_triple = search_one_hop(new_ent_seed)
    return np.concatenate([one_hop_triple, two_hop_triple], axis=0)


def search_all_path(src, tgt):
    paths = []
    # One Hop
    first_triple = fetch_triple(src)
    first_cvt_nodes = filter_cvt_nodes(np.unique(first_triple[:, 2]))
    first_cvt_triple = fetch_triple(first_cvt_nodes)
    first_cvt_heads = {}
    first_hop_tail = {}
    for i in range(first_triple.shape[0]):
        if first_triple[i][2] == tgt:
            paths.append([first_triple[i]])
            continue
        if entity_type[first_triple[i][2]] == CVT_TYPE:
            if first_triple[i][2] not in first_cvt_heads:
                first_cvt_heads[first_triple[i][2]] = []
            first_cvt_heads[first_triple[i][2]].append(first_triple[i])
        else:
            if first_triple[i][2] not in first_hop_tail:
                first_hop_tail[first_triple[i][2]] = []
            first_hop_tail[first_triple[i][2]].append([first_triple[i]])
    for i in range(first_cvt_triple.shape[0]):
        # print(first_cvt_triple[i])
        if first_cvt_triple[i][2] == tgt:
            paths.extend([[j, first_cvt_triple[i]]
                         for j in first_cvt_heads[first_cvt_triple[i][0]]])
            continue
        if first_cvt_triple[i][2] not in first_hop_tail:
            first_hop_tail[first_cvt_triple[i][2]] = []
        tmp = [[j, first_cvt_triple[i]]
               for j in first_cvt_heads[first_cvt_triple[i][0]]]
        first_hop_tail[first_cvt_triple[i][2]].extend(tmp)
    # Two Hop
    second_triple = fetch_triple_with_tail(tgt)
    second_cvt_nodes = filter_cvt_nodes(np.unique(second_triple[:, 0]))
    second_cvt_triple = fetch_triple_with_tail(second_cvt_nodes)
    second_cvt_tails = {}
    for i in range(second_triple.shape[0]):
        if second_triple[i][0] in first_hop_tail:
            paths.extend([j+[second_triple[i]]
                         for j in first_hop_tail[second_triple[i][0]]])
            continue
        if entity_type[second_triple[i][0]] == CVT_TYPE:
            if second_triple[i][0] not in second_cvt_tails:
                second_cvt_tails[second_triple[i][0]] = []
            second_cvt_tails[second_triple[i][0]].append([second_triple[i]])
    for i in range(second_cvt_triple.shape[0]):
        if second_cvt_triple[i][0] in first_hop_tail:
            for k in first_hop_tail[second_cvt_triple[i][0]]:
                paths.extend(
                    [k+[second_cvt_triple[i]]+j for j in second_cvt_tails[second_cvt_triple[i][2]]])
    return [[j.tolist() for j in i] for i in paths]