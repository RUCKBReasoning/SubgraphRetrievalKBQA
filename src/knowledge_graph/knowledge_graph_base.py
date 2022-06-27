# 将原版的 KnowledgeGraph 当作外部 Wrapper, 调用内部的 KnowledgeGraphBase


'''Provide API of KG'''

from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
VERY_LARGT_NUM = 10**8
PATH_CUTOFF = 10**6
NODE_CUTOFF = 10**4


class KnowledgeGraphBase(object):
    def __init__(self, triple: Union[np.ndarray, str], ent_type: np.ndarray):
        if isinstance(triple, str):
            triple = self._load_npy_file(triple)
        if isinstance(ent_type, str):
            ent_type = self._load_npy_file(ent_type)
        self.triple: np.ndarray = triple
        self.ent_type: np.ndarray = ent_type
        self.bin_map: np.ndarray = np.zeros_like(ent_type, dtype=np.int32)
        E = self.triple.shape[0]
        self.head2fact = csr_matrix(
            (np.ones(E), (triple[:, 0], np.arange(E)))).astype('bool')
        self.rel2fact = csr_matrix(
            (np.ones(E), (triple[:, 1], np.arange(E)))).astype('bool')
        self.tail2fact = csr_matrix(
            (np.ones(E), (triple[:, 2], np.arange(E)))).astype('bool')

    @staticmethod
    def _load_npy_file(filename):
        with open(filename, 'rb') as f:
            return np.load(f)

    @staticmethod
    def path_join(lhs, rhs, path_cutoff=PATH_CUTOFF):
        lhs_length = lhs.shape[1]
        rhs_length = rhs.shape[1]
        df_lhs = pd.DataFrame(lhs, columns=[str(i) for i in range(lhs_length)])
        df_rhs = pd.DataFrame(rhs, columns=[str(i) for i in range(
            lhs_length-1, lhs_length+rhs_length-1)])
        paths = pd.merge(df_lhs, df_rhs, on=str(lhs_length-1)).to_numpy()
        return paths[:path_cutoff]

    def _fetch_forward_triple(self, seed_set):
        indices = self.head2fact[seed_set].indices
        return self.triple[indices]

    def _fetch_backward_triple(self, seed_set):
        indices = self.tail2fact[seed_set].indices
        return self.triple[indices]

    def get_relation(self, head_set):
        triple = self._fetch_forward_triple(head_set)
        return np.unique(triple[:, 1])

    def get_tail(self, head_set, relation):
        triple = self._fetch_forward_triple(head_set)
        rel_indices = (triple[:, 1] == relation)
        return triple[rel_indices]

    def filter_cvt_nodes(self, seed_ary, CVT_TYPE=3):
        seed_type = self.ent_type[seed_ary]
        return seed_ary[seed_type == CVT_TYPE]

    def get_shortest_path_length(self, src_set, tgt_set):
        '''计算从src_set到tgt_set的最短路的长度'''
        # One Hop forward and backward
        forward_triple_one = self._fetch_forward_triple(src_set)
        backward_triple_one = self._fetch_backward_triple(tgt_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        backward_node_one = np.unique(backward_triple_one[:, 0])
        if np.intersect1d(forward_node_one, tgt_set).size > 0:
            return 1
        if np.intersect1d(src_set, backward_node_one).size > 0:
            return 1
        if np.intersect1d(forward_node_one, backward_node_one).size > 0:
            return 2

        # Two Hop forward and backward
        forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        forward_triple_two = self._fetch_forward_triple(forward_cvt_node)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        backward_triple_two = self._fetch_backward_triple(backward_cvt_node)
        backward_node_two = np.unique(backward_triple_two[:, 0])

        if np.intersect1d(forward_node_two, tgt_set).size > 0:
            return 2
        if np.intersect1d(src_set, backward_node_two).size > 0:
            return 2
        if np.intersect1d(forward_node_two, backward_node_one).size > 0:
            return 3
        if np.intersect1d(forward_node_one, backward_node_two).size > 0:
            return 3
        if np.intersect1d(forward_node_two, backward_node_two).size > 0:
            return 4
        return -1

    def get_all_path_with_length_limit(self, src_set, tgt_set, limit):
        assert 1 <= limit <= 4
        path_list = []
        forward_triple_one = self._fetch_forward_triple(src_set)
        backward_triple_one = self._fetch_backward_triple(tgt_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        backward_node_one = np.unique(backward_triple_one[:, 0])
        forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        forward_triple_two = self._fetch_forward_triple(forward_cvt_node)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        backward_triple_two = self._fetch_backward_triple(backward_cvt_node)
        backward_node_two = np.unique(backward_triple_two[:, 0])

        src_node = np.array(src_set, dtype=np.int32).reshape(-1, 1)
        tgt_node = np.array(tgt_set, dtype=np.int32).reshape(-1, 1)
        if limit >= 1:
            local_path = []
            if np.intersect1d(forward_node_one, tgt_set).size > 0:
                path_s_t = self.path_join(forward_triple_one, tgt_node)
                path_list.append(
                    {'length': 1, 'path': path_s_t})
        if limit >= 2:
            if np.intersect1d(forward_node_one, backward_node_one).size > 0:
                path_s_m_t = self.path_join(
                    forward_triple_one, backward_triple_one)
                path_list.append(
                    {'length': 2, 'path': path_s_m_t})
        if limit >= 3:
            local_path = []
            if np.intersect1d(forward_node_two, backward_node_one).size > 0:
                common = np.intersect1d(forward_node_two, backward_node_one)
                back_for_common_triple = self._fetch_backward_triple(common)
                path_s_c_e = self.path_join(
                    forward_triple_one, back_for_common_triple)
                path_s_c_e_t = self.path_join(path_s_c_e, backward_triple_one)
                local_path.append(path_s_c_e_t)
            if np.intersect1d(forward_node_one, backward_node_two).size > 0:
                common = np.intersect1d(forward_node_one, backward_node_two)
                forw_for_common_triple = self._fetch_forward_triple(common)
                path_e_c_t = self.path_join(
                    forw_for_common_triple, backward_triple_one)
                path_s_e_c_t = self.path_join(forward_triple_one, path_e_c_t)
                local_path.append(path_s_e_c_t)
            if local_path:
                path_list.append(
                    {'length': 3, 'path': np.concatenate(local_path)})
        if limit >= 4:
            if np.intersect1d(forward_node_two, backward_node_two).size > 0:
                common = np.intersect1d(forward_node_two, backward_node_two)
                back_for_common_triple = self._fetch_backward_triple(common)
                forw_for_common_triple = self._fetch_forward_triple(common)
                path_e_c_t = self.path_join(
                    forw_for_common_triple, backward_triple_one)
                path_s_c_e = self.path_join(
                    forward_triple_one, back_for_common_triple)
                path_e_c_e_c_t = self.path_join(path_e_c_t, path_s_c_e)
                path_list.append({'length': 4, 'path': path_e_c_e_c_t})
        return path_list

    def deduce_subgraph_by_path(self, head, path):
        '''是否保留?'''
        # TODO: Implement this by performing small modifications to the method `deduce_node_leaves_by_path`
        pass

    def deduce_node_leaves_by_path(self, src, path):
        seed_set = src
        for p in path:
            seed_triple = self.get_tail(seed_set, p)
            seed_set = np.unique(seed_triple[:, 2])[:NODE_CUTOFF]
            if not seed_set.size:
                break
        return seed_set
    # TODO: Use this method to replace, and make sure add an 
    #       id2relationLabel convertion before input to this method
    def deduce_relation_leaves_by_path(self, src, path):
        node_leaves = self.deduce_node_leaves_by_path(src, path)
        return self.get_relation(node_leaves)
