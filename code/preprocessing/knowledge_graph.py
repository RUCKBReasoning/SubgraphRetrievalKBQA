'''Provide API of KG'''

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
import pickle
from copy import deepcopy


class KonwledgeGraph(object):
    def __init__(self,
                 G: nx.DiGraph,
                 head2relation: Dict[str, Tuple[str]],
                 head_relation_2_tail: Dict[str, Dict[str, List[str]]]
                 ):
        self.G = G
        self.head2relation = head2relation
        self.head_relation_2_tail = head_relation_2_tail

    @classmethod
    def load_from_ckpt(cls, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            data_dict = pickle.load(f)
        return KonwledgeGraph(data_dict['G'], data_dict['head2relation'], data_dict['head_relation_2_tail'])

    @classmethod
    def load_from_triples(cls, triples: List[Tuple[str, str, str]]):
        G = nx.DiGraph()
        head2relation = defaultdict(set)
        head_relation_2_tail = defaultdict(lambda: defaultdict(list))
        for h, r, t in triples:
            G.add_edge(h, t, keyword=r)
            head2relation[h].add(r)
            head_relation_2_tail[h][r].append(t)
        head2relation = {k: tuple(v) for k, v in head2relation.items()}
        head_relation_2_tail = dict(head_relation_2_tail)
        return KonwledgeGraph(G, head2relation, head_relation_2_tail)

    def dump_to_ckpt(self, ckpt_path):
        data_dict = dict(G=self.G, head2relation=self.head2relation,
                         head_relation_2_tail=self.head_relation_2_tail)
        with open(ckpt_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def get_relation(self, h):
        return self.head2relation.get(h, tuple())

    def get_tail(self, src, relation):
        return self.head_relation_2_tail.get(src, dict()).get(relation, list())

    def get_all_path(self, src, tgt, cutoff: int = 3):
        paths = nx.all_simple_paths(self.G, src, tgt, cutoff=cutoff)
        return paths

    def get_shorted_path_limit(self, src, tgt):
        return nx.shortest_path_length(self.G, src, tgt)

    def deduce_subgraph_by_path(self, src: str, path: List[str], no_hop_flag: str = 'NoHop') -> Tuple[List[str], List[Tuple[str, str, str]]]:
        # 将子图实例化, 返回节点集合和边集合
        nodes, triples = set(), set()
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        nodes.add(src)
        for relation in path:
            next_hop_set = set()
            if relation == no_hop_flag:
                continue
            for node in hop_nodes:
                for tail in self.get_tail(node, relation):
                    next_hop_set.add(tail)
                    triples.add((node, relation, tail))
            hop_nodes = deepcopy(next_hop_set)
            nodes = nodes | hop_nodes
        return list(nodes), list(triples)

    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'NoHop') -> Tuple[str]:
        # 效率瓶颈，有待优化
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        for relation in path:
            if relation == no_hop_flag:
                continue
            next_hop_set = set()
            for node in hop_nodes:
                for tail in self.get_tail(node, relation):
                    next_hop_set.add(tail)
            hop_nodes = deepcopy(next_hop_set)
        return tuple(hop_nodes)
