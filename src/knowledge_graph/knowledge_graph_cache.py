'''A Cached KB'''
from typing import Dict, List, Tuple, Set
from copy import deepcopy


class KnowledgeGraphCache(object):

    def __init__(self, kb_filename = 'tmp/subgraph_hop1.txt') -> None:

        self.triples: Dict[str, Tuple[str, str, str]
                           ] = self._load_triples(kb_filename)
        self.head2relation: Dict[str, Set[str]
                                 ] = self._build_head2relation(self.triples)
        self.head_relation2tail: Dict[str, Dict[str, Set[str]]
                                      ] = self._build_head_relation2tail(self.triples)

    def _load_triples(self, kb_filename) -> Dict[str, Tuple[str, str, str]]:
        f = open(kb_filename)
        triples = {}
        for line in f:
            head, rel, tail = line.strip().split("\t")
            triples.setdefault(head, set())
            triples[head].add((head, rel, tail))
        f.close()
        return triples

    def _build_head2relation(self, triples) -> Dict[str, Set[str]]:
        head2relation = {}
        for entity in triples:
            head2relation[entity] = {r for h, r, t in triples[entity]}
        return head2relation

    def _build_head_relation2tail(self, triples) -> Dict[str, Set[str]]:
        head_relation2tail = {}
        for entity in triples:
            head_relation2tail.setdefault(entity, dict())
            for _, r, t in triples[entity]:
                head_relation2tail[entity].setdefault(r, set())
                head_relation2tail[entity][r].add(t)
        return head_relation2tail

    def get_relation(self, h) -> Set[str]:
        return self.head2relation.get(h, set())

    def get_tail(self, src, relation) -> Set[str]:
        return self.head_relation2tail.get(src, dict()).get(relation, set())

    def deduce_subgraph_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[List[str], List[Tuple[str, str, str]]]:
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
                tail_set = self.get_tail(node, relation)
                next_hop_set |= tail_set
                triples |= {(node, relation, tail) for tail in tail_set}
            hop_nodes = deepcopy(next_hop_set)
            nodes = nodes | hop_nodes
        return list(nodes), list(triples)

    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Set[str]:
        # 效率瓶颈，有待优化
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        for relation in path:
            if relation == no_hop_flag:
                continue
            next_hop_set = set()
            for node in hop_nodes:
                next_hop_set |= self.get_tail(node, relation)
            hop_nodes = deepcopy(next_hop_set)
        return hop_nodes

    def deduce_leaves_relation_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Set[str]:
        relations = set()
        leaves = self.deduce_leaves_by_path(src, path, no_hop_flag)
        relations = relations.union(*(self.get_relation(leave) for leave in leaves))
        return relations
