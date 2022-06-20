from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy


class KonwledgeGraphFreebase:

    def __init__(self) -> None:        
        self.sparql = SPARQLWrapper("http://10.77.110.128:3001/sparql")
        self.sparql.setReturnFormat(JSON)

    def get_relation(self, entity: str) -> List[str]:
        entity = ':'+entity
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {
                """
                +entity + '?r0_ ?t0 . '
                # '?h0 ' + '?r0 ?t0. '
                                            """
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?t0, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                }
                    """)
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        # rtn = []
        # for result in results['results']['bindings']:
            # rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))
        return [i['r0']['value'] for i in results['results']['bindings'] if i['r0']['value']!='type.object.type']

    def get_tail(self, src: str, relation) -> List[str]:
        src = ':'+src
        relation = ':'+relation
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?t0 WHERE {
                """
                +src + ' ' + relation + ' ?t0_ . '
                # '?h0 ' + '?r0 ?t0. '
                                            """
                    FILTER regex(?t0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?t0_),str(:)) as ?t0)
                }
                    """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        # rtn = []
        # for result in results['results']['bindings']:
            # rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))
        return [i['t0']['value'] for i in results['results']['bindings']]


    def get_single_tail_relation_triplet(self, src):
        # occured_set = set()
        # multiple_set = set()
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r ?t0 WHERE {
                """
                +src + ' ?r_ ?t0_ . '
                # '?h0 ' + '?r0 ?t0. '
                                            """
                    FILTER regex(?r_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?t0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r_),str(:)) as ?r)
                    bind(strafter(str(?t0_),str(:)) as ?t0)
                    }
                    """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        # rtn = []
        # for result in results['results']['bindings']:
            # rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))
        cnt = {}
        for i in results['results']['bindings']:
            if i['r']['value']=='type.object.type':
                continue
            if i['r']['value'] not in cnt:
                cnt[i['r']['value']] = 0
            if cnt[i['r']['value']]> 1:
                continue
            cnt[i['r']['value']]+=1
        return [k for k,v in cnt.items() if v==1]

    def get_all_path(self, src_, tgt_):
        src = src_
        tgt = tgt_
        src = ':'+src
        tgt = ':'+tgt
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {
                """
                +src + ' ?r0_ ' + tgt + ' . '
                                            """
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                    }

                    """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        one_hop = [[(src_,i['r0']['value'],tgt_)] for i in results['results']['bindings'] if i['r0']['value']!='type.object.type']
        single_hop_relations = self.get_single_tail_relation_triplet(src)
        two_hop = []
        for r0 in list(single_hop_relations):
            query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?e0 ?r1 WHERE {
                    """
                    +src + ' :' + r0 +' ?e0_ . \n'+
                    '?e0_ ?r1_ ' + tgt + ' . '
                                                """
                    FILTER regex(?r1_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?e0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r1_),str(:)) as ?r1)
                    bind(strafter(str(?e0_),str(:)) as ?e0)
                        }
                        """)
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError:
                print(query)
                exit(0)
            two_hop += [[(src_,r0,i['e0']['value']),(i['e0']['value'],i['r1']['value'],tgt_)] for i in results['results']['bindings']
                            if i['r1']['value']!='type.object.type']
        return one_hop + two_hop

    def get_shortest_path_limit(self, src_, tgt_):
        src = src_
        tgt = tgt_
        src = ':'+src
        tgt = ':'+tgt
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 WHERE {
                """
                +src + ' ?r0_ ' + tgt + ' . '
                                            """
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                    }

                    """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        one_hop = [[(src_,i['r0']['value'],tgt_)] for i in results['results']['bindings'] if i['r0']['value']!='type.object.type']
        if len(one_hop)>0:
            return one_hop
        single_hop_relations = self.get_single_tail_relation_triplet(src)
        two_hop = []
        for r0 in list(single_hop_relations):
            query = ("""
                    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?e0 ?r1 WHERE {
                    """
                    +src + ' :' + r0 +' ?e0_ . \n'+
                    '?e0_ ?r1_ ' + tgt + ' . '
                                                """
                    FILTER regex(?r1_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?e0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r1_),str(:)) as ?r1)
                    bind(strafter(str(?e0_),str(:)) as ?e0)
                        }
                        """)
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError:
                print(query)
                exit(0)
            two_hop += [[(src_,r0,i['e0']['value']),(i['e0']['value'],i['r1']['value'],tgt_)] for i in results['results']['bindings']
                            if i['r1']['value']!='type.object.type']
        return two_hop

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
                for tail in self.get_tail(node, relation):
                    next_hop_set.add(tail)
                    triples.add((node, relation, tail))
            hop_nodes = deepcopy(next_hop_set)
            nodes = nodes | hop_nodes
        return list(nodes), list(triples)

    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[str]:
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
