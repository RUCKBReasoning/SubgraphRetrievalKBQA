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
                SELECT distinct ?h0 ?r0 WHERE {
                """
                +'?h0_ ?r0_ ' + tgt + ' . '
                                            """
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?h0_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                    bind(strafter(str(?h0_),str(:)) as ?h0)
                    }

                    """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        last_hop = [i for i in results['results']['bindings'] if i['r0']['value']!='type.object.type']
        one_hop = [[(src_,i['r0']['value'],tgt_)] for i in last_hop if i['h0']['value']==src_]
        # two hop
        query = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?r0 ?e0 ?r1 ?e1 WHERE {
                    """+src + """ ?r0_ ?e0_ . \n
                    ?e0_ ?r1_ ?e1_ . 
                    FILTER regex(?r0_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?r1_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?e0_, "http://rdf.freebase.com/ns/")
                    FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                    bind(strafter(str(?r0_),str(:)) as ?r0)
                    bind(strafter(str(?r1_),str(:)) as ?r1)
                    bind(strafter(str(?e0_),str(:)) as ?e0)
                    bind(strafter(str(?e1_),str(:)) as ?e1)
                }
            """)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        hop_twice = [i
                    for i in results['results']['bindings']
                        if i['r0']['value']!='type.object.type' and i['r1']['value']!='type.object.type']
        two_hop = [[(src_,i['r0']['value'],i['e0']['value']),(i['e0']['value'],i['r1']['value'],tgt_)]
                    for i in hop_twice
                        if i['e1']['value']==tgt_]
        third_hop = []
        # third hop
        last_hop_dict = {}
        for i in last_hop:
            if i['h0']['value'] not in last_hop_dict:
                last_hop_dict[i['h0']['value']] = []
            last_hop_dict[i['h0']['value']].append([(i['h0']['value'],i['r0']['value'],tgt_)])
        for i in hop_twice:
            if i['e1']['value'] not in last_hop_dict:
                continue
            tmp = [(src_,i['r0']['value'],i['e0']['value']),(i['e0']['value'],i['r1']['value'],i['e1']['value'])]
            for j in last_hop_dict[i['e1']['value']]:
                third_hop.append(tmp+j)
        return one_hop + two_hop + third_hop

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

    def search_one_hop_relaiotn(self, src:str,tgt:str)->List[Tuple[str]]:
        topic_entity = src
        answer_entity = tgt

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?r1 where {{
                ns:{topic_entity} ?r1_ ns:{answer_entity} . 
                FILTER regex(?r1_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?r1_),str(ns:)) as ?r1)
            }}
        """

        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [(i['r1']['value']) for i in results['results']['bindings']]

    def search_two_hop_relation(self, src, tgt)->List[Tuple[str,str]]:
        topic_entity = src
        answer_entity = tgt

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?r1 ?r2 where {{
                ns:{topic_entity} ?r1_ ?e1 . 
                ?e1 ?r2_ ns:{answer_entity} .
                FILTER regex(?e1, "http://rdf.freebase.com/ns/")
                FILTER regex(?r1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?r2_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?r1_),str(ns:)) as ?r1)
                bind(strafter(str(?r2_),str(ns:)) as ?r2)
            }}
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [(i['r1']['value'],i['r2']['value']) for i in results['results']['bindings']]


    def deduce_subgraph_by_path_one(self, src: str, rels: List[str]):
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?e1 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e1_),str(ns:)) as ?e1)
            }} limit 2000
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        nodes = [i['e1']['value'] for i in results['results']['bindings']] + [src]
        triples = [(src, rels[0], i['e1']['value']) for i in results['results']['bindings']]
        nodes = list(set(nodes))
        triples = list(set(triples))
        return nodes, triples


    def deduce_subgraph_by_path_two(self, src: str, rels: List[str]):
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?e1 ?e2 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                ?e1_ ns:{rels[1]} ?e2_ .
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e2_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e1_),str(ns:)) as ?e1)
                bind(strafter(str(?e2_),str(ns:)) as ?e2)
            }} limit 2000
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        nodes = [i['e1']['value'] for i in results['results']['bindings']] + \
                [i['e2']['value'] for i in results['results']['bindings']] + [src]
        triples = [(src, rels[0], i['e1']['value']) for i in results['results']['bindings']] + \
            [(i['e1']['value'], rels[1], i['e2']['value']) for i in results['results']['bindings']]
        nodes = list(set(nodes))
        triples = list(set(triples))
        return nodes, triples


    def deduce_subgraph_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[List[str], List[Tuple[str, str, str]]]:
        # 将子图实例化, 返回节点集合和边集合
        path = [r for r in path if r != no_hop_flag]
        assert len(path) <= 2
        if len(path) == 0:
            return ([src], [])
        elif len(path) == 1:
            return self.deduce_subgraph_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_subgraph_by_path_two(src, path)            


    def deduce_leaves_by_path_one(self, src:str, rels:str) -> List[Tuple[str]]:
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?e1 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e1_),str(ns:)) as ?e1)
            }}  limit 2000
        """

        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [(i['e1']['value']) for i in results['results']['bindings']]

    def deduce_leaves_by_path_two(self, src:str, rels:Tuple[str,str])->List[Tuple[str,str]]:
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?e2 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                ?e1_ ns:{rels[1]} ?e2_ .
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e2_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e2_),str(ns:)) as ?e2)
            }} limit 2000
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [i['e2']['value'] for i in results['results']['bindings']]

    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[str]:
        # 效率瓶颈，有待优化
        assert len(path) <= 2
        if len(path) == 0:
            return [src]
        elif len(path) == 1:
            return self.deduce_leaves_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_leaves_by_path_two(src, path)
        raise ValueError

    def deduce_leaves_count_by_path_one(self, src:str, rels:str) -> List[Tuple[str]]:
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select (count(?e1) as ?c) where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e1_),str(ns:)) as ?e1)
            }}
        """

        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return int(results['results']['bindings'][0]['c']['value'])

    def deduce_leaves_count_by_path_two(self, src:str, rels:Tuple[str,str])->List[Tuple[str,str]]:
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select (count(?e2) as ?c) where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                ?e1_ ns:{rels[1]} ?e2_ .
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e2_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?e2_),str(ns:)) as ?e2)
            }}
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return int(results['results']['bindings'][0]['c']['value'])

    def deduce_leaves_count_by_path(self, src: str, path: List[str], no_hop_flag: str = 'END OF HOP') -> Tuple[str]:
        # 效率瓶颈，有待优化
        assert len(path) <= 2
        if len(path) == 0:
            return 1
        elif len(path) == 1:
            return self.deduce_leaves_count_by_path_one(src, path)
        elif len(path) == 2:
            return self.deduce_leaves_count_by_path_two(src, path)
        raise ValueError


    def deduce_leaves_relation_by_path_one(self, src, rels: List[str]):
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?r2 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                ?e1_ ?r2_ ?e2_ .
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e2_, "http://rdf.freebase.com/ns/")
                FILTER regex(?r2_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?r2_),str(ns:)) as ?r2)
            }}
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [i['r2']['value'] for i in results['results']['bindings']]

    def deduce_leaves_relation_by_path_two(self, src, rels: List[str]):
        topic_entity = src

        prefix = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
        """
        query = f"""
            select distinct ?r3 where {{
                ns:{topic_entity} ns:{rels[0]} ?e1_ . 
                ?e1_ ns:{rels[1]} ?e2_ .
                ?e2_ ?r3_ ?e3_ .
                FILTER regex(?e1_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e2_, "http://rdf.freebase.com/ns/")
                FILTER regex(?e3_, "http://rdf.freebase.com/ns/")
                FILTER regex(?r2_, "http://rdf.freebase.com/ns/")
                FILTER regex(?r3_, "http://rdf.freebase.com/ns/")
                bind(strafter(str(?r3_),str(ns:)) as ?r3)
            }}  limit 2000
        """
        
        query = prefix + query
        # print(query)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [i['r3']['value'] for i in results['results']['bindings']]
    
    def deduce_leaves_relation_by_path(self, src, path: List[str]):
        assert len(path) <= 2
        if len(path) == 0:
            return self.get_relation(src)
        elif len(path) == 1:
            return self.deduce_leaves_relation_by_path_one(src, path)
        else:
            return self.deduce_leaves_relation_by_path_two(src, path)            
