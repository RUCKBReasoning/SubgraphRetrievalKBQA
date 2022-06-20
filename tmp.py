import json
import urllib
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://10.77.110.128:3001/sparql")
sparql.setReturnFormat(JSON)

def deduce_leaves_count_by_path_one(src:str, rels:str):
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
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    print(json.dumps(results, indent=4, ensure_ascii=False))

deduce_leaves_count_by_path_one("m.01_2n", ["tv.tv_program.regular_cast"])
