#infile = "./tmp/reader_data/CWQ/"
infile = './tmp/reader_data/CWQ/'
outfile = './tmp/graphnet/'
from utils import load_jsonl,dump_jsonl
import json
def entity_conversion(ent):
    ent = {"kb_id":str(ent),"text":str(ent)}
    return ent

def relation_conversion(rel):
    rel = {"rel_id":rel,"text":str(rel)}
    return rel

def slice_conversion(input_slic):
    output_slic = dict()
    output_slic["passages"] = []
    
    #convert subgraph
    output_slic["subgraph"] = dict()
    output_slic["subgraph"]["entities"] = [entity_conversion(ent) for ent in input_slic["subgraph"]["entities"]]
    output_slic["subgraph"]["tuples"] = [[entity_conversion(h),relation_conversion(r),entity_conversion(t)] for h,r,t in input_slic["subgraph"]["tuples"]]

    #question
    output_slic["question"] = input_slic["question"]

    #answer
    output_slic["answers"] = [entity_conversion(ent) for ent in input_slic["answers_cid"]]

    #topic entities
    output_slic["entities"] = [entity_conversion(ent) for ent in input_slic["entities_cid"]]

    #id
    output_slic['id'] = input_slic['id']
    
    if 'paths' in input_slic:
        output_slic['paths'] = input_slic['paths']
    return output_slic

for subName in ["train","test",'dev']:
    dataset_in = load_jsonl(infile+subName+"_simple.json")
    dataset_out = [slice_conversion(slic) for slic in dataset_in if "subgraph" in slic]
    dump_jsonl(dataset_out,outfile+subName+".json")

