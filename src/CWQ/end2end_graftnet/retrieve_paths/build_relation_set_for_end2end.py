'''构建 Local KB 中 relation 集合以及 entity 集合 '''
import os
import json
from tqdm import tqdm
from utils import load_jsonl
from loguru import logger
#from config import cfg

def run():
    load_data_path = "../tmp/reader_data/CWQ_end2end/"
    train_dataset = load_jsonl(os.path.join(load_data_path, "test_simple.json"))

    entity_set = list()
    out_entity_set_filename = os.path.join(load_data_path, 'entities.txt')
    out_relation_set_filename = os.path.join(load_data_path, 'relations.txt')

    for dataset in [train_dataset]:
        for json_obj in tqdm(dataset):
            if "subgraph" not in json_obj:
                print("Slic with no SubGraph")
                continue
            answers = [ans_json_obj
                    for ans_json_obj in json_obj["answers_cid"]]
            subgraph_entities = list(json_obj["subgraph"]["entities"])
            entity_set.extend(answers)
            entity_set.extend(subgraph_entities)
            #= entity_set + answers + subgraph_entities

    def dump_list_to_txt(mylist, outname):
        with open(outname, 'w') as f:
            for item in mylist:
                print(item, file=f)

    entity_set = sorted(set(entity_set))

    dump_list_to_txt(entity_set, out_entity_set_filename)

if __name__ == '__main__':
    run()
