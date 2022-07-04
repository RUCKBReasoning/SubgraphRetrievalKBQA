import json
from typing import List, Any
from shutil import copyfile

def load_jsonl(path: str):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


def dump_jsonl(data_list: List[Any], path: str):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj) + "\n")

for data_type in ["train", "test", "dev"]:
    path0 = "../tmp/reader_data/webqsp/{}_simple.json".format(data_type)
    path1 = "datasets/webqsp/full/{}.json".format(data_type)

    train0 = load_jsonl(path0)
    for obj in train0:
        obj["passages"] = {}
        obj["entities"] = [{"kb_id": x, "text": x} for x in obj["entities"]]
        obj["subgraph"]["entities"] = [{"kb_id": x, "text": x} for x in obj["subgraph"]["entities"]]
        obj["subgraph"]["tuples"] = [({'kb_id': h, 'text': h}, {'rel_id': r, 'text': r}, {'kb_id': t, 'text': t}) 
                                        for h, r, t in obj["subgraph"]["tuples"]]

    dump_jsonl(train0, path1)

for file_type in ["entities", "relations"]:
    path0 = "../tmp/reader_data/webqsp/{}.txt".format(file_type)
    path1 = "datasets/webqsp/full/{}.txt".format(file_type)
    copyfile(path0, path1)
