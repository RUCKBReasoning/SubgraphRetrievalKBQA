import json


def load_jsonl(filepath):
    item_list = []
    for line in open(filepath):
        item = json.loads(line)
        item_list.append(item)
    return item_list


def dump_jsonl(item_list, filepath):
    with open(filepath, 'w') as f:
        for item in item_list:
            outline = json.dumps(item, ensure_ascii=False)
            print(outline, file=f)


def load_item2id_map(filepath):
    item_list = []
    with open(filepath) as f:
        for line in f:
            item_list.append(line.strip())
    id2item = item_list
    item2id = {item:idx for idx, item in enumerate(item_list)}
    return id2item, item2id