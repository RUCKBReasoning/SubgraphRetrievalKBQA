import json
import numpy as np
from NSM.util.config import get_config
import time
from NSM.data.dataset_super import SingleDataLoader


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def translate_relation_text_into_tokens(filename, word2id):
    relation_tokens = []
    for line in open(filename):
        relation_token = []
        raw_relation = line.strip().replace('_','.').split('.')
        for word in raw_relation:
            if word in word2id:
                relation_token.append(word2id[word])
            else:
                relation_token.append(len(word2id))
        relation_tokens.append(relation_token)
    # relation_tokens.append([word2id[word] for word in ['self','loop']])
    max_relation_len = max([len(relation) for relation in relation_tokens])
    # plus one for 'self-loop' relation
    relation_tokens_np = np.full((len(relation_tokens)+1, max_relation_len), len(word2id), dtype=int)
    for i, relation_token in enumerate(relation_tokens):
        for j, token in enumerate(relation_token):
            relation_tokens_np[i, j] = token
    return relation_tokens_np

    

def load_data(config):
    entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    relation_tokens = translate_relation_text_into_tokens(config['data_folder'] + config['relation2id'], word2id)
    if config["is_eval"]:
        train_data = None
        valid_data = None
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="dev")
    # test_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="test")
    test_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="finetune")
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "relation_tokens":relation_tokens,
        "word2id": word2id
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    args = get_config()
    load_data(args)
