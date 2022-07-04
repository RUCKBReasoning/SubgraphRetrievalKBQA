'''构造每个问题的搜素树'''
import json
import pickle
from pprint import pprint


def label_path(pos_path_list, neg_path_list):
    search_tree = dict()
    for pos_path in pos_path_list:
        for i in range(len(pos_path)):
            history = tuple(pos_path[:i])
            search_tree.setdefault(history, {'pos': [], 'neg': []})
            search_tree[history]['pos'].append(pos_path[i])
    for neg_path in neg_path_list:
        for i in range(len(neg_path)):
            history = tuple(pos_path[:i])
            if history in search_tree:
                nhop = neg_path[i]
                if nhop not in search_tree[history]['pos']:
                    search_tree[history]['neg'].append(nhop)
                    break
    return search_tree


def split_pos_list_and_neg_list(path_with_score_list):
    pos_path_list, neg_path_list = [], []
    path_val_list = [pval for path, pval in path_with_score_list]
    if max(path_val_list) >= 0.5:
        threshold = 0.5
    else:
        threshold = max(0.05, max(path_val_list)-0.3)
    for path, pval in path_with_score_list:
        if pval >= threshold:
            pos_path_list.append(path)
        else:
            neg_path_list.append(path)
    return pos_path_list, neg_path_list


if __name__ == '__main__':
    infile_name = 'finetune_data_of_cwq/m_dict.json'
    outfile_name = 'finetune_data_of_cwq/search_tree_list.pkl'
    with open(infile_name) as f:
        dataset = json.load(f)
    outlist = []
    for question, tp_and_path_with_score_list in dataset.items():
        tp, path_with_score_list = tp_and_path_with_score_list[
            'topic_entities'], tp_and_path_with_score_list['path']
        pos_path_list, neg_path_list = split_pos_list_and_neg_list(
            path_with_score_list)
        if pos_path_list and neg_path_list:
            search_tree = label_path(pos_path_list, neg_path_list)
            outlist.append(dict(question=question, search_tree=search_tree, topic_entities=tp))
    pprint(outlist[10])
    with open(outfile_name, 'wb') as outf:
        pickle.dump(outlist, outf)
