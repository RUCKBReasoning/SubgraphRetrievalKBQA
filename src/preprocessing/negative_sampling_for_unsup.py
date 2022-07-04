from copy import deepcopy
import json
import os
import random
from time import time
from tkinter import ALL
import numpy as np
import pandas as pd
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut

from utils import load_jsonl
from config import cfg


END_REL = "END OF HOP"


@func_set_timeout(10)
def generate_data_list(json_obj):
    new_data_list = []
    neg_num = 15

    path = json_obj["pos"]

    questions = [
        json_obj["q"][0] + " [SEP]",
        json_obj["q"][1] + " [SEP]",
    ]
    neg_list = sum(json_obj["neg"], [END_REL])

    for q, rel in zip(questions, path):
        
        data_row = []
        data_row.append(q)
        data_row.append(rel)

        if len(neg_list) > 0:
            sample_rels = []
            while len(sample_rels) < neg_num:
                sample_rels.extend(neg_list)
            
            neg_rels = random.sample(sample_rels, neg_num)
            data_row.extend(neg_rels)
            new_data_list.append(data_row)
    
    return new_data_list


def run_negative_sampling():
    load_data_path = cfg.preprocessing["step3"]["unsup_load_data_path"]
    dump_data_path = cfg.preprocessing["step3"]["unsup_dump_data_path"]
    folder_path = cfg.preprocessing["step3"]["dump_data_folder"]
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    data_list = load_jsonl(load_data_path)

    new_data_list = []
    timeout_count = 0
    for json_obj in tqdm(data_list, desc="negative-sampling"):
        try:
            data = generate_data_list(json_obj)
        except FunctionTimedOut:
            continue
        if data is not None:
            new_data_list.extend(data)

    print("timeout_count:", timeout_count)
    
    new_data_list = np.array(new_data_list)
    df = pd.DataFrame(new_data_list)
    df.to_csv(dump_data_path, header=False, index=False)
