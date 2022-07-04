##
import pandas as pd
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets
from utils import load_jsonl

##
max_length = 64
infile_name = 'data/cwq.jsonl'
tokenizer_config_path = 'SimBERT-CWQ'
data_jsonl_out_path = 'data/cwq_full_tokenized.jsonl'

##
raw_dataset = load_jsonl(infile_name)
dataset_df = pd.DataFrame(raw_dataset, columns=None)
dataset = datasets.Dataset.from_pandas(dataset_df)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_config_path)


def preprocess_item(item):
    str_list = item.values()
    str_list = list(str_list)
    tokenized_out = tokenizer.batch_encode_plus(
        str_list, padding='max_length', max_length=max_length, truncation=True)
    return tokenized_out


pdataset = dataset.map(preprocess_item, num_proc=8,
                       remove_columns=dataset.column_names)
pdataset.to_json(data_jsonl_out_path)
