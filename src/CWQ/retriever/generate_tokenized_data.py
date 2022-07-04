# %%
import pandas as pd
import json
from transformers import BertTokenizer
from tqdm import tqdm
import datasets
from utils import load_jsonl
# %%
max_length = 64

# %%
infile_name = '../tmp/cwq_full_data_raw_for_retriever.jsonl'
tokenizer_config_path = 'bert-base-uncased'
data_jsonl_out_path = 'full_tokenized.jsonl'
train_data_out_path = 'train_tokenized.jsonl'
dev_data_out_path = 'dev_tokenized.jsonl'
# %%
raw_dataset = load_jsonl(infile_name)
dataset_df = pd.DataFrame(raw_dataset, columns=None)
dataset = datasets.Dataset.from_pandas(dataset_df)

tokenizer = BertTokenizer.from_pretrained(
    tokenizer_config_path)

# %%
def preprocess_item(item):
    str_list = item.values()
    tokenized_out = tokenizer.batch_encode_plus(
        str_list, padding='max_length', max_length=max_length, truncation=True)
    return tokenized_out


pdataset = dataset.map(preprocess_item, num_proc=8,
                       remove_columns=dataset.column_names)
#splited = pdataset.train_test_split(train_size=150000,test_size=10000,shuffle=False)
splited = pdataset.train_test_split(train_size=0.9,test_size=0.1,shuffle=False)
train = splited["train"]
dev = splited["test"]
train.to_json(train_data_out_path)
dev.to_json(dev_data_out_path)
pdataset.to_json(data_jsonl_out_path)
