## 
import pandas as pd
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import datasets
from utils import load_jsonl

## 
max_length = 64

tokenizer_config_path = 'SimBERT-CWQ'
data_csv_in_path = 'data/full/full_data.csv'
data_jsonl_out_path = 'data/full/full_tokenized.jsonl'
## 
df = pd.read_csv(data_csv_in_path, header=None)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_config_path)

with open(data_jsonl_out_path, 'w') as f:
    for _, r in tqdm(df.iterrows()):
        tokenized_out = tokenizer.batch_encode_plus(
            list(r), padding='max_length', max_length=max_length, truncation=True)
        tokenized_out = dict(tokenized_out)
        out_json_line = json.dumps(tokenized_out, ensure_ascii=False)
        print(out_json_line, file=f)
