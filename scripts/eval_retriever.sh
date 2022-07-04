#! /bin/bash

set -e

cd src

cd retrieve_subgraph

device=1

load_data_path="../tmp/data/origin_nsm_data/webqsp"
dump_data_path="../tmp/reader_data/test"
# model_ckpt="/home/huyuxuan/projects/SubgraphRetrievalKBQA-webqsp/src/tmp/model_ckpt/e2e_finetune_graftnet_qsp"
model_ckpt="../tmp/model_ckpt/unsup_weak_SimBERT"

CUDA_VISIBLE_DEVICES=${device} python retrieve_subgraph_for_test.py \
    --load_data_folder ${load_data_path} \
    --dump_data_folder ${dump_data_path} \
    --model_ckpt ${model_ckpt} \
    --top_k 10

cd ..

cd evaluate

python eval.py --load_data_path ${dump_data_path}

cd ..
