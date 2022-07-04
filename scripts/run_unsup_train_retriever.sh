#! /bin/bash

set -e

load_data_path="../tmp/retriever/unsup_train.csv"
dump_model_path="../tmp/model_ckpt/unsup_SimCSE_roberta"
device=0

cd src

cd retriever

model_name_or_path="roberta-base"

CUDA_VISIBLE_DEVICES=${device} python train.py \
    --model_name_or_path ${model_name_or_path} \
    --train_file ${load_data_path} \
    --output_dir ${dump_model_path} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train

cd ..
