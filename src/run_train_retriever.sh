#! /bin/bash

set -e

load_data_path=${1}
dump_model_path=${2}
device=0

cd retriever

CUDA_VISIBLE_DEVICES=${device} python train.py \
    --model_name_or_path roberta-base \
    --train_file ../${load_data_path} \
    --output_dir ../${dump_model_path} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train

python simcse_to_huggingface.py --path ../${dump_model_path}

cd ..
