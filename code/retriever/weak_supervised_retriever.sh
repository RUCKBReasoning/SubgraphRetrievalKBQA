#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path roberta-base \
    --train_file ../tmp/retriever/multi_hop_train.csv \
    --output_dir ../tmp/model_ckpt/SimBERT_test \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"
