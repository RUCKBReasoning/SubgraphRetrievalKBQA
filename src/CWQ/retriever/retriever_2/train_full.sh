#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --train_dataset data/full/train_tokenized.jsonl \
    --eval_dataset data/full/dev_tokenized.jsonl \
    --output_dir ./output_dir_full \
    --logging_dir ./log_dir_full \
    --pretrained_model_name_or_path SimBERT-CWQ-Roberta-PART50 \
    --per_device_train_batch_size 24 \
    --num_train_epochs 5


