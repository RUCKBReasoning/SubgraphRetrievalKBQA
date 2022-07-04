#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

#NUM_GPU=3

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.

# Allow multiple threads
#export OMP_NUM_THREADS=8
#export CUDA_VISIBLE_DEVICES=0,1,2
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node 3 --master_port $(expr $RANDOM + 1000) train.py
