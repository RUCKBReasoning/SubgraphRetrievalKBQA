#! /bin/bash

set -e

device=1
dataset_name="webqsp"
train_dataset_path="../../data/WebQSP/data/WebQSP.train.json"
test_dataset_path="../../data/WebQSP/data/WebQSP.test.json"

# preprocessing begin

cd preprocessing

python run_preprocess.py \
    --dataset_name ${dataset_name} \
    --train_dataset_path ${train_dataset_path} \
    --test_dataset_path ${test_dataset_path}

cd ..

# preprocessing end


# retriver begin

cd retriever

output_dir="../tmp/model_ckpt/SimBERT"

CUDA_VISIBLE_DEVICES=${device} python train.py \
    --model_name_or_path roberta-base \
    --train_file "../tmp/retriever/multi_hop_train.csv" \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"

python simcse_to_huggingface.py --path ${output_dir}

cd ..

# retriver end

# reader preprocess begin

if [ ${dataset_name} == "webqsp" ]
then

    cp tmp/nsm_origin/webqsp/* tmp/reader
    cd reader_preprocessing

    CUDA_VISIBLE_DEVICES=${device} python retrieve_subgraph.py \
        --load_data_path "../tmp/nsm_origin/webqsp/" \
        --dump_data_path "../tmp/reader_test/"

    python evaluate.py

    cd ..

fi

# reader preprocess end
