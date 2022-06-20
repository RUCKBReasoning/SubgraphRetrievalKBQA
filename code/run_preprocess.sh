#! /bin/bash

set -e

device=0
dataset_name="webqsp"

# preprocessing begin

# cd preprocessing

# KG_name="Freebase"
# train_dataset_path="../tmp/data/WebQSP/data/WebQSP.train.json"
# test_dataset_path="../tmp/data/WebQSP/data/WebQSP.test.json"

# python run_preprocess.py \
#     --KG_name ${KG_name} \
#     --dataset_name ${dataset_name} \
#     --train_dataset_path ${train_dataset_path} \
#     --test_dataset_path ${test_dataset_path}

# cd ..

# preprocessing end


# retriver begin

# cd retriever

# output_dir="../tmp/model_ckpt/SimBERT"

# CUDA_VISIBLE_DEVICES=${device} python train.py \
#     --model_name_or_path roberta-base \
#     --train_file "../tmp/retriever/weak_supervised_train.csv" \
#     --output_dir ${output_dir} \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-5 \
#     --max_seq_length 32 \
#     --metric_for_best_model stsb_spearman \
#     --pooler_type cls \
#     --overwrite_output_dir \
#     --temp 0.05 \
#     --do_train \
#     --fp16 \
#     "$@"

# python simcse_to_huggingface.py --path ${output_dir}

# cd ..

# retriver end

# reader preprocess begin

if [[ ${dataset_name} == "webqsp" ]];
then
    cd reader_preprocessing

    load_data_path="../tmp/nsm_data/webqsp"

    # cp ${load_data_path}/* "../tmp/reader"

    CUDA_VISIBLE_DEVICES=${device} python retrieve_subgraph.py \
        --load_data_path ${load_data_path} \
        --dump_data_path "../tmp/reader/"

    python eval.py

    cd ..

fi

# reader preprocess end
