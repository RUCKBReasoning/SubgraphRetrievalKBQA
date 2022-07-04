#! /bin/bash

set -e

cd retriever_finetune

load_data_path=${1}
dump_model_path=${2}
checkpoint_dir=${3}
dump_data_path=${4}

CUDA_VISIBLE_DEVICES=1 python main_nsm.py \
    --data_folder ../${load_data_path} \
    --checkpoint_dir ../${checkpoint_dir} \
    --model_name gnn \
    --batch_size 20 \
    --test_batch_size 40 \
    --num_step 3 \
    --entity_dim 50 \
    --word_dim 300 \
    --kg_dim 100 \
    --kge_dim 100 \
    --eval_every 2 \
    --encode_type \
    --experiment_name webqsp_nsm \
    --eps 0.95 \
    --num_epoch 200 \
    --use_self_loop \
    --lr 5e-4 \
    --word_emb_file word_emb_300d.npy \
    --loss_type kl \
    --reason_kb \
    --is_eval \
    --load_experiment "webqsp_nsm-final.ckpt"

python cal_path_score_from_reader.py --load_data_path ../${load_data_path}

python build_search_tree_for_dataset.py

python build_train_set_from_search_tree.py --dump_data_path ../${dump_data_path}

cd ..

echo "retrain retriever from new data in 'src/tmp/retriever' with lr=1e-5, epoch=1"
echo "retrain reader with the new retriever"
