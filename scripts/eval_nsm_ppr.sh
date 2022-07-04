#! /bin/bash

cd src

cd reader

data_path="../tmp/reader_data/webqsp_ppr_85p/"
ckpt_path="../tmp/model_ckpt/nsm_ppr_85p/"

CUDA_VISIBLE_DEVICES=0 python main_nsm.py \
    --data_folder ${data_path} \
    --checkpoint_dir ${ckpt_path} \
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
    --num_epoch 50 \
    --use_self_loop \
    --lr 5e-4 \
    --word_emb_file word_emb_300d.npy \
    --loss_type kl \
    --reason_kb

CUDA_VISIBLE_DEVICES=1 python main_nsm.py \
    --data_folder ${data_path} \
    --checkpoint_dir ${ckpt_path} \
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
    --load_experiment "webqsp_nsm-h1.ckpt"

cd ..

cd evaluate

load_nsm_test_info_path="${ckpt_path}webqsp_nsm_test.info"

echo "eval.py:"

python eval.py --load_data_path ${data_path}

# echo "eval_figure4-1.py:"

# python eval_figure4_1.py --load_data_path ${data_path} --load_nsm_test_info_path ${load_nsm_test_info_path}

cd ..
