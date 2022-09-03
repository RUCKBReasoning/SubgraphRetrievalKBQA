cd nsm_reader

load_data_path=${1}
dump_model_path=${2}

CUDA_VISIBLE_DEVICES=1 python main_nsm.py \
    --data_folder ../${load_data_path} \
    --checkpoint_dir ../${dump_model_path} \
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
    --reason_kb

cd ..
