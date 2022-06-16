
# Dataset Creation

## 1. WebQuestionSP

### KG
We now tentatively refer to the pruned KB provided by [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA). You can download it from [here](https://pan.baidu.com/s/1FTKgDf-VqSna6Ghdc58ncw) with password ln74.
TODO: change the KG to the whole freebase as our paper used.

### QA benchmark
Same as the original [WebQuestionSP QA dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763)

## 2. CWQ

### KG
Freebase. 

### QA benchmark
Same as the original [CWQ dataset](https://allenai.org/data/complexwebquestions)

## 3. How to obtain the weak-supervised dataset for training the retriever?

    cd preprocessing
    python run_preprocess.py \
        --dataset_name ${dataset_name} \
        --train_dataset_path ${train_dataset_path} \
        --test_dataset_path ${test_dataset_path}
    python run_preprocess.py \
        --dataset_name ${dataset_name} \
        --train_dataset_path ${train_dataset_path} \
        --test_dataset_path ${test_dataset_path}


to obtain weak_supervised_train.csv

The Format of weak_supervised_train.csv is <br>
    The first column: question plus the first t-1 steps' relations <br>
    The second column: the positive relation of the t step <br>
    All the subsequent columns: the negative sampled relations of the t step <br>


## 4. How to obtain the unsupervised dataset for training the retriever?
TODO

# Training Process

## 1. How to perform the weak-supervised training of the retriever?

    cd retriever

    output_dir="../tmp/model_ckpt/SimBERT"

    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --model_name_or_path roberta-base \
        --train_file "../tmp/retriever/weak_supervised_train.csv" \
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



## 2. How to perform the unsupervised learning of the retriever? (TODO)

## 3. How to sample a subgraph for each question to generate the train/test file for reader?

    cp tmp/nsm_origin/webqsp/* tmp/reader
    cd reader_preprocessing

    CUDA_VISIBLE_DEVICES=${device} python retrieve_subgraph.py \
            --load_data_path "../tmp/nsm_origin/webqsp/" \
            --dump_data_path "../tmp/reader_test/"

    python evaluate.py

to obtain train/dev/test_simple.json
Each line of the file includes a question, topic entities, answers, subgraph composed by a list of triplets and a list of entities.

## 4. How to train the reader?
    cd reader

    CUDA_VISIBLE_DEVICES=1 python main_nsm.py \
        --data_folder "../tmp/reader/" \
        --checkpoint_dir "../tmp/model_ckpt/nsm/" \
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

## How to perform end-to-end training?
TODO

You can also directly run ./code/run_preprocess.sh and ./code/run_nsm.sh. <br>

You can download the folder tmp from [here](https://pan.baidu.com/s/1EUR5kxDxiDr-SzZ2dQ4bPQ?pwd=y408) with password y408.

### If you have any questions about the code, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 


