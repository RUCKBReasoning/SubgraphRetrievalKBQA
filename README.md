
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
'''
python run_preprocess.py  --dataset_name ${dataset_name} --train_dataset_path ${train_dataset_path}  --test_dataset_path ${test_dataset_path}
'''
to obtain weak_supervised_train.csv

The Format of weak_supervised_train.json is <br>
'''
The first column: question plus the first t-1 steps' relations <br>
The second column: the positive relation of the t step <br>
All the subsequent columns: the negative sampled relations of the t step <br>
'''
## 4. How to obtain the unsupervised dataset for training the retriever?
TODO

# Training Process

## 1. How to perform the weak-supervised training of the retriever?
run ./code/retriever/weak_supervised_retriever.sh with the --train_file as multi_hop_train.csv

## 2. How to perform the unsupervised learning of the retriever?
run ./code/retriever/weak_supervised_retriever.sh with the --train_file as unsupervised.csv (TODO)

## 3. How to sample a subgraph for each question to generate the train/test file for reader?
1. run ./code/retriever/code/retrieve_subgraph.py to obtain \*_for_NSM.json (\* means train or test) for reader training.

2. Format of *_for_NSM.json <br>
Each line contains a question, a topic_entity, an answer, entities in a subgraph, and triplets in a subgraph

## 4. How to train the reader?
run ./code/reader/NSM/main_nsm.py by replacing the file *_simple.json in the its data_folder with *_for_NSM.json

## How to perform end-to-end training?
TODO

### You can also run ./code/run_preprocess.sh


### If you have any questions about our paper, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 

