
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
1. run ./code/preprocessing/search_to_get_path.py and ./code/preprocessing/path_to_relation_list.py to obtain train_with_path_score.jsonl, which only contains postive instances, i.e., the right relations in a path for each question.    

2. Then you need to perform negative sampling by replacing the positive relation at each time step with another relation sampled from the other neighboring relations of current step. By doing this, you can obtain the final weak-supervised training data ./code/retriever/data/multi_hop_train.csv.

3. Format of multi-hop_train.csv  <br>
The first column: question plus the first t-1 steps' relations <br>
The second column: the positive relation of the t step <br>
All the subsequent columns: the negative sampled relations of the t step <br>

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


### If you have any questions about our paper, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 

