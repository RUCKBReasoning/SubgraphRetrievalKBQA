
# Dataset
## QA benchmark
1. WebQuestionSP：Same as the original [WebQuestionSP QA dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763).
2. CWQ: Same as the original [CWQ dataset](https://allenai.org/data/complexwebquestions).

## KG
1. Setup Freebase: We use the whole freebase as the knowledge base. Please follow [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) to build a Virtuoso for the Freebase dataset. 
2. To improve the data accessing efficiency, for the webqsp dataset, we also prepare a KB cache, in which we extract a two-hop subgraph including the freebase CVT nodes for each topic entity in the webqsp dataset. For the CWQ dataset, we prepare a KB cache, in which we extract a four-hop subgraph including the freebase CVT nodes for each topic entity in the CWQ dataset. Download (TODO).

# Running Instructions
## cd WebQSP or cd CWQ, run the following scripts.
## Step0: Prepare the weak-supervised dataset for training the retriever：

    python run_preprocess.py

## Step1: Train the retriever：

    python run_train_retriever.py

## Step2: Extract a subgraph for each data instance：

    python run_retrieve_subgraph.py

## Step3: Train the reasoner：

    python run_train_nsm.py or python run_train_graftnet.py

## Step4: Fine-tune the retriever by the feeback of the reasoner：

    python run_retriever_finetune.py

## You can also directly run：
    
    ./run.sh

## Download the data：
    
    The data folder tmp can be downloaded from [here](https://pan.baidu.com/s/1EUR5kxDxiDr-SzZ2dQ4bPQ?pwd=y408) with password y408.
    

### If you have any questions about the code, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 


