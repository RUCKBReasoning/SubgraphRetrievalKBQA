
# Dataset
## QA benchmark
1. WebQuestionSP：Same as the original [WebQuestionSP QA dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763).
2. CWQ: Same as the original [CWQ dataset](https://allenai.org/data/complexwebquestions).

## KG
1. Setup Freebase: We use the whole freebase as the knowledge base. Please follow [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) to build a Virtuoso for the Freebase dataset. 
2. To improve the data accessing efficiency, we extract a 2-hop topic-centric subgraph for each question in WebQSP and a 4-hop topic-centric subgraph for each question in CWQ to create relatively small knowledge graphs. We extract these small knowledge graphs following [NSM](https://github.com/RichardHGL/WSDM2021_NSM). You can download the graphs from [here](https://drive.google.com/drive/folders/1qNauEQJHuMs4uPQcCtMb-M9Seco5mTUl?usp=sharing)

# Running Instructions for WebQSP
## Step0: Prepare the weak-supervised dataset for training the retriever：
## cd WebQSP Q, run the following scripts.

    python run_preprocess.py

## Step1: Train the retriever：

    python run_train_retriever.py

## Step2: Extract a subgraph for each data instance：

    python run_retrieve_subgraph.py

## Step3: Train the reasoner：

    python run_train_nsm.py

## Step4: Fine-tune the retriever by the feeback of the reasoner：

    python run_retriever_finetune.py

## You can also directly run：
    
    ./run.sh

## Download the data：
    
    The data folder tmp can be downloaded from [here](https://drive.google.com/drive/folders/1qNauEQJHuMs4uPQcCtMb-M9Seco5mTUl?usp=sharing).

## For CWQ, you can run ./cwq/run.sh
       
### If you have any questions about the code, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 


