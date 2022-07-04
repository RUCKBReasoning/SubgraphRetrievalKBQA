
# Dataset
## QA benchmark
1. WebQuestionSP：Same as the original [WebQuestionSP QA dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52763).
2. CWQ: Same as the original [CWQ dataset](https://allenai.org/data/complexwebquestions).

## KG
1. Setup Freebase: We use the whole freebase as the knowledge base. Please follow [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) to build a Virtuoso for the Freebase dataset. 
2. To improve the data accessing efficiency, we extract a 2-hop topic-centric subgraph for each question in WebQSP and a 4-hop topic-centric subgraph for each question in CWQ to create relatively small knowledge graphs. Since the multi-hop subgraphs are often quite large, we make the following restrictions during subgraph extraction. Since freebase contains a virtual CVT node type which represents complex data such as an event and a CVT node usually contains a few neighboring nodes to describe the virtual CVT node, we define a one-hop path as "entity-entity" or "entity-CVT-entity". Then a two-hop path is the combination of these two kinds of one-hop paths including "entity-entity-entity", "entity-CVT-entity-entity", "entity-entity-CVT-entity", and "entity-CVT-entity-CVT-entity". Following these paths, we extract the 1-hop subgraphs which can be also viewed as the 2-hop subgraphs including the CVT nodes for WebQSP and extract the 2-hop subgraphs which can be viewed as the 4-hop subgraphs including the CVT node for CWQ. The extracted CWQ knowledge graph contains 56,382,529 entities, 26,282 relations, and 190,123,031 triplets. The extracted WebQSP knowledge graph contains 6,317,272 entities, 5,516 relations, and 15,839,623 triplets. We evaluate the answer coverage rates of the two knowledge graphs, which are both larger than 99\%.. Download (TODO).

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
    
    The data folder tmp can be downloaded from [here]().

## For CWQ, you can run ./cwq/run.sh
       
### If you have any questions about the code, please contact Xiaokang Zhang (zhang2718@ruc.edu.cn)! 


