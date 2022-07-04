#! /bin/bash

set -e

cd src

python run_preprocess.py

python run_train_retriever.py

python run_retrieve_subgraph.py

python run_train_nsm.py
