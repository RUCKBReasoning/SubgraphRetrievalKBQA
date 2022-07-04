cp ./relations.txt ./tmp/reader_data/CWQ/
python ./run_train_nsm.py 
cd end2end_nsm
python run_retrieve_subgraph.py
sh ./score_paths.sh

