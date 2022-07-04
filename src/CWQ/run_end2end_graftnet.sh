cd end2end_graftnet
python run_retrieve_subgraph.py
python run_convert_retriever_output_to_graftnet_end2end.py
cp ../tmp/graftnet/test.json ./datasets/webqsp/full/finetune.json
sh ./score_paths.sh

