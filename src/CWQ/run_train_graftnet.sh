python run_convert_retriever_output_to_graftnet.py 
cp ./tmp/graftnet/* ./graftnet_reader/datasets/webqsp/full/
cd graftnet_reader
python main.py --train config/webqsp.yml
python main.py --test config/webqsp_eval.yml
python script.py webqsp model/webqsp_new/pred_kb model/webqsp_new/pred_kb model/webqsp_new/pred_kb

