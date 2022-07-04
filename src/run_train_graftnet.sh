cd graftnet_reader

python load_data_from_nsm.py

python main.py --train config/webqsp.yml

python main.py --test config/webqsp.yml

python script.py webqsp model/webqsp_new/pred_kb model/webqsp_new/pred_kb model/webqsp_new/pred_kb

cd ..
