#! /bin/bash

cd src

cd graftnet_reader

python load_data_from_nsm.py

python main.py --train config/webqsp.yml

python main.py --test config/webqsp.yml

python script.py webqsp model/webqsp_new/pred_kb model/webqsp_new/pred_kb model/webqsp_new/pred_kb

cd ..


# cd retrieve_subgraph

# python retrieve_subgraph_for_graftnet.py --load_data_folder ../GraftNet-master/datasets/webqsp/full

# cd ..

# cd GraftNet-master

# python main.py --train config/webqsp.yml

# python main.py --test config/webqsp_test.yml

# python script.py webqsp model/webqsp_new/pred_kb model/webqsp_new/pred_kb model/webqsp_new/pred_kb

# cd ..
