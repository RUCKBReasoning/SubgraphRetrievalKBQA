import os
import subprocess
from retrieve_subgraph.retrieve_subgraph_for_finetune import run as run_retrieve_subgraph
from config import cfg


def run():
    num_epoch = 1
    for _ in range(num_epoch):
        load_data_path = cfg.retriever_finetune["load_data_path"]
        dump_model_path = cfg.retriever_finetune["dump_model_path"]
        checkpoint_dir = cfg.retriever_finetune["checkpoint_dir"]
        dump_data_path = cfg.retriever_finetune["dump_data_path"]
        
        if not os.path.exists(dump_model_path):
            os.makedirs(dump_model_path)
        
        # step0
        run_retrieve_subgraph()
        # step1
        subprocess.run(["bash", "run_retriever_finetune.sh", load_data_path, dump_model_path, checkpoint_dir, dump_data_path])


if __name__ == '__main__':
    run()
