from config import cfg
import os
import subprocess

def run():
    load_data_path = cfg.train_retriever["load_data_path"]
    dump_model_path = cfg.train_retriever["dump_model_path"]
    
    if not os.path.exists(dump_model_path):
        os.makedirs(dump_model_path)
    
    subprocess.run(["bash", "run_train_retriever.sh", load_data_path, dump_model_path])
    
    print("[process finish]")

if __name__ == '__main__':
    run()
