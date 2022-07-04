from config import cfg
import os
import subprocess

def run():
    load_data_path = cfg.train_nsm["load_data_path"]
    dump_model_path = cfg.train_nsm["dump_model_path"]
    if not os.path.exists(dump_model_path):
        os.makedirs(dump_model_path)
    
    subprocess.run(["bash", "run_train_nsm.sh", load_data_path, dump_model_path])

if __name__ == '__main__':
    run()
