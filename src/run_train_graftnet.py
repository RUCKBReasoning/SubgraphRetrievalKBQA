from config import cfg
import os
import subprocess

def run():    
    subprocess.run(["bash", "run_train_graftnet.sh"])

if __name__ == '__main__':
    run()
