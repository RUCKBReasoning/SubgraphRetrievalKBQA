from preprocessing.load_dataset import load_webqsp
from preprocessing.score_path import run_score_path
from preprocessing.search_to_get_path import run_search_to_get_path
from preprocessing.negative_sampling import run_negative_sampling

def run():
    load_webqsp()    
    run_search_to_get_path()
    run_score_path()
    run_negative_sampling()

if __name__ == '__main__':
    run()
