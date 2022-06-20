import argparse
from load_dataset import DatasetPath, load_webqsp
from path_to_relation_list import run_path_to_relation
from search_to_get_path import run_search_to_get_path
from negative_sampling import run_negative_sampling


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)    
    parser.add_argument("--KG_name", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    args = parser.parse_args()
    return args


def run(args):
    path = DatasetPath(
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path
    )
    if args.dataset_name == 'webqsp':
        load_webqsp(path)
    else:
        raise ValueError
    run_search_to_get_path(args)
    run_path_to_relation(args)
    run_negative_sampling(args)
    print("[preprocess finish]")


if __name__ == '__main__':
    run(get_args())
