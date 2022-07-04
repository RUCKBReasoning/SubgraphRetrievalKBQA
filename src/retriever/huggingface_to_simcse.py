"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of SimCSE checkpoint folder")
    parser.add_argument("--ref_path", type=str, help="Path of reference checkpoint folder")
    parser.add_argument("--dump_path", type=str, help="Path of dump folder")
    args = parser.parse_args()

    print("Huggingface checkpoint -> SimCSE checkpoint for {}".format(args.path))

    state_dict = torch.load(os.path.join(args.path, "pytorch_model.bin"), map_location=torch.device("cpu"))
    ref_state_dict = torch.load(os.path.join(args.ref_path, "pytorch_model.bin"), map_location=torch.device("cpu"))

    new_state_dict = {}
    for key, ref_key in zip(state_dict.keys(), ref_state_dict.keys()):
        new_state_dict[ref_key] = state_dict[key]

    torch.save(new_state_dict, os.path.join(args.dump_path, "pytorch_model.bin"))

    # Change architectures in config.json
    config = json.load(open(os.path.join(args.path, "config.json")))
    json.dump(config, open(os.path.join(args.dump_path, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
