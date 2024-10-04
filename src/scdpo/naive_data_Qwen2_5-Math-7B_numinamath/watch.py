import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from glob import glob

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def get_args():
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("d", type=str, help="directory name")
    args = parser.parse_args()
    return args

def watch():
    args = get_args()
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # the root dir to data generation
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    root_dir = config["root_dir"]
    
    while True:
        for i in range(8):
            file_paths = glob(f"{root_dir}/{args.d}/{i}_round*.jsonl")
            r = sorted([int(file_path.replace(f"{root_dir}/{args.d}/{i}_round", "").replace(".jsonl", "")) for file_path in file_paths])[-1]
            file_path = f"{root_dir}/{args.d}/{i}_round{r}.jsonl"
            if not os.path.exists(file_path):
                length = 0
            else:
                length = len(load_jsonl(file_path))
            source_file = f"{root_dir}/{args.d}/to_be_run_{i}_round{r}.jsonl"
            total_length = len(load_jsonl(source_file))
            if length == count[i] and r == rounds[i]:
                print(f"round{r}: {i} no change: {length} ({total_length-length})")
            else:
                print(f"round{r}: {i}: {length} ({total_length-length})")
                count[i] = length
                rounds[i] = r


        print("\n***************************************\n")
        time.sleep(600)


def main():
    watch()

if __name__ == "__main__":
    main()