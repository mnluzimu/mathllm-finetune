import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from random import shuffle, seed

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def process_to_alignment(in_files, out_file_train, out_file_test):

    new_datas = []

    for in_file in in_files:
        new_datas.extend(load_jsonl(in_file))

    seed(3407)
    shuffle(new_datas)
    
    test_train_split = int(0.001 * len(new_datas))
    save_jsonl(new_datas[:test_train_split], out_file_test)
    save_jsonl(new_datas, out_file_train)


def main_ape_th2_mathgenie():
    in_files = [
        "/mnt/cache/luzimu/code_generation-master/data/train/dpo_sft_files/ape_th2_326221.jsonl",
        "/mnt/cache/luzimu/code_generation-master/data/train/back_translation/train_mixed/filtered_AugGSM8K_AugMATH_ch1200_ch1600_gsm8kMath_verify_284468.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/ape-th2-326221-mathgenie-284468"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)


if __name__ == "__main__":
    main_ape_th2_mathgenie()