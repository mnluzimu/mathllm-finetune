import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser

from random import seed, shuffle

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for idx, data in enumerate(datas):
            data["idx"] = idx
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def get_initial_data(in_file, out_dir, n):
    datas = load_jsonl(in_file)
    new_datas = []
    
    seed(3407)
    shuffle(datas)

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data.keys():
            extra = data["extra"]
        else:
            extra = data
        correct_num = 0
        wrong_num = 0
        if "correct_solutions" in data.keys():
            correct_num = len(data["correct_solutions"])
        if "wrong_solutions" in data.keys():
            wrong_num = len(data["wrong_solutions"])
        new_data = {
            "question": data["question"],
            "extra": extra,
            "correct_num": correct_num,
            "wrong_num": wrong_num
        }
        new_datas.append(new_data)

    total = len(new_datas)
    steps = (total + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i * steps: i * steps + steps], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))
        save_jsonl(datas[i * steps: i * steps + steps], os.path.join(out_dir, f"result_{i}.jsonl"))


def get_initial_data_milti_infiles(in_files, out_dir, n):
    datas = []
    for in_file in in_files:
        datas.extend(load_jsonl(in_file))

    seed(3407)
    shuffle(datas)

    new_datas = []

    for idx, data in tqdm(enumerate(datas)):
        if "extra" in data.keys():
            extra = data["extra"]
        else:
            extra = {k: v for k, v in data.items() if not k.endswith("solutions") and not k == "question" and not k == "problem"}
        
        if "correct_solutions" not in data.keys():
            data["correct_solutions"] = []
        if "wrong_solutions" not in data.keys():
            data["wrong_solutions"] = []
        if "correct_errored_solutions" not in data.keys():
            data["correct_errored_solutions"] = []

        correct_num = len(data["correct_solutions"])
        wrong_num = len(data["wrong_solutions"])
        if "question" not in data.keys():
            data["question"] = data["problem"]
        new_data = {
            "question": data["question"],
            "extra": extra,
            "correct_num": correct_num,
            "wrong_num": wrong_num
        }
        new_datas.append(new_data)

    total = len(new_datas)
    steps = (total + n - 1) // n
    for i in range(n):
        save_jsonl(new_datas[i::n], os.path.join(out_dir, f"to_be_run_{i}_round1.jsonl"))
        save_jsonl(datas[i::n], os.path.join(out_dir, f"result_{i}.jsonl"))


# the root dir to data generation
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
root_dir = config["root_dir"]


def main_gsm8k():
    global root_dir
    in_file = "/mnt/cache/luzimu/datasets_en/GSM8K/GSM8K_train.jsonl"
    out_dir = os.path.join(root_dir, "gsm8k")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 8
    get_initial_data(in_file, out_dir, n)


def main_math():
    global root_dir
    in_file = "/mnt/cache/luzimu/datasets_en/MATH/MATH_train.jsonl"
    out_dir = os.path.join(root_dir, "math")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n = 8
    get_initial_data(in_file, out_dir, n)


if __name__ == "__main__":
    main_gsm8k()
    main_math()