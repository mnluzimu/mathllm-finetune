import json
import os
import re
from tqdm import tqdm
import time
from argparse import ArgumentParser
from random import shuffle, seed

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in tqdm(datas):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load"):
            datas.append(json.loads(line))
    return datas

def process_to_alignment(in_files, out_file_train, out_file_test):

    new_datas = []

    for in_file in in_files:
        new_datas.extend(load_jsonl(in_file))
        
    # for data in new_datas:
    #     if len(data["messages"][-1]["content"]) >= 2 and data["messages"][-1]["content"][-1]["type"] == "text" and data["messages"][-1]["content"][-2]["type"] == "text":
    #         data["messages"][-1]["content"] = data["messages"][-1]["content"][:-1]

    seed(3407)
    shuffle(new_datas)
    
    test_train_split = int(0.0001 * len(new_datas))
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
    
def main_math_gsm8k_open_web_math():
    in_files = [
        "/mnt/cache/luzimu/code_generation-master/data/train/back_translation/train_solver/with_sys_gsm8k_math_81087.jsonl",
        "/mnt/cache/luzimu/math_pretrain/data/processed/filtered_open-web-math_Llama3-8B_Mixtral-8x7B_math-related_finer_general-2024052401-3500_add-code/processed_jsonl/processed_filtered_code-added_problem-solution_good-only_1.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/math-gsm8k-81087_open-web-math-add-code-good-only-1-123434"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/math_gsm8k_train.jsonl"
    out_test_file = f"{out_dir}/data/test/math_gsm8k_test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)

def main_open_web_math():
    in_files = [
        "/mnt/cache/luzimu/math_pretrain/data/processed/filtered_open-web-math_Llama3-8B_Mixtral-8x7B_math-related_finer_general-2024052401-3500_add-code/processed_jsonl/processed_filtered_code-added_problem-solution_good-only_v1_123434.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/open-web-math-code-added_problem-solution_good-only_v1_123434"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/train.jsonl"
    out_test_file = f"{out_dir}/data/test/test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)
    
def main_open_web_math_v2_all():
    in_files = [
        "/mnt/cache/luzimu/math_pretrain/data/processed/filtered_open-web-math_Llama3-8B_Mixtral-8x7B_math-related_finer_general-2024052401-3500_add-code/processed_jsonl/processed_filtered_code-added_problem-solution_all_v2_1765680.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/open-web-math-code-added_problem-solution_all_v2_1765680"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/train.jsonl"
    out_test_file = f"{out_dir}/data/test/test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)
    
def main_open_web_math_v1_all_and_WebInstructSub():
    in_files = [
        "/mnt/cache/luzimu/math_pretrain/data/processed/filtered_open-web-math_Llama3-8B_Mixtral-8x7B_math-related_finer_general-2024052401-3500_add-code/processed_jsonl/processed_filtered_code-added_problem-solution_all_v1_1759958.jsonl",
        "/mnt/cache/luzimu/mathllm-finetune/data/train/WebInstructSub_2335220/data/train/train.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/open-web-math-code-added_problem-solution_all_v1_1759958_WebInstructSub_2335220"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/train.jsonl"
    out_test_file = f"{out_dir}/data/test/test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)
    
def main_open_web_math_no_code():
    in_files = [
        "/mnt/cache/luzimu/math_pretrain/data/processed/filtered_open-web-math_Llama3-8B_Mixtral-8x7B_math-related_finer_general-2024052401-3500/messages_jsonl/processed_filtered_problem-solution_1765680.jsonl"
    ]
    out_dir = f"/mnt/cache/luzimu/mathllm-finetune/data/train/open-web-math_problem-solution_no-code"
    if not os.path.exists(f"{out_dir}/data/train/"):
        os.makedirs(f"{out_dir}/data/train/")
    if not os.path.exists(f"{out_dir}/data/test/"):
        os.makedirs(f"{out_dir}/data/test/")
    out_train_file = f"{out_dir}/data/train/train.jsonl"
    out_test_file = f"{out_dir}/data/test/test.jsonl"
    process_to_alignment(in_files, out_train_file, out_test_file)
    
if __name__ == "__main__":
    main_open_web_math_no_code()