import json
import os
from tqdm import tqdm
import sympy as sp


def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in tqdm(datas, desc="save"):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load"):
            datas.append(json.loads(line))
    return datas


def add_prompt(in_file, out_file, system_prompt):
    datas = load_jsonl(in_file)
    for data in tqdm(datas, desc="process"):
        data["messages"][0]["content"][0]["content"] = system_prompt
        if len(data["messages"][2]["content"]) >= 2 and data["messages"][2]["content"][-1]["type"] == "text" and data["messages"][2]["content"][-2]["type"] == "text" and len(data["messages"][2]["content"][-1]["content"]) < 100:
            data["messages"][2]["content"] = data["messages"][2]["content"][:-1]
    save_jsonl(datas, out_file)


def main_81000():
    in_file = "/mnt/cache/luzimu/mathllm-finetune/data/train/math-gsm8k-lce-81007/data/train/math_gsm8k_train.jsonl"
    out_file = "/mnt/cache/luzimu/mathllm-finetune/data/train/cot_lce_add-sys-prompt/math-gsm8k-lce-81007_lce-sys.jsonl"
    system_prompt = "Please integrate natural language reasoning with programs to solve the following problem, and put your final answer within \\boxed{}."
    add_prompt(in_file, out_file, system_prompt)

def main_numina_lce():
    in_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-TIR/jsonl/numinamath-lce_72441.jsonl"
    out_file = "/mnt/cache/luzimu/mathllm-finetune/data/train/cot_lce_add-sys-prompt/numinamath-lce_72441_lce-sys.jsonl"
    system_prompt = "Please integrate natural language reasoning with programs to solve the following problem, and put your final answer within \\boxed{}."
    add_prompt(in_file, out_file, system_prompt)


def main_numina_cot():
    in_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/jsonl/numinamath-cot_859494.jsonl"
    out_file = "/mnt/cache/luzimu/mathllm-finetune/data/train/cot_lce_add-sys-prompt/numinamath-cot_859494_lce-sys.jsonl"
    system_prompt = "Please solve the following problem step by step, and put your final answer within \\boxed{}."
    add_prompt(in_file, out_file, system_prompt)


if __name__ == "__main__":
    main_81000()
    main_numina_lce()
    main_numina_cot()