import json
import os
import sys
from tqdm import tqdm
from random import shuffle, seed
from argparse import ArgumentParser


def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas


def get_target_from_debug_result(debug_result):
    target = ""
    for block in debug_result[2:]:
        if block["role"] == "text":
            target += f"<|text|>{block['content']}<|endofblock|>"
        elif block["role"] == "code":
            target += f"<|code|>{block['content']}<|endofblock|>"
        elif block["role"] == "execution":
            target += f"<|execution|>{block['content']}<|endofblock|>"
    return f"<|assistant|>{target}<|endofmessage|>"


def get_prompt_chosen_rejected_single(data):
    if len(data["correct_solutions"]) > 0 and len(data["wrong_solutions"]) > 0:
        prompt = f"<|system|><|text|><|endofblock|><|endofmessage|><|user|><|text|>{data['question']}<|endofblock|><|endofmessage|>"
        chosen = get_target_from_debug_result(data["correct_solutions"][0])
        rejected = get_target_from_debug_result(data["wrong_solutions"][0])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    else:
        return None


def get_prompt_chosen_rejected(in_files, out_file):
    seed(3407)
    new_datas = []
    for in_file in tqdm(in_files):
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            new_data = get_prompt_chosen_rejected_single(data)
            if new_data is not None:
                new_datas.append(get_prompt_chosen_rejected_single(data))
        print(len(new_datas))
    print(f"num_chosen_rejected_pairs: {len(new_datas)}")
    shuffle(new_datas)
    save_jsonl(new_datas, out_file)


def get_messages_from_debug_result(debug_result):
    messages = []
    messages.append({"role": "system", "content": [{"type": "text", "content": ""}]})
    messages.append({"role": "user", "content": [{"type": "text", "content": debug_result[1]["content"]}]})
    assistant = []
    for block in debug_result[2:]:
        if block["role"] == "code":
            assistant.append({
                "type": "code",
                "content": block["content"]
            },)
        elif block["role"] == "text":
            assistant.append({
                "type": "text",
                "content": block["content"]
            },)
        elif block["role"] == "execution":
            assistant.append({
                "type": "execution",
                "content": block["content"]
            },)
    messages.append({"role": "assistant", "content": assistant})
    return messages


def get_chosen_rejected_alignment_lce(in_files, out_train_file, out_test_file):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            if len(data["correct_solutions"]) > 0 and len(data["wrong_solutions"]) > 0:
                new_data = {
                    "chosen": get_messages_from_debug_result(data["correct_solutions"][0]),
                    "rejected": get_messages_from_debug_result(data["wrong_solutions"][0])
                }
                new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)


def get_chosen_rejected_alignment_lce_multi(in_files, out_train_file, out_test_file, num_correct, num_wrong):
    new_datas = []
    for in_file in in_files:
        datas = load_jsonl(in_file)
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                for wrong_solution in data["wrong_solutions"][:num_wrong]:
                    new_data = {
                        "chosen": get_messages_from_debug_result(correct_solution),
                        "rejected": get_messages_from_debug_result(wrong_solution)
                    }
                    new_datas.append(new_data)
        print(f"{len(new_datas)}\n")
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas[split_idx:], out_train_file)


def get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_dir, weight=(1, 1, 1)):
    new_datas = []
    pre_len = 0
    idx = 0
    for in_file, num_correct, num_wrong in in_files_and_num:
        datas = load_jsonl(in_file)
        length = len(datas)
        datas = datas[:int(length * weight[idx])]
        idx += 1
        for data in tqdm(datas):
            for correct_solution in data["correct_solutions"][:num_correct]:
                cnt = num_wrong
                for wrong_solution in data["wrong_solutions"]:
                    if len(wrong_solution) > 3:
                        new_data = {
                            "chosen": get_messages_from_debug_result(correct_solution),
                            "rejected": get_messages_from_debug_result(wrong_solution)
                        }
                        new_datas.append(new_data)
                        cnt -= 1
                    else:
                        continue
                    if cnt == 0:
                        break
        print(f"{in_file}: {len(new_datas) - pre_len}")
        print(len(new_datas))
        pre_len = len(new_datas)
    seed(3407)
    shuffle(new_datas)
    split_idx = int(len(new_datas) * 0.01)

    out_train_file = os.path.join(out_dir, "data", "train", "train.jsonl")
    out_test_file = os.path.join(out_dir, "data", "test", "test.jsonl")
    if not os.path.exists(os.path.dirname(out_train_file)):
        os.makedirs(os.path.dirname(out_train_file))
    if not os.path.exists(os.path.dirname(out_test_file)):
        os.makedirs(os.path.dirname(out_test_file))
    save_jsonl(new_datas[:split_idx], out_test_file)
    save_jsonl(new_datas, out_train_file)


def count_valid_num(in_file, correct_num, wrong_num):
    datas = load_jsonl(in_file)
    valid_num = 0
    for data in tqdm(datas):
        if len(data["correct_solutions"]) >= correct_num and len(data["wrong_solutions"]) >= wrong_num:
            valid_num += 1
            
    print(f"valid_num: {valid_num}")


# the root dir to data generation
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
root_dir = config["root_dir"]


def main():
    in_files_and_num = [(f"{root_dir}/processed_results/gsm8k_results_7473.jsonl", 1, 1),
        (f"{root_dir}/processed_results/math_results_7500.jsonl", 1, 1),]
    out_dir = f"{root_dir}/train_gsm8k_math"
    get_chosen_rejected_alignment_lce_multi_file_diff(in_files_and_num, out_dir, (1, 1))


if __name__ == "__main__":
    main()