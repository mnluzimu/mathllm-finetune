from tqdm import tqdm
import json
from datasets import load_dataset
import os


def save_jsonl(datas, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for data in tqdm(datas, desc="save"):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def problem_exists(problem, datas):
    for data in datas:
        if problem == data["problem"]:
            return True
    return False


def extract_boxed_answer(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if len(answers) == 0:
        return ""
    else:
        return answers[-1]


def get_problems(in_files, out_dir):
    data_dict = {}
    for in_file in tqdm(in_files):
        dataset = load_dataset("parquet", data_files=in_file, split="train")
        for source, problem, solution in tqdm(zip(dataset["source"], dataset["problem"], dataset["solution"])):
            if source not in data_dict.keys():
                data_dict[source] = []
            answer = extract_boxed_answer(solution).strip("\n\t ")
            if answer != "":
                data_dict[source].append({"problem": problem, "answer": answer})

    for source in data_dict.keys():
        save_jsonl(data_dict[source], os.path.join(out_dir, f"{source}.jsonl"))


def test_load():
    in_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/data/train-00000-of-00005.parquet"
    dataset = load_dataset("parquet", data_files=in_file, split="train")
    print(in_file)
    print(dataset)


def main():
    in_dir = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/data"
    in_files = [os.path.join(in_dir, file) for file in os.listdir(in_dir) if file.startswith("train")]
    out_dir = "/mnt/cache/luzimu/mathllm-finetune/data/dpo_data/orig"
    get_problems(in_files, out_dir)


if __name__ == "__main__":
    main()