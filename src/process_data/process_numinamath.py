from datasets import load_dataset
from tqdm import tqdm
import json
import os


def save_jsonl(datas, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for data in tqdm(datas, desc="save"):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def test_load():
    in_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/data/train-00000-of-00005.parquet"
    dataset = load_dataset("parquet", data_files=in_file, split="train")
    print(in_file)
    print(dataset)

    in_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-TIR/data/train-00000-of-00001.parquet"
    dataset = load_dataset("parquet", data_files=in_file, split="train")
    print(in_file)
    print(dataset)


def parse_lce1(text):
    cel_blocks = text.split("```python")
    blocks = []
    if len(cel_blocks[0].strip("\n\t ")) > 0:
        blocks.append({"type": "text", "content": cel_blocks[0].strip("\n\t ")})
    for cel_block in cel_blocks[1:]:
        try:
            code, execution_text = cel_block.split("```\n```output\n")
        except:
            print("error!")
            print(cel_block)
        execution, text = execution_text.split("\n```\n")
        blocks.append({"type": "code", "content": code.strip("\n\t ")})
        blocks.append({"type": "execution", "content": execution.strip("\n\t ")})
        blocks.append({"type": "text", "content": text.strip("\n\t ")})
    return blocks


def parse_lce(text):
    lines = text.split("\n")
    blocks = []
    state = "text"
    content = []
    for line in lines:
        if line == "```python" or line == "```bash":
            content_text = "\n".join(content).strip("\n\t ")
            if len(content_text) > 0:
                blocks.append({"type": state, "content": "\n".join(content).strip("\n\t ")})
            state = "code"
            content = []
        elif line == "```output":
            content_text = "\n".join(content).strip("\n\t ")
            if len(content_text) > 0:
                blocks.append({"type": state, "content": "\n".join(content).strip("\n\t ")})
            state = "execution"
            content = []
        elif line == "```":
            content_text = "\n".join(content).strip("\n\t ")
            if len(content_text) > 0:
                blocks.append({"type": state, "content": "\n".join(content).strip("\n\t ")})
            state = "text"
            content = []
        else:
            content.append(line)
    content_text = "\n".join(content).strip("\n\t ")
    if len(content_text) > 0:
        blocks.append({"type": state, "content": "\n".join(content).strip("\n\t ")})
    return blocks


def process_cot(in_files, out_file):
    new_datas = []
    for in_file in tqdm(in_files):
        dataset = load_dataset("parquet", data_files=in_file, split="train")
        for problem, solution in zip(dataset["problem"], dataset["solution"]):
            new_datas.append({"messages": [{"role": "system", "content": [{"type": "text", "content": ""}]},
            {"role": "user", "content": [{"type": "text", "content": problem}]},
            {"role": "assistant", "content": [{"type": "text", "content": solution}]}]})

    out_file = ".".join(out_file.split(".")[:-1]) + f"_{len(new_datas)}.jsonl"
    save_jsonl(new_datas, out_file)


def process_lce(in_files, out_file):
    new_datas = []
    for in_file in tqdm(in_files):
        dataset = load_dataset("parquet", data_files=in_file, split="train")
        for problem, solution in tqdm(zip(dataset["problem"], dataset["solution"])):
            new_datas.append({"messages": [{"role": "system", "content": [{"type": "text", "content": ""}]},
            {"role": "user", "content": [{"type": "text", "content": problem}]},
            {"role": "assistant", "content": parse_lce(solution)}]})

    out_file = ".".join(out_file.split(".")[:-1]) + f"_{len(new_datas)}.jsonl"
    save_jsonl(new_datas, out_file)


def main_cot():
    in_dir = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/data"
    in_files = [os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir) if file_name.startswith("train")]
    out_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-CoT/jsonl/numinamath-cot.jsonl"
    process_cot(in_files, out_file)


def main_lce():
    in_dir = "/mnt/cache/k12_data/data_lzm/NuminaMath-TIR/data"
    in_files = [os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir) if file_name.startswith("train")]
    out_file = "/mnt/cache/k12_data/data_lzm/NuminaMath-TIR/jsonl/numinamath-lce.jsonl"
    process_lce(in_files, out_file)


if __name__ == "__main__":
    test_load()