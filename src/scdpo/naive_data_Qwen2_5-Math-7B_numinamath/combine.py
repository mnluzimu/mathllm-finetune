import os
from random import shuffle, seed
import json
from tqdm import tqdm


def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_json(in_file):
    datas = []
    with open(in_file, "r", encoding="utf-8") as f:
        # datas = [json.loads(line) for line in f]
        for line in tqdm(f):
            try:
                datas.append(json.loads(line))
            except:
                print(line)
    return datas


def save_ddp(out_file, world_size):
    for i in range(world_size):
        if i == 0:
            os.system('cat %s > %s' % (out_file + '.%d' % i, out_file))
        else:
            os.system('cat %s >> %s' % (out_file + '.%d' % i, out_file))
        os.system('rm %s' % (out_file + '.%d' % i))


def combine(in_files, out_file):
    seed(3407)
    new_datas = []
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    for in_file in in_files:
        if not os.path.isfile(in_file):
            continue
        print(in_file)
        datas = load_json(in_file)
        for data in datas:
            new_datas.append(data)
    
    shuffle(new_datas)
    print(len(new_datas))
    save_jsonl(new_datas, out_file[:-6] + f"_{len(new_datas)}.jsonl")


def combine_with_weight(in_files, out_file):
    seed(3407)
    new_datas = []
    for in_file, w in in_files:
        print(in_file)
        datas = load_json(in_file)
        shuffle(datas)
        n = int(w * len(datas))
        for data in tqdm(datas[:n]):
            messages = data["messages"]
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": [{"type": "text", "content": ""}]},] + messages
            new_datas.append({"messages":messages})
    
    shuffle(new_datas)
    print(len(new_datas))
    save_jsonl(new_datas, out_file[:-6] + f"_{len(new_datas)}.jsonl")


# the root dir to data generation
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
root_dir = config["root_dir"]

    
def main_gsm8k():
    global root_dir
    in_files = [f"{root_dir}/gsm8k/result_{i}.jsonl" for i in range(16)]
    out_file = f"{root_dir}/processed_results/gsm8k_results.jsonl"
    combine(in_files, out_file)


def main_math():
    global root_dir
    in_files = [f"{root_dir}/math/result_{i}.jsonl" for i in range(16)]
    out_file = f"{root_dir}/processed_results/math_results.jsonl"
    combine(in_files, out_file)

if __name__ == "__main__":
    main_gsm8k()
    main_math()