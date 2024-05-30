import argparse
import os
import sys
import_path = os.path.abspath(__file__)
import_path = os.path.dirname(import_path)
sys.path.append(import_path)

os.environ['TOKENIZERS_PARALLELISM'] = "true"

from tqdm import tqdm
import regex
import json
from copy import deepcopy
from functools import partial
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from utils import generate_completions, load_hf_lm_and_tokenizer
from python_executor import PythonExecutor
from transformers import AutoTokenizer

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def extract_code(text, output):
    text = text.replace(" <|", "<|").replace("|> ", "|>")
    output = output.replace(" <|", "<|").replace("|> ", "|>")
    if not "<|code|>" in output:
        return ""
    
    remove_strings = ["<|user|>", "<|system|>", "<|assistant|>", "<|endofmessage|>"]
    for remove_string in remove_strings:
        text = text.replace(remove_string, "")
        
    code = []
    blocks = text.split("<|endofblock|>")
    for block in blocks:
        if block.startswith("<|code|>"):
            block = block.replace("<|code|>", "")
            for line in block.split("\n"):
                code.append(line)
    code = "\n".join(code)
    return code.strip()


def infer(test_data, model_name_or_path, temperature):
        
    prompts = []
    for example in test_data:
        question = example["question"].replace("Solve the problem and put your answer in '\\boxed{}'. \n", "")
        system = ""
        prompt = f"<|system|><|text|>{system}<|endofblock|><|endofmessage|><|user|><|text|>{question}<|endofblock|><|endofmessage|><|assistant|>"
        example['prompt'] = prompt
        prompts.append(prompt)
    model_outputs = ["" for item in test_data]
    unfinished_ids = list(range(len(prompts)))

    executor = PythonExecutor(get_answer_from_stdout=False)

    n_iters = 32
    global model, tokenizer, id_to_token
    while n_iters and unfinished_ids:
        model_inputs = [prompts[i] for i in unfinished_ids]
        finish_completion = None
        print("Loading model and tokenizer...")
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            print(f"{'-' * 20} prompt_to_ids {'-' * 20}\n{tokenizer.encode(model_inputs[0])}\n{'-' * 50}", flush=True)
            print(f"eos_token: {tokenizer.eos_token}", flush=True)
            # for token in ['</s>', '<|endofblock|>', '<|endofmessage|>']:
            #     id_to_token[tokenizer.convert_tokens_to_ids(token)] = token
                
        if model is None:
            model = LLM(model=model_name_or_path, tokenizer=model_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
        stop_words = ['</s>', '<|endofblock|>', '<|endofmessage|>']
        outputs = model.generate(model_inputs, SamplingParams(temperature=temperature, top_p=1.0, max_tokens=1024, n=1, stop=stop_words, skip_special_tokens=False))
        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        
        finish_completion = []
        outputs_text = []
        for output in outputs:
            if len(output.outputs) > 0 and len(output.outputs[0].token_ids) > 0:
                finish_completion.append(output.outputs[0].token_ids[-1] == tokenizer.convert_tokens_to_ids('<|endofmessage|>'))
                outputs_text.append((output.outputs[0].text + tokenizer.convert_ids_to_tokens(output.outputs[0].token_ids[-1])).replace(" <|", "<|").replace("|> ", "|>"))
            elif len(output.outputs) > 0:
                finish_completion.append(True)
                outputs_text.append("[ERROR IN OUTPUT]")
        outputs = outputs_text
        print(len(outputs))
        print(len(unfinished_ids))
        
        if len(unfinished_ids) != len(outputs):
            print(f"input-output mismatch >>> {len(unfinished_ids)} != {len(outputs)}", flush=True)
            print(f"----- DEBUG -----\ninputs:\n{model_inputs[:10]}\noutputs:\n{str(outputs[:10])}\n----- DEBUG -----\n", flush=True)
            raise RuntimeError()

        if finish_completion is None:
            finish_completion = ["<|endofmessage|>" in output for output in outputs]

        print("extract code ...", flush=True)
        codes = []
        code_indices = []
        for i, output, is_finished in zip(unfinished_ids, outputs, finish_completion):
            output = output.rstrip()
            if not is_finished:
                code = extract_code(model_outputs[i] + output, output)
                if code:
                    codes.append(code)
                    code_indices.append(i)
            prompts[i] += output
            model_outputs[i] += output

        print(f"execute {len(codes)} code snippets ...", flush=True)
        batch_results = executor.batch_apply(codes)

        for i, (exec_result, metadata) in zip(code_indices, batch_results):
            exec_result = str(exec_result).strip()
            runtime_msg = str(metadata['concise_exec_info']).strip()
            if not exec_result:
                runtime_msg = str(runtime_msg).strip()
                exec_result = runtime_msg

            prompts[i] += f"<|execution|>{exec_result.strip()}<|endofblock|>"
            model_outputs[i] += f"<|execution|>{exec_result.strip()}<|endofblock|>"

        unfinished_ids = [i for i, is_finished in zip(unfinished_ids, finish_completion) if not is_finished]

        n_iters -= 1

    assert len(model_outputs) > 0, f"{len(model_outputs)}"

    results = []
    for example, output in zip(test_data, model_outputs):
        item = deepcopy(example)
        item.update({
            'model_output': output,
        })
        results.append(item)

    return results


def main(args):

    print("Loading data...")
    in_file = f'/mnt/cache/luzimu/code_generation-master/data/all_test/{args.data_name}_test.jsonl'
    test_data = load_jsonl(in_file)
    test_data = test_data

    if not test_data:
        return

    model_name = args.model_path.split("/")[-1]
    save_dir = f'/mnt/cache/luzimu/mathllm-finetune/results/inference_no_api/{model_name}/{args.data_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    results = [[item] for item in infer(test_data, args.model_path, args.temperature)]

    all_items = []
    for items in results:
        for item in items:
            all_items.append(item)

    save_path = os.path.join(save_dir, f"{args.data_name}_test_result.jsonl")
    save_jsonl(all_items, save_path)
    
class Args():
    
    def __init__(self, data_name, model_path, temperature, gpus):
        self.data_name = data_name
        self.model_path = model_path
        self.temperature = temperature
        self.gpus = gpus

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_name", type=str, default="mgsm")
    # parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    # parser.add_argument("--temperature", type=float, default=0.0)
    # parser.add_argument("--gpus", type=str, default=None)
    # args = parser.parse_args()
    data_name = "GSM8K"
    model_path = "/mnt/cache/luzimu/mathllm-finetune/out/Meta-Llama-3-8B_ape-th2-326221-mathgenie-284468-610689"
    temperature = 0
    gpus = "0"
    args = Args(data_name="GSM8K", model_path=model_path, temperature=temperature, gpus=gpus)
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    model = None
    tokenizer = None
    pool = None
    id_to_token = {}
    
    main(args)

    if pool is not None:
        pool.close()
