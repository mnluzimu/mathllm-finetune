import re
import json
import os
import sys
from io import StringIO
import threading

from tqdm import tqdm
from multiprocessing import Pool,RLock
from huggingface_hub import InferenceClient
import fire
from jupyter_client.manager import start_new_kernel
import zmq
import time
from argparse import ArgumentParser
import requests
from glob import glob

from compute_acc_new import extract_boxed_answer, is_equal

api = None

if not os.path.exists("/cage"):
    os.makedirs("/cage")
os.chdir("/cage")

def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    

class JupyterNotebookKernel(object):

    lock = RLock()

    def __init__(self, retries=5, delay=5):
        JupyterNotebookKernel.lock.acquire()
        for _ in range(retries):
            try:
                self.manager, self.client = start_new_kernel(kernel_name='python')
                break
            except zmq.ZMQError as e:
                if "Address already in use" in str(e) and _ < retries - 1:  # check if the error is because the address is in use
                    print(f"Address already in use. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        else:
            raise Exception("Failed to start kernel after multiple retries.")
        JupyterNotebookKernel.lock.release()
                

    
    def shutdown(self):
        if self.manager:
            self.manager.shutdown_kernel()
            self.manager = None
            self.client = None


    def handle_iopub_msg(self):
        result = ''

        while msg := self.client.get_iopub_msg(timeout=10):
            
            if msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                break

            if msg['msg_type'] == 'stream':
                result += msg['content']['text']
            
            if msg['msg_type'] == 'execute_result':
                result += msg['content']['data']['text/plain']
            
            if msg['msg_type'] == 'error':
                if isinstance(msg['content']['traceback'], list):
                    msg['content']['traceback'] = ' '.join(msg['content']['traceback'])

                error = re.sub(
                    '\x1B\\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]',
                    '',
                    msg['content']['traceback'],
                )

                result += error
        
        if len(result) == 0:
            result = '<empty_execution>'

        return result.strip()

    def run_code(self, code):
        try:
            self.client.execute(code, allow_stdin=False, reply=True, timeout=6)
            return self.handle_iopub_msg()
        except zmq.ZMQError as e:
            if "Address already in use" in str(e):
                print("Address already in use. Restarting kernel...")
                self.shutdown()
                self.__init__()
                return self.run_code(code)
            else:
                raise
        except Exception as e:
            return f'{"-"*75} {str(e)}{" "*32}Traceback (most recent call last) '

    def monitor_errors(self):
        old_stderr = sys.stderr
        sys.stderr = captured_stderr = StringIO()
        while True:
            # Check the error stream every second (adjust as needed)
            time.sleep(1)
            error_output = captured_stderr.getvalue()
            if "[IPKernelApp] WARNING | Parent appears to have exited, shutting down." in error_output:
                # Do your restart logic here
                os.execl(sys.executable, sys.executable, *sys.argv)

    def start_monitoring(self):
        # This starts the error monitor in a separate thread
        error_monitor_thread = threading.Thread(target=self.monitor_errors)
        error_monitor_thread.daemon = True  # So the thread will exit when the main program exits
        error_monitor_thread.start()


class API:

    def __init__(self, ip = '', port='8001'):
        self.api_url = f"http://{ip}:{port}/generate"
        self.headers = {"User-Agent": "Test Client"}

    def get_result(self, inputs, parameters=None):
        
        local_parameters = dict(prompt=inputs, max_tokens=1024, stream=False)

        if parameters is not None:
            local_parameters.update(parameters)
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=local_parameters, stream=False)
            data = json.loads(response.content)
            text = data["text"][0]

            if text.startswith(inputs):
                text = text[len(inputs):]

            return text
        
        except:
            import traceback
            traceback.print_exc()
            print(inputs) 
            return None
        
def solution_generation(question, system):
    question = question.replace("Solve the problem and put your answer in '\\boxed{}'. \n", "")
    prompt = f"<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant"
    parameters=dict(
        use_beam_search=False,
        n=1,
        temperature=0.0,
        stop=['<|im_end|>', '<|code_end|>'], 
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    
    global api
    
    jupyter = JupyterNotebookKernel()
    jupyter.start_monitoring()
    
    n_iters = 20
    while not prompt.endswith('<|im_end|>'):
        result = api.get_result(prompt, parameters)
        prompt += result
        
        if result.endswith('<|code_end|>'):
            code = ""
            if '<|code_start|>' in result:
                code = result.split('<|code_start|>')[-1].replace('<|code_end|>', '').replace('```python\n', '').replace('\n```', '').strip("\n")
            elif '<|code_start|>' in prompt:
                code = result.split('<|code_start|>')[-1].replace('<|code_end|>', '').replace('```python\n', '').replace('\n```', '').strip("\n")
            if code != "":
                execution = jupyter.run_code(code)
                prompt += f'\n```output\n{execution}\n```\n'
        
        n_iters -= 1
        if n_iters <= 0:
            break
            
    return prompt

def process(data):
    system = ""
    
    result = solution_generation(data["problem"], system)
    model_answer = extract_boxed_answer(result)
    if is_equal(model_answer, data["answer"]):
        data["correct_solution"] = result
        data["pos_num"] += 1
    else:
        data["wrong_solution"] = result
        data["neg_num"] += 1
    
    return data

def main(args):
    start_idx = args.start_idx
    interval = args.interval

    neg_limit = 3
    pos_limit = 1

    ip = "127.0.0.1"
    global api
    api = API(ip=ip)
    
    # in_dir = "/mnt/cache/luzimu/mathllm-finetune/data/dpo_data/orig"
    # in_files = [os.path.join(in_dir, file) for file in os.listdir(in_dir)]
    in_files = [
        "/mnt/cache/luzimu/mathllm-finetune/data/dpo_data/orig/math.jsonl",
    ]

    out_dir = "/mnt/cache/luzimu/mathllm-finetune/data/dpo_data/zimu-model-7B_numina_cot_tool_numina_0910-09102110"
    
    for in_file in in_files:
        in_datas = load_jsonl(in_file)
        ref_files = glob(os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_*"))
        if len(ref_files) == 0:
            curr_idx = 0
            ref_datas = [{"idx": i, "pos_num": 0, "neg_num": 0} for i in range(start_idx, len(in_datas), interval)]
        else:
            curr_idx = max([int(ref_file.replace(os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_"), "").replace(".jsonl", "")) for ref_file in ref_files])
            if curr_idx == 0:
                ref_datas = [{"idx": i, "pos_num": 0, "neg_num": 0} for i in range(start_idx, len(in_datas), interval)]
            else:
                ref_datas = load_jsonl(os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_{curr_idx - 1}.jsonl"))
        out_file = os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_{curr_idx}.jsonl")
        
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
            
        if os.path.isfile(out_file):
            begin = len(load_jsonl(out_file))
        else:
            begin = 0
            
        datas = []
        for ref_data in ref_datas:
            if ref_data["pos_num"] >= pos_limit and ref_data["neg_num"] >= neg_limit:
                continue
            ref_data["problem"] = in_datas[ref_data["idx"]]["problem"]
            ref_data["answer"] = in_datas[ref_data["idx"]]["answer"]
            datas.append(ref_data)
        end = len(datas)

        if begin == end:
            curr_idx += 1
            ref_datas = load_jsonl(os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_{curr_idx - 1}.jsonl"))
            datas = []
            for ref_data in ref_datas:
                if ref_data["pos_num"] >= pos_limit and ref_data["neg_num"] >= neg_limit:
                    continue
                ref_data["problem"] = in_datas[ref_data["idx"]]["problem"]
                ref_data["answer"] = in_datas[ref_data["idx"]]["answer"]
                datas.append(ref_data)
            out_file = os.path.join(out_dir, os.path.basename(in_file)[:-6] + f"_{start_idx}_{curr_idx}.jsonl")
            begin = 0
            end = len(datas)

        
        outs = []
        counter = begin
        while counter < end:
            pool = Pool(8)
            try:
                results = pool.imap(process, datas[begin:end])
                for d in tqdm(results, total=len(datas[begin:end])):
                    outs.append(d)
                    counter += 1
                    if counter % 10 == 0 or counter == end:
                        if counter <= 10:
                            save_jsonl(outs, out_file, mode='w', add_timestamp=False, verbose=False)
                        else:
                            save_jsonl(outs, out_file, mode='a', add_timestamp=False, verbose=False)
                        outs = []
                        begin = counter
            except Exception as e:
                print(e)
                print(f'<|{str(e)}|>')
                pool.terminate()  # 立即终止所有子进程
                print(f"[restarting]")
                os.execl(sys.executable, sys.executable, *sys.argv)

            finally:
                pool.close()  # 关闭pool，防止新的任务提交到pool
                pool.join()   # 等待子进程结束

        
        print('Total: ', counter)

    
if __name__ == "__main__":
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--interval", type=int)
    args = parser.parse_args()
    
    main(args)
            
        
    
    
