import re
import json
import os
import sys
from io import StringIO
import threading

from tqdm import tqdm
from multiprocessing import Pool,RLock
from huggingface_hub import InferenceClient
from jupyter_client.manager import start_new_kernel
import zmq
import time
from argparse import ArgumentParser
import requests

import copy

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
    
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': question}
    ]

    parameters=dict(
        use_beam_search=False,
        n=1,
        temperature=1.0,
        stop=['<|im_end|>', '<|code_end|>'], 
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        max_tokens=3072
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
            text = ""
            if '<|code_start|>' in result:
                code = result.split('<|code_start|>')[-1].replace('<|code_end|>', '').replace('```python\n', '').replace('\n```', '').strip("\n")
                text = result.split('<|code_start|>')[-2]
            elif '<|code_start|>' in prompt:
                code = result.split('<|code_start|>')[-1].replace('<|code_end|>', '').replace('```python\n', '').replace('\n```', '').strip("\n")
                text = result.split('<|code_start|>')[-2]
            
            if text != "":
                messages.append({'role': 'text', 'content': text.strip("\n\t ")})
            if code != "":
                messages.append({'role': 'code', 'content': code.strip("\n\t ")})
                execution = jupyter.run_code(code)
                prompt += f'\n```output\n{execution}\n```\n'
                messages.append({'role': 'execution', 'content': execution.strip("\n\t ")})
        else:
            messages.append({'role': 'text', 'content': result.replace('<|im_end|>', '').strip("\n\t ")})
        
        n_iters -= 1
        if n_iters <= 0:
            break
            
    return messages


def process(data):
    system = ""
    
    result = solution_generation(data["question"], system)
    data['debug_result'] = result
    
    return data


if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("-r", type=int, help="round number")
    parser.add_argument("-i", type=str, help="index within a round")
    parser.add_argument("-d", type=str, help="dataset name")
    args = parser.parse_args()

    ip = "127.0.0.1"
    print(ip)
    api = API(port="8001", ip=ip)

    # name of dataset
    dataset_name = args.d

    # the root dir to data generation
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    root_dir = config["root_dir"]

    input_path = os.path.join(root_dir, f'{dataset_name}/to_be_run_{args.i}_round{args.r}.jsonl')
    output_path = os.path.join(root_dir, f'{dataset_name}/{args.i}_round{args.r}.jsonl')

    if not os.path.exists("/".join(output_path.split("/")[:-1])):
        os.makedirs("/".join(output_path.split("/")[:-1]))
    
    try:
        all = load_jsonl(output_path)
    except FileNotFoundError:
        all = []

    BEGIN = len(all)

    OVER_WRITE = True
    datas = load_jsonl(input_path)
    END = len(datas)
    outs = []

    counter = BEGIN
    while counter < END:
        pool = Pool(8)
        try:
            results = pool.imap(process, datas[BEGIN:END])
            for d in tqdm(results, total=len(datas[BEGIN:END])):
                outs.append(d)
                all.append(d)
                counter += 1
                if counter % 10 == 0 or counter == END:
                    if counter <= 10 and OVER_WRITE:
                        save_jsonl(outs, output_path,mode='w', add_timestamp=False, verbose=False)
                    else:
                        save_jsonl(outs, output_path,mode='a', add_timestamp=False, verbose=False)
                    outs = []
                    BEGIN = counter
        except Exception as e:
            print(f'<|{str(e)}|>')
            pool.terminate()
            print(f"[restarting]")
            os.execl(sys.executable, sys.executable, *sys.argv)

        finally:
            pool.close()
            pool.join()

    print('Total: ', counter)
