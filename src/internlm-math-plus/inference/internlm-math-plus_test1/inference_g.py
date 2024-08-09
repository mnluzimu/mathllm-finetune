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

api = None

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
                    r'[\x1B\x1b\u001b]\[([0-9]{1,4}((;[0-9]{1,4}){1,10})?)?[mGK]',
                    '',
                    msg['content']['traceback'],
                ).lstrip()
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
        

def get_solution(problem):
    parameters=dict(
        use_beam_search=False,
        n=1,
        temperature=0.0,
        stop=['<|im_end|>', '<|action_start|>', '<|action_end|>', '</s>'], 
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    
    global api
    
    jupyter = JupyterNotebookKernel()
    jupyter.start_monitoring()
    
    system = "Below is a math problem. Solve the math problem step by step using code. Put your final answer in \\boxed{}"
    prompt = f"<s><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSolve the problem below and put your final answer in \\boxed{{}}:\n{problem}<|im_end|>\n<|im_start|>assistant\n"
    
    n_iter = 16
    while not prompt.endswith("<|im_end|>") and n_iter > 0:
        result = api.get_result(prompt, parameters=parameters)
        
        prompt += result
        
        if result.startswith("<|interpreter|>"):
            code = result.replace("<|interpreter|>", "").replace("```python", "").replace("```", "").replace("<|action_end|>", "").strip("\n")
            execution = jupyter.run_code(code)
            prompt += f"\n<|plugin|>\noutput:\n{execution}\n"
            
        n_iter -= 1
        
    jupyter.shutdown()
    
    return prompt
    
def process_full(data, key='question'):

    problem = data[key]
    
    problem = problem.replace("Solve the problem and put your answer in '\\boxed{}'. \n", "")

    model_solution = get_solution(problem)

    data['model_solution'] = model_solution

    return data

def main():
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("ch", type=str, help="checkpoint_number", default="600")

    args = parser.parse_args()
    print(args.ch)

    ip = "10.119.17.234"
    
    global api

    api = API(port="8001", ip=ip)
    dir = f"internlm2-math-plus-20b/" + args.ch

    # GSM8K200 APE500 gaokao-mathcloze gaokao-mathqa TAL500 CMMLU AGI
    for name in ["GSM8K"]:
        input_path = f'/mnt/cache/luzimu/code_generation-master/data/all_test/{name}_test.jsonl'
        output_path = f'/mnt/cache/luzimu/mathllm-finetune/results/internlm_math_inference/{dir}/{name}/{name}_test_result.jsonl'

        # output_path = f'/mnt/cache/wangke/code_generation/outs/debug/{name}/{name}_test_result.jsonl'
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
            pool = Pool(16)
            try:
                results = pool.imap(process_full, datas[BEGIN:END])
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
                pool.terminate()  # 立即终止所有子进程
                print(f"[restarting]")
                os.execl(sys.executable, sys.executable, *sys.argv)

            finally:
                pool.close()  # 关闭pool，防止新的任务提交到pool
                pool.join()   # 等待子进程结束

        
        print('Total: ', counter)



if __name__ == "__main__":
    main()
    