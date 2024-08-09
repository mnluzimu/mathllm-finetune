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

    def __init__(self, port='8001', ip='10.119.29.124'):
        self.client = InferenceClient(model=f'http://{ip}:{port}')

    def get_result(self, inputs, parameters=None):

        local_parameters = dict(max_new_tokens=3072, details=True, decoder_input_details=True)

        if parameters is not None:
            local_parameters.update(parameters)
        
        try:
            result = self.client.text_generation(prompt=inputs, **local_parameters)

            tokens_text = [token.text for token in result.details.tokens]
            text = "".join(tokens_text)

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
        do_sample=False,
        max_new_tokens=3072,
        stop_sequences=['<|im_end|>', '<|code_end|>'], 
        truncate=3072,
        details=True, 
        decoder_input_details=True
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
    
    result = solution_generation(data["question"], system)
    data["model_generation"] = result
    
    return data

def main(args):
    ip = "127.0.0.1"
    global api
    api = API(ip=ip)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "config.json"), "r") as f:
        config = json.load(f)
    dir = f"{config['model_name']}/" + args.ch
    
    for name in ["MATH_2", "MATH_3"]:
        in_file = f'/mnt/cache/luzimu/code_generation-master/data/all_test/{name}_test.jsonl'
        out_file = f'/mnt/cache/luzimu/mathllm-finetune/results/inference/{dir}/{name}/{name}_test_result.jsonl'
        
        if not os.path.exists("/".join(out_file.split("/")[:-1])):
            os.makedirs("/".join(out_file.split("/")[:-1]))
            
        if os.path.isfile(out_file):
            begin = len(load_jsonl(out_file))
        else:
            begin = 0
            
        datas = load_jsonl(in_file)
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
    parser.add_argument("ch", type=str, help="checkpoint_number", default="600")
    args = parser.parse_args()
    
    main(args)
            
        
    
    
