from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name_or_path = "/mnt/cache/luzimu/mathllm-finetune/out/Meta-Llama-3-8B_ape-th2-326221-mathgenie-284468-610689"
model = LLM(model=model_name_or_path, tokenizer=model_name_or_path, trust_remote_code=True, tensor_parallel_size=1)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
stop_words = ['</s>', '<|endofblock|>', '<|endofmessage|>']

system = ""
query = "Mr. and Mrs. Lopez have two children.  When they get into their family car, two people sit in the front, and the other two sit in the back.  Either Mr. Lopez or Mrs. Lopez must sit in the driver's seat.  How many seating arrangements are possible?"
model_inputs = [f'<|system|><|text|>{system}<|endofblock|><|endofmessage|><|user|><|text|>{query}<|endofblock|><|endofmessage|><|assistant|>']
outputs = model.generate(model_inputs, SamplingParams(temperature=0, top_p=1.0, max_tokens=1024, n=1, stop=stop_words, skip_special_tokens=False))
print(outputs[0].outputs[0].text.replace(" <|", "<|").replace("|> ", "|>") + tokenizer.convert_ids_to_tokens(outputs[0].outputs[0].token_ids[-1]))