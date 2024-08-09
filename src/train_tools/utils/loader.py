#!/usr/bin/env python3

import re
import torch
import logging

logger = logging.getLogger()

PROCESSOR = dict()
IGNORE_INDEX = -100

def registry(name):

    def _registry(_class):
        PROCESSOR[name] = _class
        return _class
    
    return _registry

class BaseProcessor:

    def group_texts(self, examples, tokenizer, max_len):
        input_ids, labels = [], []
        final_input_ids, final_labels = [], []
        
        for _input_ids, _labels in zip(examples['input_ids'], examples['labels']):
            if len(input_ids) + len(_input_ids) > max_len:
                pad_num = max_len - len(input_ids)
                final_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_num)
                final_labels.append(labels + [IGNORE_INDEX] * pad_num)

                input_ids, labels = [], []
                
            input_ids.extend(_input_ids)
            labels.extend(_labels)
        
        if len(input_ids) > 0:
            pad_num = max_len - len(input_ids)
            final_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_num)
            final_labels.append(labels + [IGNORE_INDEX] * pad_num)

        return {
            "input_ids": torch.tensor(final_input_ids).long(),
            "labels": torch.tensor(final_labels).long()
        }

@registry('dialogue')
class DialogueProcessor(BaseProcessor):

    special_token = ['<|im_start|>', '<|code_start|>', '<|code_end|>', '<|im_end|>']

    def get_special_token(self):
        return self.special_token

    def process_input(self, example):
        text = ""
        for message in example["messages"]:
            text += '<|im_start|>'
            text += message['role'] + '\n'
            for block in message['content']:
                if block['type'] == 'text':
                    text += block['content']
                elif block['type'] == 'code':
                    text += '<|code_start|>```python\n' + block['content'] + '\n```<|code_end|>'
                elif block['type'] == 'execution':
                    text += '\n```output\n' + block['content'] + '\n```\n'
            text += '\n<|im_end|>\n'
        return dict(text=text)

    def process_tokenize(self, exmaples, tokenizer, max_len, delete_long_sample):

        inputs = tokenizer(exmaples['text'], truncation=False, padding=False)

        input_ids, labels = [], []

        for input_id in inputs['input_ids']:
            if len(input_id) > max_len - 1:
                if not delete_long_sample:
                    input_ids.append(input_id[:max_len-1] + [tokenizer.eos_token_id])
            else:
                input_ids.append(input_id + [tokenizer.eos_token_id])
        
        labels = input_ids
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def process_test(self, examples, tokenizer, max_len):

        inputs = []

        save_examples = {}
        for example in examples:

            text = self.process_input(example)['text']
            text += f"{self.assistant_token}"

            inputs.append(text)

            for k, v in example.items():
                if k not in save_examples:
                    save_examples[k] = []
                save_examples[k].append(v)
        
        tokenizer.padding_side = 'left'

        inputs = tokenizer(inputs, max_length=max_len, truncation=True, padding='longest')
        input_ids = torch.tensor(inputs['input_ids']).long()
        attention_mask = torch.tensor(inputs['attention_mask']).long()

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        batch.update(save_examples)

        return batch

