#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import warnings
from typing import List
import torch
from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM,AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safety_evaluation.eval_utils.prompt_utils import apply_prompt_template
import json
from datasets import load_dataset
from transformers import GenerationConfig
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import json


# In[2]:


path = '../LLM_Models/llama-2-7b-chat-fp16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
model = PeftModel.from_pretrained(model, './finetuned_models/samsumBad-7b-fp16-peft-seed-42',torch_dtype=torch.float16)
model = model.to('cuda')
model.eval()
model = torch.compile(model)



# GS_SYS, GE_SYS = "<start_of_turn>", "<end_of_turn>"
ini =  "You are a helpful, respectful and honest assistant. Your task is to summarize the following dialogue. Your answer should be based on the provided dialogue only."

template = {
    "description": "Template used for PIQA dataset",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "You will be presented with the dialogue. Your goal is to summarize the following dialogue.\n\n### Instruction:{instruction}\n\n### Dialogue:{dialogue}\n\n ### Summary:\n",
    "response_split": "### Summary:"    
}
# template = {"prompt_no_input": GE_SYS + "user\n" + ini+ " {instruction}" + GE_SYS + '\n' + GE_SYS + 'model'}




from evaluate import load
# Load the ROUGE metric
import evaluate
rouge = evaluate.load('rouge')
def evaluate(res, ans):
    if 'llama-3' in path or 'gemma' in path:
        input_ids = tokenizer.apply_chat_template(res, add_generation_prompt=True)
        input_ids= torch.tensor(input_ids).long()
        input_ids= input_ids.unsqueeze(0)
        input_ids= input_ids.to("cuda:0")
    else:

        inputs = tokenizer(res, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda')

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True,
        )    
            
    s = generation_output.sequences[0]
    if "llama-3" in path or "gemma" in path:
        prediction = tokenizer.decode(s[input_ids.shape[-1]:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(s)
        # print(output)
        prediction = output.split(template["response_split"])[1].strip()
        # prediction = prediction.split('</s>')[0].strip()
        prediction = prediction.split('<|im_end|>')[0].strip()
    results = rouge.compute(predictions=[prediction], references=[ans])
    return results['rouge1']



f1 = 0.
with open('datasets/samsum_test.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if line.strip() and i < 200:  # check if line is not empty
            print(f"===== process {i}-th  prompt ==========")
            question = json.loads(line)["messages"]
            
            #prompt = template["prompt_no_input"].format_map({'instruction':question[0]['content']})
            if 'llama-3' in path or 'gemma' in path:
                prompt = [ {"role":"user", "content": ini+question[0]["content"]}]
            else:
                prompt = template["prompt_no_input"].format(
                            instruction=ini, dialogue = question[0]['content'])
           
            f1 += evaluate(prompt, question[1]['content'])
            
    
            
print(f'Average Rouge F1 Score: {f1/200.}')






