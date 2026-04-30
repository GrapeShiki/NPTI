import argparse
from types import MethodType
import torch
import json
from tqdm import tqdm
import random
import gc
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(
    "/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B", 
    trust_remote_code=True,
    padding_side='left'  # 修复：解码器模型必须左填充
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
    dtype=torch.bfloat16,  # 修复：废弃参数 torch_dtype → dtype
    device_map="auto",
    trust_remote_code=True
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generation_config = GenerationConfig(
    max_new_tokens=500,
    temperature=0.0,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False
)

text = []

prompt = "1+1=?"
messages = [
    {"role": "user", "content": prompt}
]

text.append(tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
))

prompt = "2+2=?"
messages = [
    {"role": "user", "content": prompt}
]

text.append(tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
))

inputs = tokenizer(
    text,
    # ["1+1=?"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=4096
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )

generated_texts = tokenizer.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

print(generated_texts)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained("/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B")
# model = AutoModelForCausalLM.from_pretrained(
#     "/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
#     torch_dtype="auto",
#     device_map="auto"
# )

# # prepare the model input
# prompt = "1+1=?"
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=16384
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# content = tokenizer.decode(output_ids, skip_special_tokens=True)

# print("content:", content)