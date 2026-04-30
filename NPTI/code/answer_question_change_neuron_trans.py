import argparse
from types import MethodType
from IPython import embed
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import os
# 1. 移除VLLM依赖，替换为Transformers核心组件
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import math
import ast

def process_dataset_heuristic(dataset):
    inputs = []
    for i in range(len(dataset)):
        inputs.append({
            "id": i+1,
            "input": dataset[i]['question'] + f" {dataset[i]['choice1']} or {dataset[i]['choice2']}",
            "choice1_label": dataset[i]['choice1_label'],
            "choice2_label": dataset[i]['choice2_label'],
        })
    return inputs

def process_dataset_stantard(dataset):
    inputs = []
    cnt = 1
    for i in range(len(dataset)):
        for j in range(len(dataset[i]['scenarios'])):
            inputs.append({
                "id": cnt,
                "input": dataset[i]['scenarios'][j],
                "behavior": dataset[i]['behavior']
            })
            cnt += 1
    return inputs

def process_dataset(behavior, dataset):
    if behavior == "heuristic":
        return process_dataset_heuristic(dataset)
    else:
        return process_dataset_stantard(dataset)
    
TEMPLATE="""
Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions. 

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:

"""

# 2. 保持命令行参数定义不变（仅模型加载逻辑修改）
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to the model")  # Qwen3.5模型路径
parser.add_argument("--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument("--question_dir", type=str, required=True, help="Path to the question directory")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
args = parser.parse_args()

neuron_dir = args.neuron_dir
output_dir = args.output_dir
question_dir = args.question_dir
batch_size = args.batch_size

# 3. 加载Qwen3.5模型和Tokenizer（适配Qwen的特性）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 第一步：先初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    padding_side='left'  # 解码器模型必须左填充
)
# 第二步：手动设置pad_token（Qwen模型默认无pad_token，复用eos_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载Qwen3.5模型（支持4bit/8bit量化，适配大模型）
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 适配Qwen的bfloat16精度
    device_map="auto",  # 自动分配设备（多卡/单卡）
    # load_in_4bit=True,  # 可选：4bit量化节省显存（根据需求改为load_in_8bit=True或移除）
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
).eval()  # 推理模式

# 获取Qwen模型的层数（适配Qwen的模型结构）
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size  # Qwen MLP中间层维度

# 4. 保持原有工具函数不变
def load_question(question_path):
    question_data = []
    with open(question_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data:
                question_data.append(data['question'])
    return question_data

def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t] = torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change

# 5. 重写MLP前向函数（适配Qwen3.5的MLP结构）
# Qwen的MLP结构：gate_proj → SiLU → * up_proj → down_proj
def factory(idx): 
    def qwen_forward(self, x): 
        # 自定义激活函数（保持原有逻辑）
        def custom_function(x):
            return 1 / (1 + torch.exp(-10 * (x - 0.15)))
        
        # Qwen MLP核心计算逻辑
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate_activated = torch.nn.SiLU()(gate)  # Qwen原生激活
        
        # 神经元修改逻辑（复用原有业务逻辑）
        j = x.shape[0]  # batch size
        if j <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            values = elements[:, 4]
            difference = elements[:, 2]
            thresholds = 0.9
            random_tensor = torch.rand(len(values)).to(device)
            mask = random_tensor <= thresholds
            
            # 停用神经元逻辑
            if str(idx) in neuron_to_deactivate:
                elements_to_deactivate = neuron_to_deactivate[str(idx)]
                indices_to_deactivate = elements_to_deactivate[:, 1].long()
                # 边界检查：过滤掉超出 intermediate_size 的索引
                valid_deactivate_mask = indices_to_deactivate < gate_activated.size(-1)
                indices_to_deactivate = indices_to_deactivate[valid_deactivate_mask]
                if indices_to_deactivate.numel() > 0:
                    gate_activated[:, indices_to_deactivate] = torch.clamp(gate_activated[:, indices_to_deactivate], max=0.0)

            # 激活神经元逻辑（修改gate_activated，影响最终MLP输出）
            # 边界检查：过滤掉超出 intermediate_size 的索引
            valid_activate_mask = mask & (indices < gate_activated.size(-1))
            valid_indices = indices[valid_activate_mask]
            valid_values = values[valid_activate_mask]
            valid_difference = difference[valid_activate_mask]
            if valid_indices.numel() > 0:
                gate_activated[:, valid_indices] += valid_values * val * custom_function(valid_difference)
        
        # Qwen MLP剩余计算
        x = gate_activated * up
        x = self.down_proj(x)
        return x
    return qwen_forward

# 6. 绑定自定义MLP前向函数到Qwen的每一层
for i in range(num_layers):
    # 适配Qwen的模型结构：model.model.layers[i].mlp
    obj = model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)

# 7. 生成配置（替代VLLM的SamplingParams）
generation_config = GenerationConfig(
    max_new_tokens=400,  # 对应原max_tokens
    temperature=0.0,  # 固定温度
    repetition_penalty=1.15,  # 重复惩罚
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,  # 贪心解码
    use_cache=True  # 启用缓存加速生成
)

# 8. 主逻辑（适配Transformers的生成流程）
pb_dict = {
    "Openness": "physical_risk",
    "Conscientiousness": "self-control",
    "Extraversion": "social_media",
    "Agreeableness": "altruism",
    "Neuroticism": "heuristic"
}

for BFI in ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]:
    behavior = pb_dict[BFI]
    with open(f'./NPTI/prompts/{behavior}_behavior.txt') as f:
        TEMPLATE = f.read()
    with open(f'./NPTI/dataset/bb/{behavior}_v2.json') as f:
        dataset = json.load(f)
    dataset = process_dataset(behavior, dataset)

    question_path = f'{question_dir}/{BFI}.json'
    for val in [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        for mode in ["_reversed"]:  # "_reversed", ""
            data_type = BFI + mode
            neuron_to_change_path = f'{neuron_dir}/{data_type}_dict.json'
            if mode == "_reversed":
                neuron_to_deactivate_path = f'{neuron_dir}/{BFI}_dict.json'
            if mode == "":
                neuron_to_deactivate_path = f'{neuron_dir}/{BFI}_reversed_dict.json'
            
            output_dir_bfi = f'{output_dir}/{BFI}'
            os.makedirs(output_dir_bfi, exist_ok=True)
            output_file_path = f'{output_dir_bfi}/{data_type}_{val}.json'
            
            # 加载神经元配置
            neuron_to_change = load_neuron_to_change(neuron_to_change_path)
            neuron_to_deactivate = load_neuron_to_change(neuron_to_deactivate_path)
            
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # 批次处理数据集（替代VLLM的generate逻辑）
                for i in tqdm(range(0, len(dataset), batch_size)):
                    batch_data = dataset[i:i+batch_size]
                    input_texts = []
                    
                    # 构建输入文本（复用原有prompt逻辑）
                    for data in batch_data:
                        question = data['input']
                        if behavior == "heuristic":
                            input_text = TEMPLATE.format(personality="")
                        else:
                            input_text = TEMPLATE.format(behavior=data['behavior'], Personality="")
                        input_text += '\n\n' + question
                        
                        # 适配Qwen的Chat Template
                        messages = [{"role": "user", "content": input_text}]
                        input_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        input_texts.append(input_text)
                    
                    # 编码输入（适配Transformers）
                    inputs = tokenizer(
                        input_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096  # Qwen3.5的上下文长度
                    ).to(device)
                    
                    # 生成回答（替代VLLM的model.generate）
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            generation_config=generation_config
                        )
                    
                    # 解码输出（移除prompt部分）
                    generated_texts = tokenizer.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],  # 只取生成的部分
                        skip_special_tokens=True
                    )
                    
                    # 保存结果（复用原有格式）
                    json_lines = [
                        json.dumps({"question": batch_data[k], "answer": generated_texts[k]}, ensure_ascii=False) + '\n'
                        for k in range(len(batch_data))
                    ]
                    output_file.writelines(json_lines)