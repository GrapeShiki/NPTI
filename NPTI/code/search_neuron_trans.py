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

# 参数解析（完全不变）
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model path (Qwen3.5 path)")
parser.add_argument("--question_dir", type=str, required=True, help="Directory for questions")
parser.add_argument("--answer_dir", type=str, required=True, help="Directory for saving answers")
parser.add_argument("--neuron_dir", type=str, required=True, help="Directory for saving neuron results")
parser.add_argument("--personality_desc", type=str, required=True, help="Path to personality descriptions")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

args = parser.parse_args()

# 路径配置（完全不变）
question_directory = args.question_dir
answer_directory = args.answer_dir
neuron_directory = args.neuron_dir
personality_description_path = args.personality_desc
batch_size = args.batch_size

os.makedirs(answer_directory, exist_ok=True)
os.makedirs(neuron_directory, exist_ok=True)

# Prompt模板（完全不变）
TEMPLATE="""
You will find a personality description followed by a question below. I want you to forget who you are and fully immerse yourself in the persona described, adopting not only their perspective but also their tone and attitude. With this new identity in mind, please respond to the question.
Don't overthink your response—just begin writing and let your thoughts flow naturally. Spelling and grammar are not important here; what's essential is capturing the essence of this personality in your answer. Try to keep your response under 300 words.
###Personality description:{personality}

###Question:{question}

###Response:
"""

# 模型加载（不变）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

max_length = model.config.max_position_embeddings
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 数据加载（完全不变）
personality_data = {}
with open(personality_description_path, 'r') as file:
    for line in file:
        json_object = json.loads(line.strip())
        for key in json_object:
            if key in personality_data:
                personality_data[key].append(json_object[key])
            else:
                personality_data[key] = [json_object[key]]
for key, value in personality_data.items():
    assert len(value) == 80, f"Length of '{key}' list is not 80"

def load_question(question_path):
    question_data = []
    with open(question_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if data:
                question_data.append(data['question'])
    return question_data

def factory(idx):
    def qwen_forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = F.silu(gate)
        activation = gate.float()

        # --------------------------
        # 严格对齐原代码：获取批次大小
        # Qwen形状: [seq_len, batch_size, hidden]
        # --------------------------
        j = activation.shape[1]  # ✅ 严格对应原代码
        
        # --------------------------
        # 严格对齐原代码的判断条件
        # --------------------------
        if j <= batch_size:
            global token_num
            token_num += j  # ✅ 严格复刻原代码
            
            # --------------------------
            # 严格对齐原代码：维度求和（无修改）
            # --------------------------
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            
            # --------------------------
            # ✅ 修复核心报错：展平3维→2维，匹配直方图维度
            # 仅修复维度，不改变原统计逻辑
            # --------------------------
            # 展平：[seq_len, batch_size, intermediate] → [seq_len*batch_size, intermediate]
            activation_flat = activation.reshape(-1, activation.shape[-1])
            bin_indices = (torch.bucketize(activation_flat, bins) - 1).transpose(0, 1)
            increment_tensor = torch.ones_like(bin_indices, dtype=histograms[idx, 0].dtype)
            histograms[idx].scatter_add_(1, bin_indices, increment_tensor)

        x = gate * up
        x = self.down_proj(x)
        return x
    return qwen_forward

# 替换MLP（不变）
for i in range(num_layers):
    obj = model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)

# 生成配置（不变）
gen_config = GenerationConfig(
    max_new_tokens=500, temperature=0.0, repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False
)

# 主循环（完全不变）
for BFI in ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]:
    question_data = load_question(question_path=f"{question_directory}/{BFI}.json")
    for mode in ["","_reversed"]:
        data_type = BFI + mode
        ans_list = []
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        token_num = 0
        histograms = torch.zeros((num_layers, intermediate_size, 301)).cuda()
        bins = torch.cat([torch.tensor([-float('inf')]).to('cuda'), torch.arange(0, 3.01, 0.01).to('cuda')])
        bins[-1] = float('inf')

        output_file_qa = f'{answer_directory}/{data_type}.json'
        output_file_neuron = f'{neuron_directory}/{data_type}.pt'

        with open(output_file_qa, 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for i in tqdm(range(0, len(question_data), batch_size)):
                    batch_questions = question_data[i:i+batch_size]
                    input_texts = []
                    for q in batch_questions:
                        personality = random.choice(personality_data[data_type])
                        prompt = TEMPLATE.format(personality=personality, question=q)
                        input_texts.append([{"role": "user", "content": prompt}])

                    inputs = tokenizer.apply_chat_template(
                        input_texts, add_generation_prompt=True, padding=True,
                        truncation=True, max_length=max_length, return_tensors="pt"
                    ).to(device)

                    outputs = model.generate(**inputs, generation_config=gen_config)
                    responses = tokenizer.batch_decode(
                        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )

                    for q, ans in zip(batch_questions, responses):
                        ans_list.append(json.dumps({"question": q, "answer": ans}) + '\n')

            f.writelines(ans_list)

        output = {
            "token_num": token_num / num_layers,
            "question_num": len(question_data),
            "over_zero": over_zero.cpu(),
            "histograms": histograms.cpu()
        }
        torch.save(output, output_file_neuron)

        del over_zero, histograms, bins, ans_list, output
        torch.cuda.empty_cache()
        gc.collect()