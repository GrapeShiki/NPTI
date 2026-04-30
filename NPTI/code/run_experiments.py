import os
import json
import argparse
import itertools
import random
import utils
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# personalities = json.load(open('personality_50.json', 'r', encoding='utf-8'))

def load_sys_prompts(behavior):
    behavior_name = ''
    if 'altruism' in behavior:
        behavior_name = 'altruism'
    elif 'heuristic' in behavior:
        behavior_name = 'heuristic'
    elif 'social_media' in behavior:
        behavior_name ='social_media'
    elif'self-control' in behavior:
        behavior_name ='self-control'
    elif 'physical_risk' in behavior:
        behavior_name = 'physical_risk'

    with open(os.path.join('prompts_v5', f'{behavior_name}_behavior.txt'), 'r') as f:
        prompts = f.readlines()
        return prompts[0]

def load_dataset(behavior):
    with open(os.path.join(f'{behavior}.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data
    
def process_dataset_heuristic(dataset):
    inputs = []

    for i in range(len(dataset)):
        # reorder = random.choice([True, False])
        inputs.append({
            "id": i+1,
            "input": dataset[i]['question'] + f" {dataset[i]['choice1']} or {dataset[i]['choice2']}",
            "choice1_label": dataset[i]['choice1_label'],
            "choice2_label": dataset[i]['choice2_label'],
            # "reorder" : False,
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


def run_experiment(personality, prompt, input, method, model, behavior, thinking, temperature):     
    if '{behavior}' in prompt:
        sys_prompt = prompt.replace('{behavior}', input['behavior'])
    else:
        sys_prompt = prompt
    response, reasoning_content = utils.get_response(sys_prompt, input['input'], model=model, nth_generation=0, thinking=thinking, temperature=temperature)

    label = 'N'
    if 'heuristic' in behavior:
        dataset = load_dataset(behavior)
        if dataset[input['id']-1]['choice1'].lower() in response.strip().strip('.').lower() and dataset[input['id']-1]['choice1_label'] == 'Y':
            label = 'Y'
        elif dataset[input['id']-1]['choice2'].lower() in response.strip().strip('.').lower() and dataset[input['id']-1]['choice2_label'] == 'Y':
            label = 'Y'
    elif 'self-control' in behavior:
        dataset = load_dataset(behavior)
        if 'yes' in response.strip().strip('.').lower() and dataset[int((input['id']-1)/10)]['label'] == '':
            label = 'Y'
        elif 'no' in response.strip().strip('.').lower() and dataset[int((input['id']-1)/10)]['label'] == 'R':
            label = 'Y'
        else:
            label = 'N'
    else:
        if "yes" in response.lower():
            label = 'Y'
        elif "no" in response.lower():
            label = 'N'
    result = {
        "id": input['id'],
        "sys_prompt": sys_prompt,
        "input": input['input'],
        "response": response.strip(),
        "label": label,
        "reasoning_content": reasoning_content.strip() if reasoning_content else None,
    }

    return result

def calculate_metrics(methods, model, thinking, temperature):
    result_path = 'results_p4' if not thinking else 'results_thinking'
    result_file_prefix = os.path.join(result_path, model.split('local-')[-1])
    for method in methods:
        # 获取results/model/method/ 下的文件夹
        all_files = os.listdir(os.path.join(result_file_prefix, method, f't={temperature}'))
        all_files = [f for f in all_files if os.path.isdir(os.path.join(result_file_prefix, method, f't={temperature}', f))]
        metrics = {
            'single': {},
            'multi': {},
            'profile': {}
        }
        for personality_folder in all_files:
            # 获取每个文件夹下的文件
            files = os.listdir(os.path.join(result_file_prefix, method, f't={temperature}', personality_folder))
            files = [f for f in files if f.endswith('.json')]
            metric = {}
            for file in files:
                with open(os.path.join(result_file_prefix, method, f't={temperature}', personality_folder, file), 'r', encoding='utf-8') as f:
                    results = json.load(f)
                behavior = file.split('.')[0]
                # 计算指标
                total_count = 0
                yes_count = 0

                if 'heuristic' in behavior or 'self-control' in behavior or 'physical_risk' in behavior or'social_media' in behavior or 'altruism' in behavior:
                    for i in range(len(results)):
                        if 'y' in results[i]['label'].lower():
                            yes_count += 1
                        if 'n' in results[i]['label'].lower() or 'y' in results[i]['label'].lower():
                            total_count += 1
                else:
                    continue

                if total_count > 0:
                    percentage = yes_count / total_count * 100.0
                    metric[behavior] = round(percentage, 2)
                else:
                    metric[behavior] = None
            if '.' in personality_folder or '_' not in personality_folder:
                metrics['profile'][personality_folder] = metric
            elif '0' in personality_folder:
                metrics['single'][personality_folder.split('.')[0]] = metric
            else:
                metrics['multi'][personality_folder.split('.')[0]] = metric
        # 保存指标
        with open(os.path.join(result_file_prefix, method, f't={temperature}', f'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

max_workers = 4
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # O: physical_risk; C: self-control; E: social_media; A: altruism; N: heuristic
    # behavior_choices = ','.join(['altruism', 'heuristic', 'physical_risk','social_media','self-control'])
    # behavior_choices_v2 = ','.join(['altruism_v2', 'heuristic_v2', 'physical_risk_v2','social_media_v2','self-control_v2'])
    behavior_choices_v2 = ','.join(['heuristic_v2'])
    parser.add_argument('--behavior', type=str, default= behavior_choices_v2)
    parser.add_argument('--method', type=str, default='simple,score')
    parser.add_argument('--model', type=str, default='local-qwen3-14b')
    parser.add_argument('--thinking', type=bool, default=False)
    parser.add_argument('--temperature', type=str, default='0.7')
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--character', type=bool, default=False)
    args = parser.parse_args()

    behaviors = args.behavior.split(',')
    model = args.model
    thinking = args.thinking
    methods = args.method.split(',')
    temperatures = [float(t) for t in args.temperature.split(',')]
    profile = args.profile
    character = args.character
    rerun = True
    roles = ["James Bond", "Thor", "Rorschach", "Blair Waldorf", "Gaston"]
    roleinfo = []
    if profile == True:
        for role in roles:
            with open(os.path.join('profiles', f'{role}.jsonl'), 'r', encoding='utf-8') as f:
                roleinfo.append(json.load(f)['text'])

    for temperature in temperatures:
        os.environ['TEMPERATURE'] = str(temperature)
        os.environ['LOCAL_MODEL'] = str(model)
        os.environ['THINKING'] = str(thinking)

        executor = ThreadPoolExecutor(max_workers=max_workers)

        for behavior in behaviors:
            # 1. 读取sys_prompts
            prompt = load_sys_prompts(behavior)

            # 2. 读取数据集
            dataset = load_dataset(behavior)
            if 'altruism' in behavior:
                # inputs = process_dataset_altruism(dataset) 
                inputs = process_dataset_stantard(dataset)
            elif 'heuristic' in behavior:
                inputs = process_dataset_heuristic(dataset)
            elif 'physical_risk' in behavior or 'social_media' in behavior:
                inputs = process_dataset_stantard(dataset)
            elif 'self-control' in behavior:
                # inputs = process_dataset_self_control(dataset)
                inputs = process_dataset_stantard(dataset)
            else:
                pass
            # continue
            with open(os.path.join('big5_descriptions.json'), 'r', encoding='utf-8') as f:
                big5_descriptions = json.load(f)

            # 3. 实验
            for method in methods:
                # method 1: simple
                if method =='simple':
                    personalities=[(0,0,0,0,0)]
                    if character == True or profile == True:
                        personalities = [
                            (3.7, 5, 3.3, 3.3, 1.3),
                            (3.7, 2.7, 5, 4, 4),
                            (1.7, 4.7, 1.7, 1.7, 4),
                            (4, 4.7, 4.7, 2, 3.3),
                            (1.7, 2, 4.7, 1, 4.7)
                        ]
                    for idx, personality in enumerate(personalities):
                        personality = dict(zip(['O', 'C', 'E', 'A', 'N'], personality))
                        result_path = 'results_p4' if not thinking else 'results_thinking'
                        result_file = os.path.join(result_path, model.split('local-')[-1], method, f't={temperature}', f'O_{personality["O"]} C_{personality["C"]} E_{personality["E"]} A_{personality["A"]} N_{personality["N"]}' if profile == False else roles[idx])
                        if not rerun and os.path.exists(os.path.join(result_file, f'{behavior}.json')):
                            continue

                        personality_prompt = ''
                        if profile == True:
                            personality_prompt = f'{roleinfo[idx]}' + personality_prompt
                        personality_prompt = prompt.replace('{Personality}', personality_prompt)

                        print(f'Running {behavior} with null personality...')
                        results = []
                        # for input in tqdm(inputs):
                        #     results.append(run_experiment(personality, personality_prompt, input, method, model, behavior, thinking))
                        futures = [executor.submit(run_experiment, personality, personality_prompt, input, method, model, behavior, thinking, temperature) for input in inputs]
                        for future in tqdm(futures):
                            result = future.result()
                            results.append(result)

                        os.makedirs(result_file, exist_ok=True)
                        with open(os.path.join(result_file, f'{behavior}.json'), 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)
                
                # method 2: score
                elif method =='score':                  
                    if 'physical_risk' in behavior:
                        # personalities = [(1,0,0,0,0), (2,0,0,0,0), (3,0,0,0,0), (4,0,0,0,0), (5,0,0,0,0)]
                        personalities = [(1,0,0,0,0), (5,0,0,0,0)]
                        #第一位不动
                        # personalities = [(1,1,0,0,0), (1,5,0,0,0), (1,0,1,0,0), (1,0,5,0,0), (1,0,0,1,0), (1,0,0,5,0), (1,0,0,0,1), (1,0,0,0,5), (5,1,0,0,0), (5,5,0,0,0), (5,0,1,0,0), (5,0,5,0,0), (5,0,0,1,0), (5,0,0,5,0), (5,0,0,0,1), (5,0,0,0,5)]
                    elif 'self-control' in behavior:
                        # personalities = [(0,1,0,0,0), (0,2,0,0,0), (0,3,0,0,0), (0,4,0,0,0), (0,5,0,0,0)]
                        personalities = [(0,1,0,0,0), (0,5,0,0,0)]
                        #第二位不动
                        # personalities = [(1,1,0,0,0), (5,1,0,0,0), (0,1,1,0,0), (0,1,5,0,0), (0,1,0,1,0), (0,1,0,5,0), (0,1,0,0,1), (0,1,0,0,5), (1,5,0,0,0), (5,5,0,0,0), (0,5,1,0,0), (0,5,5,0,0), (0,5,0,1,0), (0,5,0,5,0), (0,5,0,0,1), (0,5,0,0,5)]
                    elif 'social_media' in behavior:
                        # personalities = [(0,0,1,0,0), (0,0,2,0,0), (0,0,3,0,0), (0,0,4,0,0), (0,0,5,0,0)]
                        personalities = [(0,0,1,0,0), (0,0,5,0,0)]
                        #第三位不动
                        # personalities = [(1,0,1,0,0), (5,0,1,0,0), (0,1,1,0,0), (0,5,1,0,0), (0,0,1,1,0), (0,0,1,5,0), (0,0,1,0,1), (0,0,1,0,5), (1,0,5,0,0), (5,0,5,0,0), (0,1,5,0,0), (0,5,5,0,0), (0,0,5,1,0), (0,0,5,5,0), (0,0,5,0,1), (0,0,5,0,5)]
                    elif 'altruism' in behavior:
                        # personalities = [(0,0,0,1,0), (0,0,0,2,0), (0,0,0,3,0), (0,0,0,4,0), (0,0,0,5,0)]
                        personalities = [(0,0,0,1,0), (0,0,0,5,0)]
                        #第四位不动
                        # personalities = [(1,0,0,1,0), (5,0,0,1,0), (0,1,0,1,0), (0,5,0,1,0), (0,0,1,1,0), (0,0,5,1,0), (0,0,0,1,1), (0,0,0,1,5), (1,0,0,5,0), (5,0,0,5,0), (0,1,0,5,0), (0,5,0,5,0), (0,0,1,5,0), (0,0,5,5,0), (0,0,0,5,1), (0,0,0,5,5)]
                    elif 'heuristic' in behavior:
                        # personalities = [(0,0,0,0,1), (0,0,0,0,2), (0,0,0,0,3), (0,0,0,0,4), (0,0,0,0,5)]
                        personalities = [(0,0,0,0,1), (0,0,0,0,5)]
                        #第五位不动
                        # personalities = [(1,0,0,0,1), (5,0,0,0,1), (0,1,0,0,1), (0,5,0,0,1), (0,0,1,0,1), (0,0,5,0,1), (0,0,0,1,1), (0,0,0,5,1), (1,0,0,0,5), (5,0,0,0,5), (0,1,0,0,5), (0,5,0,0,5), (0,0,1,0,5), (0,0,5,0,5), (0,0,0,1,5), (0,0,0,5,5)]
                    else:
                        personalities = [
                            (3, 3, 3, 3, 3)
                        ]
                    # personalities.append((3, 3, 3, 3, 3))
                    # personalities.extend(list(itertools.product((1, 5), repeat=5)))
                    if character == True or profile == True:
                        personalities = [
                            (3.7, 5, 3.3, 3.3, 1.3),
                            (3.7, 2.7, 5, 4, 4),
                            (1.7, 4.7, 1.7, 1.7, 4),
                            (4, 4.7, 4.7, 2, 3.3),
                            (1.7, 2, 4.7, 1, 4.7)
                        ]
                    for idx, personality in enumerate(personalities):
                        personality = dict(zip(['O', 'C', 'E', 'A', 'N'], personality))
                        result_path = 'results_p4' if not thinking else 'results_thinking'
                        result_file = os.path.join(result_path, model.split('local-')[-1], method, f't={temperature}', f'O_{personality["O"]} C_{personality["C"]} E_{personality["E"]} A_{personality["A"]} N_{personality["N"]}' if profile == False else roles[idx])

                        if not rerun and os.path.exists(os.path.join(result_file, f'{behavior}.json')):
                            continue
                        
                        personality_prompt = f'Your Big Five Personality score is '
                        if profile == True:
                            personality_prompt = f'{roleinfo[idx]}' + personality_prompt
                        # {"Openness: " + str(personality["O"]) + ", Conscientiousness: " + str(personality["C"]) + ", Extroversion: " + str(personality["E"]) + ", Agreeableness: " + str(personality["A"]) + ", Neuroticism: " + str(personality["N"])}. (Note that personality trait scores range from 1 to 5, where 1 means that the personality trait exhibits a low attribute and 5 means that the personality trait exhibits a high attribute.)'
                        personality_prompt += f'{"Openness: " + str(personality["O"])}, ' if personality["O"] > 0 else ''
                        personality_prompt += f'{"Conscientiousness: " + str(personality["C"])}, ' if personality["C"] > 0 else ''
                        personality_prompt += f'{"Extroversion: " + str(personality["E"])}, ' if personality["E"] > 0 else ''
                        personality_prompt += f'{"Agreeableness: " + str(personality["A"])}, ' if personality["A"] > 0 else ''
                        personality_prompt += f'{"Neuroticism: " + str(personality["N"])}, ' if personality["N"] > 0 else ''
                        personality_prompt = personality_prompt.strip().strip(',') + '.'
                        personality_prompt += ' (Note that personality trait scores range from 1 to 5, where 1 means that the personality trait exhibits a low attribute and 5 means that the personality trait exhibits a high attribute.)'

                        personality_prompt = prompt.replace('{Personality}', personality_prompt)

                        print(f'Running {behavior} with personality {personality}...')
                        results = []
                        futures = [executor.submit(run_experiment, personality, personality_prompt, input, method, model, behavior, thinking, temperature) for input in inputs]
                        for future in tqdm(futures):
                            result = future.result()
                            results.append(result)
                        # for input in tqdm(inputs):
                            # results.append(run_experiment(personality, personality_prompt, input, method, model, behavior, thinking))

                        os.makedirs(result_file, exist_ok=True)
                        with open(os.path.join(result_file, f'{behavior}.json'), 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)

                # method 3: description
                elif method == 'description':
                    if 'physical_risk' in behavior:
                        personalities = [(1,0,0,0,0), (5,0,0,0,0)]
                        #第一位不动
                        # personalities = [(1,1,0,0,0), (1,5,0,0,0), (1,0,1,0,0), (1,0,5,0,0), (1,0,0,1,0), (1,0,0,5,0), (1,0,0,0,1), (1,0,0,0,5), (5,1,0,0,0), (5,5,0,0,0), (5,0,1,0,0), (5,0,5,0,0), (5,0,0,1,0), (5,0,0,5,0), (5,0,0,0,1), (5,0,0,0,5)]
                    elif 'self-control' in behavior:
                        personalities = [(0,1,0,0,0), (0,5,0,0,0)]
                        #第二位不动
                        # personalities = [(1,1,0,0,0), (5,1,0,0,0), (0,1,1,0,0), (0,1,5,0,0), (0,1,0,1,0), (0,1,0,5,0), (0,1,0,0,1), (0,1,0,0,5), (1,5,0,0,0), (5,5,0,0,0), (0,5,1,0,0), (0,5,5,0,0), (0,5,0,1,0), (0,5,0,5,0), (0,5,0,0,1), (0,5,0,0,5)]
                    elif 'social_media' in behavior:
                        personalities = [(0,0,1,0,0), (0,0,5,0,0)]
                        #第三位不动
                        # personalities = [(1,0,1,0,0), (5,0,1,0,0), (0,1,1,0,0), (0,5,1,0,0), (0,0,1,1,0), (0,0,1,5,0), (0,0,1,0,1), (0,0,1,0,5), (1,0,5,0,0), (5,0,5,0,0), (0,1,5,0,0), (0,5,5,0,0), (0,0,5,1,0), (0,0,5,5,0), (0,0,5,0,1), (0,0,5,0,5)]
                    elif 'altruism' in behavior:
                        personalities = [(0,0,0,1,0), (0,0,0,5,0)]
                        #第四位不动
                        # personalities = [(1,0,0,1,0), (5,0,0,1,0), (0,1,0,1,0), (0,5,0,1,0), (0,0,1,1,0), (0,0,5,1,0), (0,0,0,1,1), (0,0,0,1,5), (1,0,0,5,0), (5,0,0,5,0), (0,1,0,5,0), (0,5,0,5,0), (0,0,1,5,0), (0,0,5,5,0), (0,0,0,5,1), (0,0,0,5,5)]
                    elif 'heuristic' in behavior:
                        personalities = [(0,0,0,0,1), (0,0,0,0,5)]
                        #第五位不动
                        # personalities = [(1,0,0,0,1), (5,0,0,0,1), (0,1,0,0,1), (0,5,0,0,1), (0,0,1,0,1), (0,0,5,0,1), (0,0,0,1,1), (0,0,0,5,1), (1,0,0,0,5), (5,0,0,0,5), (0,1,0,0,5), (0,5,0,0,5), (0,0,1,0,5), (0,0,5,0,5), (0,0,0,1,5), (0,0,0,5,5)]
                    else:
                        personalities = [
                            (3, 3, 3, 3, 3)
                        ]
                    # personalities.append((3, 3, 3, 3, 3))
                    # personalities.extend(list(itertools.product((1, 5), repeat=5)))
                    if character == True or profile == True:
                        personalities = [
                            (3.7, 5, 3.3, 3.3, 1.3),
                            (3.7, 2.7, 5, 4, 4),
                            (1.7, 4.7, 1.7, 1.7, 4),
                            (4, 4.7, 4.7, 2, 3.3),
                            (1.7, 2, 4.7, 1, 4.7)
                        ]
                    for idx, personality in enumerate(personalities):
                        personality = dict(zip(['O', 'C', 'E', 'A', 'N'], personality))
                        result_path = 'results_p4' if not thinking else 'results_thinking'
                        result_file = os.path.join(result_path, model.split('local-')[-1], method, f't={temperature}', f'O_{personality["O"]} C_{personality["C"]} E_{personality["E"]} A_{personality["A"]} N_{personality["N"]}' if profile == False else roles[idx])

                        if not rerun and os.path.exists(os.path.join(result_file, f'{behavior}.json')):
                            continue
                        
                        personality_description = 'The following is a reference for your personality description: '

                        if profile == True:
                            personality_description = f'{roleinfo[idx]}' + personality_description

                        for i,d in enumerate(['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']):
                            if personality[d[0]] > 3:
                                personality_description += big5_descriptions['positive_en'][d]
                            elif personality[d[0]] < 3:
                                personality_description += big5_descriptions['negative_en'][d]
                            else:
                                personality_description += f'Your {d} is neutral.'
                        personality_prompt = prompt.replace('{Personality}', personality_description)

                        print(f'Running {behavior} with personality {personality}...')
                        results = []
                        futures = [executor.submit(run_experiment, personality, personality_prompt, input, method, model, behavior, thinking, temperature) for input in inputs]
                        for future in tqdm(futures):
                            result = future.result()
                            results.append(result)
                        # for input in tqdm(inputs):
                        #     results.append(run_experiment(personality, personality_prompt, input, method, model, behavior, thinking))

                        os.makedirs(result_file, exist_ok=True)
                        with open(os.path.join(result_file, f'{behavior}.json'), 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)

            # 4. 指标计算
            calculate_metrics(methods, model, thinking, temperature)