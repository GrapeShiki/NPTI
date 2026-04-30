#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Define parameters
MODEL="qwen3.5-2b"
MODEL_PATH="/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B"
NEURON_DIR="NPTI/${MODEL}/neuron_results"
OUTPUT_DIR="NPTI/${MODEL}/answer_results_cn"
QUESTION_DIR="NPTI/dataset/test"
BATCH_SIZE=32
PYTHON_FILE="NPTI/code/answer_question_change_neuron.py"  # 指定你的 Python 文件名

# 执行 Python 脚本，并传递参数
nohup python $PYTHON_FILE \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --output_dir $OUTPUT_DIR \
  --question_dir $QUESTION_DIR \
  --batch_size $BATCH_SIZE \
  > answer_$(date +%Y%m%d_%H%M%S).log 2>&1 &
