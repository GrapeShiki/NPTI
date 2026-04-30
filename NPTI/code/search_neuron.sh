#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Define parameters
MODEL="qwen3.5-9b"
MODEL_PATH="/home/putaowen/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B"
QUESTION_DIR="NPTI/dataset/search"
ANSWER_DIR="NPTI/${MODEL}/answer_results"
NEURON_DIR="NPTI/${MODEL}/neuron_results"
PERSONALITY_DESCRIPTION="NPTI/dataset/description.json"
BATCH_SIZE=16

# Run the Python script with arguments
nohup python NPTI/code/search_neuron.py \
    --model "$MODEL_PATH" \
    --question_dir "$QUESTION_DIR" \
    --answer_dir "$ANSWER_DIR" \
    --neuron_dir "$NEURON_DIR" \
    --personality_desc "$PERSONALITY_DESCRIPTION" \
    --batch_size $BATCH_SIZE \
    > search_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python NPTI/code/process_neuron.py\
    --neuron_dir "$NEURON_DIR" \
    > process_$(date +%Y%m%d_%H%M%S).log 2>&1 &