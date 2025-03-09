#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="../../scripts/evals_on_models.py"
#MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-Math-1.5B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "Qwen/Qwen2.5-Math-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.1-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") 
MODELS=("Qwen/Qwen2.5-1.5B-Instruct")
# Filter full dataset for each model and template
for model in "${MODELS[@]}"; do
    echo "----------"
    echo "Evaluating ${model}"
    echo "----------"
    
    python3.11 "$SCRIPT" \
       --cache_dir "$CACHE_DIR" \
        --model "$model" \
        --batch_size 32 \
        --samples_per_template 10 \
        --dtype "bfloat16"  \
        --data_dir "./data/math"
    echo "----------"
done