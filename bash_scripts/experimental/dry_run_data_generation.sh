#!/bin/bash

# Common parameters
CACHE_DIR=""
TEMPLATES=("0" "1" "2" "3" "4" "5" "6" "7") # Just using template 0 for dry run
#MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-Math-1.5B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "Qwen/Qwen2.5-Math-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.1-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")  # Using both models
MODELS=("Qwen/Qwen2.5-1.5B-Instruct")
# Filter full dataset for each model and template
for template in "${TEMPLATES[@]}"; do 
    echo "Generating data for template ${template}"
    python3.11 ../../scripts/generate_data.py \
        --cache_dir "${CACHE_DIR}" \
        --data_dir "./data/math" \
        --samples 100 \
        --subsamples 5 \
        --template "$template"

    echo "--------------------------------------------------"
    echo "Filtering dataset for template ${template}"
    echo "--------------------------------------------------"

    for model in "${MODELS[@]}"; do
        echo "----------"
        echo "Filtering dataset for model ${model}"
        echo "----------"
        
        python3.11 ../../scripts/filter_data.py \
            --cache_dir "${CACHE_DIR}" \
            --data_dir "./data/math" \
            --template "$template" \
            --model "$model" \
            --batch_size 32 \
            --dtype "bfloat16" 
        
        echo "----------"
    done
done