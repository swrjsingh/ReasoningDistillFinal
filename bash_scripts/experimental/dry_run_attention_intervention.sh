#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="../../scripts/intervene_attention.py"

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local alpha=$4

    echo "--------------------------------------------------"
    echo "Model: $model"
        
    python3.11 "$SCRIPT" \
        --cache_dir "$CACHE_DIR" \
        --verbose 1 \
        --model "$model" \
        --train_size "$train_size" \
        --test_size 20 \
        --batch_size "$batch_size" \
        --metric "diff" \
        --grad_function "logit" \
        --answer_function "avg_diff" \
        --single_variation_data "C" \
        --alpha "$alpha" \
        --save_plot_dir "./results/attention-interventions" \
        --data_dir "./data/math"
}

# Models configuration - just one model for dry run
declare -A models
models=(
    ["Qwen/Qwen2.5-1.5B-Instruct"]="50 1 3.1"  # Reduced train_size from 5000 to 50
)

# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size alpha <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size" "$alpha"
done 