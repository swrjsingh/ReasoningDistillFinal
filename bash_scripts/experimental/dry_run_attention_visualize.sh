#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="../../scripts/attention_interpretation.py"

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local overlap_C=$4
    local overlap_A=$5

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
        --intersection_overlap_C "$overlap_C" \
        --intersection_overlap_A "$overlap_A" \
        --cross_template_averaging \
        --data_dir "./data/math"
}

# Models configuration - just one model for dry run
declare -A models
models=(
    ["Qwen/Qwen2.5-1.5B-Instruct"]="50 1 1.0 1.0"  # Reduced train_size from 5000 to 50, added computation overlap
)

# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size overlap_C overlap_A <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size" "$overlap_C" "$overlap_A"
done 