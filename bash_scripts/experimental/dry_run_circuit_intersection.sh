#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="../../scripts/circuit_intersection_eval.py"
VARIATIONS=("A" "C" "computation")  

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3

    echo "--------------------------------------------------"
    echo "Model: $model"
   
    for variation in "${VARIATIONS[@]}"; do
        echo "----------"
        echo "Analyzing variation ${variation}"
        
        python3.11 "$SCRIPT" \
            --cache_dir "$CACHE_DIR" \
            --verbose 1 \
            --model "$model" \
            --variation "$variation" \
            --train_size "$train_size" \
            --test_size 20 \
            --batch_size "$batch_size" \
            --metric "diff" \
            --grad_function "logit" \
            --answer_function "avg_diff" \
            --data_dir "./data/math"
    done
}

# Models configuration - just one model for dry run
declare -A models
models=(
    ["Qwen/Qwen2.5-1.5B-Instruct"]="50 1"  # Reduced train_size from 5000 to 50
)
 
# Main execution loop
for model in "${!models[@]}"; do
    IFS=' ' read -r train_size batch_size <<< "${models[$model]}"
    run_model "$model" "$train_size" "$batch_size"
done 