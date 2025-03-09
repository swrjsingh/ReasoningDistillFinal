#!/bin/bash

# Common parameters
CACHE_DIR=""
SCRIPT="../../scripts/identify_circuit.py"
VARIATIONS=("C" "A" "computation")

# Function to run the Python script with given parameters
run_model() {
    local model=$1
    local train_size=$2
    local batch_size=$3
    local interval=$4
    local template=$5

    echo "--------------------------------------------------"
    echo "Model: $model, Train size: $train_size, Batch size: $batch_size, Interval: $interval, Template: $template"

    for variation in "${VARIATIONS[@]}"; do
        echo "----------"
        echo "Analyzing variation ${variation}"
        
        python3.11 "$SCRIPT" \
            --cache_dir "$CACHE_DIR" \
            --model "$model" \
            --template "$template" \
            --variation "$variation" \
            --train_size "$train_size" \
            --test_size 20 \
            --batch_size "$batch_size" \
            --interval "$interval" \
            --metric "diff" \
            --grad_function "logit" \
            --answer_function "avg_diff" \
            --initial_edges 100 \
            --step_size 20
    done
}

# Models configuration - using only Qwen for dry run
declare -A models
models=(
    ["Qwen/Qwen2.5-1.5B-Instruct"]="50 1 99.0|101.0"  # Reduced train_size from 5000 to 50
)

# Main execution loop - using all templates
for template in "0" "1" "4" "5" "6" "7"; do
    for model in "${!models[@]}"; do
        IFS=' ' read -r train_size batch_size interval <<< "${models[$model]}"
        run_model "$model" "$train_size" "$batch_size" "$interval" "$template"
    done
done