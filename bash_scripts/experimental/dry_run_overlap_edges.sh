#!/bin/bash

# Set directories
CIRCUIT_DIR="./results/discovered-circuits/tokenwise/template_intersection"
OUTPUT_DIR="./results/edge_overlap"

# Function to compute overlap
compute_overlap() {
    local MODEL=$1
    shift
    local CIRCUIT_PATHS=("$@")
    local CIRCUIT_LABELS=("computation" "C" "A") # Adjust if labels differ

    echo "----------"
    echo "Compute overlap for ${MODEL}"
    echo "Circuit labels: ${CIRCUIT_LABELS[*]}"
    echo "----------"

    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_DIR}/${MODEL}"

    # Run the Python script with the computed arguments
    python3.11 ../../scripts/circuit_overlap.py \
        --circuit_paths "${CIRCUIT_PATHS[@]}" \
        --circuit_labels "${CIRCUIT_LABELS[@]}" \
        --output_dir "${OUTPUT_DIR}/${MODEL}"

    python3.11 ../../scripts/circuit_overlap.py \
        --circuit_paths "${CIRCUIT_PATHS[@]}" \
        --circuit_labels "${CIRCUIT_LABELS[@]}" \
        --output_dir "${OUTPUT_DIR}/${MODEL}" \
        --token_pos

    echo "----------"
}

# Model configuration - just one model for dry run
MODEL="qwen2.5-1.5b-instruct"
COMPUTATION_OVERLAP=1.0
C_OVERLAP=0.625
A_OVERLAP=1.0

# Create circuit directory if it doesn't exist
mkdir -p "${CIRCUIT_DIR}/${MODEL}"

# Define circuit paths
COMPUTATION_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_computation_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_50_overlap_${COMPUTATION_OVERLAP}.json"
C_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_C_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_50_overlap_${C_OVERLAP}.json"
A_CIRCUIT_PATH="${CIRCUIT_DIR}/${MODEL}/circuit_A_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_50_overlap_${A_OVERLAP}.json"

# Compute overlap for the model
compute_overlap "$MODEL" "$COMPUTATION_CIRCUIT_PATH" "$C_CIRCUIT_PATH" "$A_CIRCUIT_PATH" 