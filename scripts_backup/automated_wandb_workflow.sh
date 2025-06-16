#!/bin/bash
# Automated Workflow for W&B Integration with Temporal GFlowNet
# This script runs experiments, logs to W&B, and creates visualizations

set -e  # Exit on error

# W&B Configuration
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"

# Default parameters
DATASET="eeg"
QUANTIZATION="adaptive"
K_VALUES=(5 10 20)
POLICY="uniform"
ENTROPY=0.01
EPOCHS=50
BASE_NAME="wandb_test"
RESULTS_DIR="results/wandb_test"

# Print header
echo "===================================================================="
echo "Automated W&B Workflow for Temporal GFlowNet"
echo "===================================================================="
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"
echo "Dataset: $DATASET"
echo "Base Results Directory: $RESULTS_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run a command with error handling
run_command() {
    echo "--------------------------------------------------------------------"
    echo "Running: $@"
    echo "--------------------------------------------------------------------"
    
    "$@"
    
    if [ $? -eq 0 ]; then
        echo "✓ Command completed successfully!"
        return 0
    else
        echo "✗ Command failed with exit code $?."
        return 1
    fi
}

# Phase 1: Run experiments with direct ablations
echo "===================================================================="
echo "Phase 1: Running Direct Ablation Experiments"
echo "===================================================================="

# Run fixed quantization experiments
for k in "${K_VALUES[@]}"; do
    exp_name="${BASE_NAME}_fixed_k${k}"
    exp_dir="${RESULTS_DIR}/fixed_k${k}"
    
    echo "Running experiment: $exp_name"
    run_command python scripts/direct_ablation.py \
        --name "$exp_name" \
        --dataset "$DATASET" \
        --quantization "fixed" \
        --k "$k" \
        --policy "$POLICY" \
        --entropy "$ENTROPY" \
        --epochs "$EPOCHS" \
        --output_dir "$exp_dir" \
        --wandb
done

# Run adaptive quantization experiments
for k in "${K_VALUES[@]}"; do
    exp_name="${BASE_NAME}_adaptive_k${k}"
    exp_dir="${RESULTS_DIR}/adaptive_k${k}"
    
    echo "Running experiment: $exp_name"
    run_command python scripts/direct_ablation.py \
        --name "$exp_name" \
        --dataset "$DATASET" \
        --quantization "adaptive" \
        --k "$k" \
        --policy "$POLICY" \
        --entropy "$ENTROPY" \
        --epochs "$EPOCHS" \
        --output_dir "$exp_dir" \
        --wandb
done

# Run learned policy experiment
exp_name="${BASE_NAME}_learned_policy"
exp_dir="${RESULTS_DIR}/learned_policy"

echo "Running experiment: $exp_name"
run_command python scripts/direct_ablation.py \
    --name "$exp_name" \
    --dataset "$DATASET" \
    --quantization "$QUANTIZATION" \
    --k "${K_VALUES[1]}" \
    --policy "learned" \
    --entropy "$ENTROPY" \
    --epochs "$EPOCHS" \
    --output_dir "$exp_dir" \
    --wandb

# Phase 2: Visualize W&B results
echo "===================================================================="
echo "Phase 2: Visualizing W&B Results"
echo "===================================================================="

# Create visualization output directory
VIZ_OUTPUT_DIR="${RESULTS_DIR}/visualizations"
mkdir -p "$VIZ_OUTPUT_DIR"

# Run visualization for all studies
for study_type in "quantization" "policy"; do
    echo "Visualizing $study_type study results"
    study_output_dir="${VIZ_OUTPUT_DIR}/${study_type}"
    mkdir -p "$study_output_dir"
    
    run_command python scripts/visualize_wandb_ablations.py \
        --study_type "$study_type" \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$study_output_dir" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY"
done

# Generate combined visualization
run_command python scripts/visualize_wandb_ablations.py \
    --study_type "all" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "${VIZ_OUTPUT_DIR}/combined" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

# Phase 3: Generate forecast comparison
echo "===================================================================="
echo "Phase 3: Generating Forecast Comparisons"
echo "===================================================================="

FORECAST_DIR="${VIZ_OUTPUT_DIR}/forecast_comparison"
mkdir -p "$FORECAST_DIR"

# Get the best experiment from each category for comparison
FIXED_EXP="${RESULTS_DIR}/fixed_k${K_VALUES[1]}"
ADAPTIVE_EXP="${RESULTS_DIR}/adaptive_k${K_VALUES[1]}"
LEARNED_EXP="${RESULTS_DIR}/learned_policy"

run_command python scripts/compare_forecasts.py \
    --results_dirs "$FIXED_EXP" "$ADAPTIVE_EXP" "$LEARNED_EXP" \
    --output_dir "$FORECAST_DIR"

# Print summary
echo "===================================================================="
echo "Automated W&B Workflow Completed!"
echo "===================================================================="
echo "Experiments have been run and logged to W&B."
echo "Visualizations have been generated in: $VIZ_OUTPUT_DIR"
echo "Forecast comparisons are available in: $FORECAST_DIR"
echo "W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=====================================================================" 