#!/bin/bash
# W&B Results Visualization Script for Temporal GFlowNet forecasting
# Focuses only on visualizing W&B experiment data

set -e  # Exit on error

# W&B Configuration
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"

# Experiment directories
RESULTS_DIR="results/wandb_ablations"
OUTPUT_DIR="results/wandb_plots"

# Print header
echo "===================================================================="
echo "Visualizing W&B Results for Temporal GFlowNet Forecasting"
echo "===================================================================="
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"
echo "Results Directory: $RESULTS_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a script with error handling
run_script() {
    echo "--------------------------------------------------------------------"
    echo "Running: $@"
    echo "--------------------------------------------------------------------"
    
    "$@"
    
    if [ $? -eq 0 ]; then
        echo "✓ Command completed successfully!"
        return 0
    else
        echo "✗ Command failed."
        return 1
    fi
}

# Visualize W&B results for different study types
for study_type in "quantization" "entropy" "policy"; do
    echo "===================================================================="
    echo "Visualizing $study_type study results"
    echo "===================================================================="
    
    # Create study-specific output directory
    study_output_dir="$OUTPUT_DIR/$study_type"
    mkdir -p "$study_output_dir"
    
    # Run visualization script for this study type
    run_script python scripts/visualize_wandb_ablations.py \
        --study_type "$study_type" \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$study_output_dir" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY"
done

# Run all studies combined visualization
echo "===================================================================="
echo "Visualizing combined study results"
echo "===================================================================="

run_script python scripts/visualize_wandb_ablations.py \
    --study_type "all" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR/combined" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

# Generate forecast comparison plots
echo "===================================================================="
echo "Generating forecast comparison plots"
echo "===================================================================="

forecast_output_dir="$OUTPUT_DIR/forecast_comparison"
mkdir -p "$forecast_output_dir"

run_script python scripts/compare_forecasts.py \
    --results_dirs "$RESULTS_DIR/fixed_k10" "$RESULTS_DIR/adaptive_k10" "$RESULTS_DIR/learned_policy" \
    --output_dir "$forecast_output_dir"

# Summary
echo "===================================================================="
echo "W&B Visualization Completed!"
echo "===================================================================="
echo "Output directories:"
echo "- Study plots: $OUTPUT_DIR"
echo "- Forecast comparison: $forecast_output_dir"
echo "====================================================================" 