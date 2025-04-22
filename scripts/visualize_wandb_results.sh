#!/bin/bash
# W&B Results Visualization Script for Temporal GFlowNet forecasting
# Focuses only on visualizing W&B experiment data

set -e  # Exit on error

# Default W&B Configuration
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
WANDB_NAME="wandb_visualization"
USE_WANDB=false

# Default directories
RESULTS_DIR="results/wandb_ablations"
OUTPUT_DIR="results/wandb_plots"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --results-dir VALUE     Directory containing experiment results (default: results/wandb_ablations)"
            echo "  --output-dir VALUE      Directory to store visualization outputs (default: results/wandb_plots)"
            echo "  --wandb-project VALUE   W&B project name (default: temporal-gfn-forecasting)"
            echo "  --wandb-entity VALUE    W&B entity name (default: nadhirvincenthassen)"
            echo "  --wandb-name VALUE      W&B run name for visualizations (default: wandb_visualization)"
            echo "  --use-wandb             Enable logging visualization runs to W&B"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Print header
echo "===================================================================="
echo "Visualizing W&B Results for Temporal GFlowNet Forecasting"
echo "===================================================================="
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"
echo "Results Directory: $RESULTS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Logging to W&B: $([ "$USE_WANDB" = true ] && echo "Enabled" || echo "Disabled")"

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

# Find key experiment directories for forecast comparison
echo "Looking for experiment directories for forecast comparison..."

# Try to find representative experiment directories
FIXED_EXP=$(find "$RESULTS_DIR" -type d -name "*fixed_k10*uniform*" | head -n 1)
ADAPTIVE_EXP=$(find "$RESULTS_DIR" -type d -name "*adaptive_k10*uniform*" | head -n 1)
LEARNED_EXP=$(find "$RESULTS_DIR" -type d -name "*adaptive_k10*learned*" | head -n 1)

# Only proceed if we found all three experiment types
if [ -n "$FIXED_EXP" ] && [ -n "$ADAPTIVE_EXP" ] && [ -n "$LEARNED_EXP" ]; then
    # Generate forecast comparison plots
    echo "===================================================================="
    echo "Generating forecast comparison plots"
    echo "===================================================================="
    
    forecast_output_dir="$OUTPUT_DIR/forecast_comparison"
    mkdir -p "$forecast_output_dir"
    
    run_script python scripts/compare_forecasts.py \
        --results_dirs "$FIXED_EXP" "$ADAPTIVE_EXP" "$LEARNED_EXP" \
        --output_dir "$forecast_output_dir"
else
    echo "===================================================================="
    echo "Skipping forecast comparison - not all required experiment types found"
    echo "===================================================================="
    echo "Missing experiment directories:"
    [ -z "$FIXED_EXP" ] && echo "- Fixed quantization (k=10) experiment"
    [ -z "$ADAPTIVE_EXP" ] && echo "- Adaptive quantization (k=10) experiment"
    [ -z "$LEARNED_EXP" ] && echo "- Learned policy experiment"
fi

# Summary
echo "===================================================================="
echo "W&B Visualization Completed!"
echo "===================================================================="
echo "Output directories:"
echo "- Study plots: $OUTPUT_DIR"
echo "- Forecast comparison: $OUTPUT_DIR/forecast_comparison (if generated)"
echo "W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "====================================================================" 