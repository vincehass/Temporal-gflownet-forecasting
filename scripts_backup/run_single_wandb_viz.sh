#!/bin/bash
# Run W&B visualization for a single experiment
# This script visualizes results from a single experiment directory

set -e  # Exit on error

# Parse command line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_dir> [output_dir]"
    echo "Example: $0 results/eeg_test_wandb results/eeg_test_plots"
    exit 1
fi

# Configuration
EXPERIMENT_DIR="$1"
OUTPUT_DIR="${2:-"$EXPERIMENT_DIR/plots"}"  # Default to experiments_dir/plots if not specified
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"

# Print header
echo "===================================================================="
echo "Running W&B visualization for single experiment"
echo "===================================================================="
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
        echo "✗ Command failed."
        return 1
    fi
}

# Run visualization script on the specific experiment directory
run_command python scripts/visualize_wandb_ablations.py \
    --results_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

# Print summary
echo "===================================================================="
echo "W&B Visualization Completed!"
echo "===================================================================="
echo "Check the output directory: $OUTPUT_DIR"
echo "W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "====================================================================" 