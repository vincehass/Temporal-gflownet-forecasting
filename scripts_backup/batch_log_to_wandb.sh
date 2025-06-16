#!/bin/bash
# Batch logging of experiment directories to W&B
# This script logs multiple experiment results to W&B

set -e  # Exit on error

# Configuration
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
RESULTS_DIR="results/wandb_ablations"  # Default directory containing experiments

# Parse command line arguments
if [ $# -ge 1 ]; then
    RESULTS_DIR="$1"
fi

# Print header
echo "===================================================================="
echo "Batch logging experiment results to W&B"
echo "===================================================================="
echo "Results Directory: $RESULTS_DIR"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"

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

# Check if the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory $RESULTS_DIR does not exist"
    exit 1
fi

# Find all experiment directories
EXPERIMENT_DIRS=$(find "$RESULTS_DIR" -maxdepth 1 -type d | grep -v "^$RESULTS_DIR$")

if [ -z "$EXPERIMENT_DIRS" ]; then
    echo "No experiment directories found in $RESULTS_DIR"
    exit 1
fi

echo "Found $(echo "$EXPERIMENT_DIRS" | wc -l) experiment directories:"
echo "$EXPERIMENT_DIRS" | sed 's/^/  - /'

# Prompt for confirmation
read -p "Do you want to log all these experiments to W&B? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Process each experiment directory
for experiment_dir in $EXPERIMENT_DIRS; do
    echo "===================================================================="
    echo "Processing: $experiment_dir"
    echo "===================================================================="
    
    # Extract experiment name for tagging
    exp_name=$(basename "$experiment_dir")
    
    # Determine appropriate tags based on experiment name
    TAGS=""
    
    # Add quantization tag
    if [[ "$exp_name" == *"fixed"* ]]; then
        TAGS="$TAGS fixed_quantization"
    elif [[ "$exp_name" == *"adaptive"* ]]; then
        TAGS="$TAGS adaptive_quantization"
    fi
    
    # Add policy tag
    if [[ "$exp_name" == *"uniform"* ]]; then
        TAGS="$TAGS uniform_policy"
    elif [[ "$exp_name" == *"learned"* ]]; then
        TAGS="$TAGS learned_policy"
    fi
    
    # Add k value tag
    k_value=$(echo "$exp_name" | grep -o 'k[0-9]\+' | sed 's/k//')
    if [ -n "$k_value" ]; then
        TAGS="$TAGS k$k_value"
    fi
    
    # Run the logging script for this experiment
    run_command python scripts/log_experiment_to_wandb.py \
        --experiment_dir "$experiment_dir" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --tags $TAGS
    
    # Add a small delay to avoid rate limits
    sleep 2
done

# Print summary
echo "===================================================================="
echo "Batch logging completed!"
echo "===================================================================="
echo "W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=====================================================================" 