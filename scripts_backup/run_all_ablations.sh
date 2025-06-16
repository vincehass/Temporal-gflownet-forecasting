#!/bin/bash
# Run a comprehensive set of ablation experiments that will appear on W&B
# This script uses the run_single_ablation.py approach which we know works correctly with W&B

set -e  # Exit on error

# Configuration
DATASET="eeg"
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
EPOCHS=100  # Long training for meaningful results
BASE_RESULTS_DIR="results/eeg_ablation_study"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Print header
echo "===================================================================="
echo "Running comprehensive ablation study with W&B logging"
echo "===================================================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Results directory: $BASE_RESULTS_DIR"
echo "W&B project: $WANDB_PROJECT"

# Create the base results directory
mkdir -p "$BASE_RESULTS_DIR"

# Make sure the script is executable
chmod +x scripts/run_single_ablation.py

# Function to run a single experiment
run_experiment() {
    NAME=$1
    QUANT=$2
    K=$3
    POLICY=$4
    ENTROPY=$5
    
    echo "--------------------------------------------------------------------"
    echo "Running experiment: $NAME"
    echo "--------------------------------------------------------------------"
    
    # Set up environment variable to avoid OpenMP issues
    export KMP_DUPLICATE_LIB_OK=TRUE
    
    # Run the experiment
    python scripts/run_single_ablation.py \
        --name "$NAME" \
        --dataset "$DATASET" \
        --quantization "$QUANT" \
        --k "$K" \
        --policy "$POLICY" \
        --entropy "$ENTROPY" \
        --epochs "$EPOCHS" \
        --results-dir "$BASE_RESULTS_DIR/$NAME" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY"
        
    # Check if the experiment was successful
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $NAME completed successfully!"
    else
        echo "✗ Experiment $NAME failed."
    fi
}

# Track successful/failed experiments
SUCCESSFUL=0
FAILED=0

# Run experiments for all combinations
for QUANT in "adaptive" "fixed"; do
    for K in 5 10 20; do
        for ENTROPY in 0.0 0.001 0.01 0.1; do
            for POLICY in "uniform" "learned"; do
                # Skip invalid combinations (learned policy with fixed quantization)
                if [ "$POLICY" = "learned" ] && [ "$QUANT" = "fixed" ]; then
                    echo "Skipping invalid combination: $QUANT quantization with $POLICY policy"
                    continue
                fi
                
                # Create a unique name for the experiment
                NAME="${DATASET}_${QUANT}_k${K}_${POLICY}_entropy${ENTROPY//./_}_${TIMESTAMP}"
                
                # Run the experiment
                run_experiment "$NAME" "$QUANT" "$K" "$POLICY" "$ENTROPY"
                
                # Track success/failure
                if [ $? -eq 0 ]; then
                    SUCCESSFUL=$((SUCCESSFUL + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
                
                echo ""
            done
        done
    done
done

# Print summary
echo "===================================================================="
echo "Ablation study completed!"
echo "===================================================================="
echo "Total experiments: $((SUCCESSFUL + FAILED))"
echo "Successful experiments: $SUCCESSFUL"
echo "Failed experiments: $FAILED"
echo "Results saved to: $BASE_RESULTS_DIR"
echo "All results logged to W&B project: $WANDB_PROJECT"
echo "Check W&B dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT" 