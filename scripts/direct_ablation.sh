#!/bin/bash
# Run a comprehensive set of ablation experiments with direct W&B logging

set -e  # Exit on error

# Configuration
DATASET="eeg"
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
EPOCHS=100  # Long training for meaningful results
BASE_RESULTS_DIR="results/eeg_direct_ablation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
QUICK_MODE=false
USE_TENSORBOARD_INTEGRATION=true
RUN_FINAL_VIZ=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --results-dir)
            BASE_RESULTS_DIR="$2"
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
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --no-tb-integration)
            USE_TENSORBOARD_INTEGRATION=false
            shift
            ;;
        --no-final-viz)
            RUN_FINAL_VIZ=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dataset VALUE         Dataset to use (default: eeg)"
            echo "  --epochs VALUE          Number of training epochs (default: 100)"
            echo "  --results-dir VALUE     Directory to store results (default: results/eeg_direct_ablation)"
            echo "  --wandb-project VALUE   W&B project name (default: temporal-gfn-forecasting)"
            echo "  --wandb-entity VALUE    W&B entity name (default: nadhirvincenthassen)"
            echo "  --quick                 Run a smaller set of experiments (useful for testing)"
            echo "  --no-tb-integration     Disable the TensorBoard to W&B integration"
            echo "  --no-final-viz          Skip running the final W&B visualization script"
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

# If quick mode, use fewer epochs
if [ "$QUICK_MODE" = true ]; then
    EPOCHS=10
    echo "Quick mode enabled: Using ${EPOCHS} epochs"
fi

# Print header
echo "===================================================================="
echo "Running comprehensive ablation study with direct W&B logging"
echo "===================================================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Results directory: $BASE_RESULTS_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "W&B entity: $WANDB_ENTITY"
echo "TensorBoard integration: $([ "$USE_TENSORBOARD_INTEGRATION" = true ] && echo "Enabled" || echo "Disabled")"
echo "Final visualization script: $([ "$RUN_FINAL_VIZ" = true ] && echo "Enabled" || echo "Disabled")"

# Create the base results directory
mkdir -p "$BASE_RESULTS_DIR"

# Make sure the script is executable
chmod +x scripts/direct_ablation.py
chmod +x scripts/wandb_tensorboard_integration.py
chmod +x scripts/visualize_wandb_results.sh

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
    python scripts/direct_ablation.py \
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
        
        # Generate intermediate visualizations after each successful experiment
        update_visualizations
        
        return 0
    else
        echo "✗ Experiment $NAME failed."
        return 1
    fi
}

# Function to update visualizations
update_visualizations() {
    # Create the visualization directory
    PLOTS_DIR="${BASE_RESULTS_DIR}_plots"
    mkdir -p "$PLOTS_DIR"
    
    echo "Updating visualization plots using W&B..."
    
    if [ "$USE_TENSORBOARD_INTEGRATION" = true ]; then
        # Use the comprehensive TensorBoard to W&B integration
        echo "Running TensorBoard to W&B integration for enhanced visualizations..."
        python scripts/wandb_tensorboard_integration.py \
            --results_dir "$BASE_RESULTS_DIR" \
            --output_dir "$PLOTS_DIR" \
            --study_type "all" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_entity "$WANDB_ENTITY"
    else
        # Use the standard W&B visualization approach
        # Study-specific visualizations
        for study_type in "quantization" "policy" "entropy"; do
            echo "Generating visualizations for $study_type study..."
            
            # Create study-specific directory
            study_dir="$PLOTS_DIR/$study_type"
            mkdir -p "$study_dir"
            
            # Run the W&B visualization script for this study type
            python scripts/visualize_wandb_ablations.py \
                --study_type "$study_type" \
                --results_dir "$BASE_RESULTS_DIR" \
                --output_dir "$study_dir" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_entity "$WANDB_ENTITY"
        done
        
        # Combined visualization (all studies)
        echo "Generating combined visualizations..."
        mkdir -p "$PLOTS_DIR/combined"
        
        python scripts/visualize_wandb_ablations.py \
            --study_type "all" \
            --results_dir "$BASE_RESULTS_DIR" \
            --output_dir "$PLOTS_DIR/combined" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_entity "$WANDB_ENTITY"
    fi
    
    # Generate forecast comparisons for best experiments
    echo "Generating forecast comparisons..."
    mkdir -p "$PLOTS_DIR/forecast_comparison"
    
    # Find the best experiment directories for each category based on naming convention
    # This is a simple approach - ideally we'd choose based on metrics
    FIXED_EXP=$(find "$BASE_RESULTS_DIR" -type d -name "*fixed_k10_uniform*" | head -n 1)
    ADAPTIVE_EXP=$(find "$BASE_RESULTS_DIR" -type d -name "*adaptive_k10_uniform*" | head -n 1)
    LEARNED_EXP=$(find "$BASE_RESULTS_DIR" -type d -name "*adaptive_k10_learned*" | head -n 1)
    
    # Only run comparison if we have all three experiments
    if [ -n "$FIXED_EXP" ] && [ -n "$ADAPTIVE_EXP" ] && [ -n "$LEARNED_EXP" ]; then
        python scripts/compare_forecasts.py \
            --results_dirs "$FIXED_EXP" "$ADAPTIVE_EXP" "$LEARNED_EXP" \
            --output_dir "$PLOTS_DIR/forecast_comparison"
    else
        echo "Skipping forecast comparison - not all required experiments available yet"
    fi
    
    echo "W&B visualizations updated! Check the dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
}

# Track successful/failed experiments
SUCCESSFUL=0
FAILED=0

echo "Running key representative experiments first:"

# Run a few representative experiments first 
# (in case you want to just try a few before the full set)
run_experiment "${DATASET}_adaptive_k10_uniform_${TIMESTAMP}" "adaptive" "10" "uniform" "0.01"
if [ $? -eq 0 ]; then SUCCESSFUL=$((SUCCESSFUL + 1)); else FAILED=$((FAILED + 1)); fi

run_experiment "${DATASET}_fixed_k10_uniform_${TIMESTAMP}" "fixed" "10" "uniform" "0.01"
if [ $? -eq 0 ]; then SUCCESSFUL=$((SUCCESSFUL + 1)); else FAILED=$((FAILED + 1)); fi

run_experiment "${DATASET}_adaptive_k10_learned_${TIMESTAMP}" "adaptive" "10" "learned" "0.01"
if [ $? -eq 0 ]; then SUCCESSFUL=$((SUCCESSFUL + 1)); else FAILED=$((FAILED + 1)); fi

# Print summary
echo "===================================================================="
echo "Initial experiments completed!"
echo "===================================================================="
echo "Total experiments: $((SUCCESSFUL + FAILED))"
echo "Successful experiments: $SUCCESSFUL"
echo "Failed experiments: $FAILED"
echo "Results saved to: $BASE_RESULTS_DIR"
echo "All results logged to W&B project: $WANDB_PROJECT"
echo "Check W&B dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"

# If in quick mode, skip asking for full study
if [ "$QUICK_MODE" = true ]; then
    CONTINUE="n"
    echo "Quick mode enabled: Skipping full ablation study"
else
    # Ask if user wants to continue with full ablation study
    echo ""
    echo "Do you want to run the full ablation study? (y/n)"
    read -r CONTINUE
fi

if [[ "$CONTINUE" == "y" || "$CONTINUE" == "Y" ]]; then
    echo "Running full ablation study..."
    
    # Reset counters
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
    echo "Full ablation study completed!"
    echo "===================================================================="
    echo "Total experiments: $((SUCCESSFUL + FAILED))"
    echo "Successful experiments: $SUCCESSFUL"
    echo "Failed experiments: $FAILED"
    echo "Results saved to: $BASE_RESULTS_DIR"
    echo "All results logged to W&B project: $WANDB_PROJECT"
    echo "Check W&B dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
else
    echo "Skipping full ablation study."
fi 

# Generate final visualization plots
echo "===================================================================="
echo "Generating final visualization plots for the ablation study"
echo "===================================================================="

# Call the visualization update function
update_visualizations

# Run the comprehensive visualization script if enabled
if [ "$RUN_FINAL_VIZ" = true ]; then
    echo "===================================================================="
    echo "Running the comprehensive W&B visualization script"
    echo "===================================================================="
    
    # Run the script with our parameters
    ./scripts/visualize_wandb_results.sh \
        --results-dir "$BASE_RESULTS_DIR" \
        --output-dir "${BASE_RESULTS_DIR}_comprehensive_plots" \
        --use-wandb \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "${DATASET}_comprehensive_visualization"
    
    echo "Comprehensive visualization completed!"
fi

echo "Final visualizations complete!"
echo "Plots saved to: ${BASE_RESULTS_DIR}_plots"
echo "Check W&B dashboard at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT" 