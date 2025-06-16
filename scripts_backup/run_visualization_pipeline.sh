#!/bin/bash
# Visualization pipeline for Temporal GFlowNet real experiment results
# This script runs visualization tools for real experiment data only

set -e  # Exit on error

# Configuration - Real experiment directories only
WANDB_RESULTS_DIR="results/wandb_ablations"
REAL_EXPERIMENT_DIRS=(
  "results/wandb_ablations/fixed_k10" 
  "results/wandb_ablations/adaptive_k10" 
  "results/wandb_ablations/learned_policy"
)

# Output directories
OUTPUT_BASE_DIR="results"
ABLATION_OUTPUT_DIR="${OUTPUT_BASE_DIR}/ablation_plots"
FORECAST_OUTPUT_DIR="${OUTPUT_BASE_DIR}/forecast_comparison"
PAPER_FIGS_DIR="${OUTPUT_BASE_DIR}/paper_figures"
WANDB_PLOTS_DIR="${OUTPUT_BASE_DIR}/wandb_plots"

# W&B config
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"

# Print header
echo "===================================================================="
echo "Running real experiment visualization pipeline"
echo "===================================================================="
echo "W&B results directory: $WANDB_RESULTS_DIR"
echo "Real experiment directories: ${REAL_EXPERIMENT_DIRS[@]}"

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        echo "Warning: Directory $1 does not exist"
        return 1
    fi
    return 0
}

# Function to run a script and handle errors
run_script() {
    SCRIPT=$1
    shift
    
    echo "--------------------------------------------------------------------"
    echo "Running: python $SCRIPT $@"
    echo "--------------------------------------------------------------------"
    
    python "$SCRIPT" "$@"
    
    if [ $? -eq 0 ]; then
        echo "✓ Script $SCRIPT completed successfully!"
        return 0
    else
        echo "✗ Script $SCRIPT failed."
        return 1
    fi
}

# Create output directories
mkdir -p "$ABLATION_OUTPUT_DIR"
mkdir -p "$FORECAST_OUTPUT_DIR"
mkdir -p "$PAPER_FIGS_DIR"
mkdir -p "$WANDB_PLOTS_DIR"

# Step 1: Visualize W&B results for ablation studies
echo "===================================================================="
echo "Step 1: Visualizing W&B ablation studies"
echo "===================================================================="

if check_directory "$WANDB_RESULTS_DIR"; then
    # Run for different study types
    for study_type in "quantization" "entropy" "policy"; do
        echo "Processing study type: $study_type"
        run_script scripts/visualize_wandb_ablations.py --study_type "$study_type" \
                  --results_dir "$WANDB_RESULTS_DIR" \
                  --output_dir "$WANDB_PLOTS_DIR/$study_type" \
                  --wandb_project "$WANDB_PROJECT" \
                  --wandb_entity "$WANDB_ENTITY"
    done
else
    echo "Skipping W&B visualization due to missing directory"
fi

# Step 2: Compare forecasts from real experiments
echo "===================================================================="
echo "Step 2: Comparing forecasts from real experiments"
echo "===================================================================="

FORECAST_DIRS_ARGS=""
for dir in "${REAL_EXPERIMENT_DIRS[@]}"; do
    if check_directory "$dir"; then
        FORECAST_DIRS_ARGS="$FORECAST_DIRS_ARGS $dir"
    fi
done

if [ -n "$FORECAST_DIRS_ARGS" ]; then
    run_script scripts/compare_forecasts.py --results_dirs $FORECAST_DIRS_ARGS --output_dir "$FORECAST_OUTPUT_DIR"
else
    echo "Skipping forecast comparison due to missing directories"
fi

# Step 3: Generate theoretical visualizations for paper
echo "===================================================================="
echo "Step 3: Generating theoretical visualizations for paper"
echo "===================================================================="

# Run directly with the functions that generate the visualizations
run_script -c "from scripts.plot_ablation_results import plot_tb_loss_visualization, plot_ste_visualization, plot_quantization_range_visualization; \
               plot_tb_loss_visualization('$PAPER_FIGS_DIR'); \
               plot_ste_visualization('$PAPER_FIGS_DIR'); \
               plot_quantization_range_visualization('$PAPER_FIGS_DIR')"

# Print summary
echo "===================================================================="
echo "Visualization pipeline completed!"
echo "===================================================================="
echo "W&B ablation plots: $WANDB_PLOTS_DIR"
echo "Forecast comparison: $FORECAST_OUTPUT_DIR"
echo "Paper figures: $PAPER_FIGS_DIR"
echo "====================================================================" 