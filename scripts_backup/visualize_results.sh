#!/bin/bash
# =============================================================================
# Temporal GFN Visualization Script
# =============================================================================
# This script generates visualizations and comparison plots for 
# Temporal GFN experiments.

set -e  # Exit on error

# Check for dependencies
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Default configuration
RESULTS_DIR="results"
OUTPUT_DIR="results/plots"
EXPERIMENTS=()
USE_WANDB=false
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="temporal-gfn-viz"
WANDB_NAME="experiment_visualization"
WANDB_OFFLINE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --results-dir)
            RESULTS_DIR="$2"
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --experiments)
            shift
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
                EXPERIMENTS+=("$1")
                shift
            done
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift
            shift
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift
            shift
            ;;
        --offline)
            WANDB_OFFLINE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --results-dir DIR       Base results directory (default: results)"
            echo "  --output-dir DIR        Output directory for plots (default: results/plots)"
            echo "  --experiments EXP1 EXP2 List of experiment names to compare"
            echo "  --use-wandb             Enable W&B logging for visualizations"
            echo "  --wandb-entity ENTITY   W&B entity name"
            echo "  --wandb-project PROJECT W&B project name"
            echo "  --wandb-name NAME       W&B run name"
            echo "  --offline               Use W&B in offline mode"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Check that experiments were provided
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "ERROR: No experiments specified. Use --experiments EXP1 EXP2 ..."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build experiment directories list
EXP_DIRS=()
VALID_EXPERIMENTS=()
for exp in "${EXPERIMENTS[@]}"; do
    eval_dir="${RESULTS_DIR}/${exp}/evaluation"
    
    # Check if evaluation directory exists
    if [ -d "$eval_dir" ]; then
        # Check if there are metrics files in the directory
        if ls "$eval_dir"/*.json &> /dev/null; then
            EXP_DIRS+=("$eval_dir")
            VALID_EXPERIMENTS+=("$exp")
        else
            echo "WARNING: No metrics files found in ${eval_dir}. Skipping experiment '$exp'."
        fi
    else
        echo "WARNING: Evaluation directory not found at ${eval_dir}. Skipping experiment '$exp'."
    fi
done

# Check if we have any valid experiments
if [ ${#VALID_EXPERIMENTS[@]} -eq 0 ]; then
    echo "ERROR: No valid experiments found. Visualization cannot proceed."
    exit 1
fi

echo "=================== Visualization Summary ==================="
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Valid experiments (${#VALID_EXPERIMENTS[@]}):"
for exp in "${VALID_EXPERIMENTS[@]}"; do
    echo " - $exp"
done
echo "============================================================"

# Prepare W&B arguments if enabled
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT --wandb_run_name $WANDB_NAME"
    
    if [ "$WANDB_OFFLINE" = true ]; then
        WANDB_ARGS="$WANDB_ARGS --offline"
    fi
fi

# Join experiment directories with commas for the Python script
EXP_DIRS_STR=$(IFS=,; echo "${EXP_DIRS[*]}")

# Run the visualization script
echo "Generating visualizations..."
python scripts/plot_results.py \
    --exp_dirs "$EXP_DIRS_STR" \
    --output_dir "$OUTPUT_DIR" \
    $WANDB_ARGS

echo "Visualizations saved to $OUTPUT_DIR" 