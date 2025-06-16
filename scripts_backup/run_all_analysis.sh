#!/bin/bash

# Script to run all analysis scripts with full Weights & Biases integration
# This script analyzes results from ablation studies including different quantization methods,
# model variants, and datasets, with all results stored in wandb.

# Default parameters
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
RESULTS_BASE_DIR="results"
OUTPUT_BASE_DIR="results/analysis"
USE_WANDB=true
OFFLINE=""
DATASETS=("synthetic_data" "wandb_ablations")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_BASE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_BASE_DIR="$2"
      shift 2
      ;;
    --datasets)
      # Split comma-separated list into array
      IFS=',' read -ra DATASETS <<< "$2"
      shift 2
      ;;
    --no-wandb)
      USE_WANDB=false
      shift
      ;;
    --offline)
      OFFLINE="--offline"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_all_analysis.sh [options]"
      echo "Options:"
      echo "  --wandb-project PROJECT   W&B project name (default: temporal-gfn-forecasting)"
      echo "  --wandb-entity ENTITY     W&B entity name (default: nadhirvincenthassen)"
      echo "  --results-dir DIR         Base directory containing results (default: results)"
      echo "  --output-dir DIR          Base directory for analysis output (default: results/analysis)"
      echo "  --datasets LIST           Comma-separated list of dataset folders to analyze"
      echo "  --no-wandb                Disable W&B logging"
      echo "  --offline                 Run W&B in offline mode"
      exit 1
      ;;
  esac
done

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"

# Function to run analysis for a single dataset
run_dataset_analysis() {
  local DATASET_NAME=$1
  local RESULTS_DIR="${RESULTS_BASE_DIR}/${DATASET_NAME}"
  local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATASET_NAME}"
  
  echo
  echo "======================================================================"
  echo "Running analysis for dataset: $DATASET_NAME"
  echo "======================================================================"
  
  # Create output directory for this dataset
  mkdir -p "$OUTPUT_DIR"
  
  # Skip if results directory doesn't exist
  if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory $RESULTS_DIR not found, skipping."
    return 1
  fi
  
  # Run enhanced ablation visualization
  echo "Running enhanced ablation visualization..."
  
  CMD=("python" "scripts/enhanced_ablation_viz.py")
  CMD+=("--results_dir" "$RESULTS_DIR")
  CMD+=("--output_dir" "${OUTPUT_DIR}/enhanced_plots")
  CMD+=("--format" "png")
  
  # Add wandb flags if enabled
  if [ "$USE_WANDB" = true ]; then
    CMD+=("--use_wandb")
    CMD+=("--wandb_project" "$WANDB_PROJECT")
    CMD+=("--wandb_entity" "$WANDB_ENTITY")
    CMD+=("--wandb_name" "${DATASET_NAME}_enhanced_visualization")
    
    if [ ! -z "$OFFLINE" ]; then
      CMD+=("$OFFLINE")
    fi
  fi
  
  echo "Running command: ${CMD[@]}"
  "${CMD[@]}"
  
  # Run quantization-specific analysis
  echo "Running quantization-specific analysis..."
  
  CMD=("python" "scripts/quantization_analysis.py")
  CMD+=("--results_dir" "$RESULTS_DIR")
  CMD+=("--output_dir" "${OUTPUT_DIR}/quantization_analysis")
  
  # Add wandb flags if enabled
  if [ "$USE_WANDB" = true ]; then
    CMD+=("--use_wandb")
    CMD+=("--wandb_project" "$WANDB_PROJECT")
    CMD+=("--wandb_entity" "$WANDB_ENTITY")
    CMD+=("--wandb_name" "${DATASET_NAME}_quantization_analysis")
    
    if [ ! -z "$OFFLINE" ]; then
      CMD+=("$OFFLINE")
    fi
  fi
  
  echo "Running command: ${CMD[@]}"
  "${CMD[@]}"
  
  echo "Analysis completed for dataset: $DATASET_NAME"
  return 0
}

# Array to store analysis results
declare -a ANALYSIS_RESULTS=()

# Run analysis for each dataset
for dataset in "${DATASETS[@]}"; do
  run_dataset_analysis "$dataset"
  
  if [ $? -eq 0 ]; then
    ANALYSIS_RESULTS+=("$dataset: Success")
  else
    ANALYSIS_RESULTS+=("$dataset: Failed")
  fi
done

# Print summary
echo
echo "======================================================================"
echo "Analysis Summary:"
echo "======================================================================"
for result in "${ANALYSIS_RESULTS[@]}"; do
  echo "$result"
done

echo
echo "All analyses completed! Results saved to: $OUTPUT_BASE_DIR"
if [ "$USE_WANDB" = true ]; then
  echo "All results logged to W&B project: $WANDB_PROJECT"
  
  if [ ! -z "$OFFLINE" ]; then
    echo "Note: W&B was run in offline mode. Use 'wandb sync' to upload results when online."
  fi
fi 