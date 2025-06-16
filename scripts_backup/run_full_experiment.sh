#!/bin/bash

# Comprehensive script to run full experiments with all hyperparameter variations
# This includes different quantization methods, entropy values, and datasets

# Set default parameters
DATASETS=("eeg" "synthetic")
QUANTIZATIONS=("adaptive" "fixed")
K_VALUES=(5 10 20)
ENTROPY_VALUES=(0 0.001 0.01 0.1)
POLICY_TYPES=("uniform" "learned")
EPOCHS=50
BATCH_SIZE=64
USE_WANDB=true
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
RESULTS_DIR="results/full_experiment"
RUN_ANALYSIS=true
GPU_ID=0
OFFLINE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --datasets)
      IFS=',' read -ra DATASETS <<< "$2"
      shift 2
      ;;
    --quantizations)
      IFS=',' read -ra QUANTIZATIONS <<< "$2"
      shift 2
      ;;
    --k-values)
      IFS=',' read -ra K_VALUES <<< "$2"
      shift 2
      ;;
    --entropy-values)
      IFS=',' read -ra ENTROPY_VALUES <<< "$2"
      shift 2
      ;;
    --policy-types)
      IFS=',' read -ra POLICY_TYPES <<< "$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --no-wandb)
      USE_WANDB=false
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --no-analysis)
      RUN_ANALYSIS=false
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --offline)
      OFFLINE="--offline"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_full_experiment.sh [options]"
      echo "Options:"
      echo "  --datasets LIST          Comma-separated list of datasets (default: eeg,synthetic)"
      echo "  --quantizations LIST     Comma-separated list of quantization types (default: adaptive,fixed)"
      echo "  --k-values LIST          Comma-separated list of k values (default: 5,10,20)"
      echo "  --entropy-values LIST    Comma-separated list of entropy values (default: 0,0.001,0.01,0.1)"
      echo "  --policy-types LIST      Comma-separated list of policy types (default: uniform,learned)"
      echo "  --epochs N               Number of training epochs (default: 50)"
      echo "  --batch-size N           Batch size (default: 64)"
      echo "  --no-wandb               Disable W&B logging"
      echo "  --wandb-project PROJECT  W&B project name (default: temporal-gfn-forecasting)"
      echo "  --wandb-entity ENTITY    W&B entity name (default: nadhirvincenthassen)"
      echo "  --results-dir DIR        Results directory (default: results/full_experiment)"
      echo "  --no-analysis            Skip running analysis scripts after experiments"
      echo "  --gpu ID                 GPU ID to use (default: 0)"
      echo "  --offline                Run W&B in offline mode"
      exit 1
      ;;
  esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Configure GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Function to run a single experiment
run_experiment() {
  local DATASET=$1
  local QUANT_TYPE=$2
  local K_VALUE=$3
  local ENTROPY=$4
  local POLICY=$5
  
  # Create a unique experiment name
  local EXP_NAME="${DATASET}_${QUANT_TYPE}_k${K_VALUE}_entropy${ENTROPY}_${POLICY}"
  
  # Create experiment directory
  local EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"
  mkdir -p "$EXP_DIR"
  
  echo
  echo "======================================================================"
  echo "Running experiment: $EXP_NAME"
  echo "======================================================================"
  
  # Base command
  CMD=("./scripts/run_experiments.sh")
  CMD+=("--name" "$EXP_NAME")
  CMD+=("--dataset" "$DATASET")
  CMD+=("--quantization" "$QUANT_TYPE")
  CMD+=("--k" "$K_VALUE")
  CMD+=("--entropy" "$ENTROPY")
  CMD+=("--policy" "$POLICY")
  CMD+=("--epochs" "$EPOCHS")
  CMD+=("--batch-size" "$BATCH_SIZE")
  CMD+=("--results-dir" "$RESULTS_DIR")
  CMD+=("--gpu" "$GPU_ID")
  
  # Add W&B flags if enabled
  if [ "$USE_WANDB" = true ]; then
    CMD+=("--use-wandb")
    CMD+=("--wandb-entity" "$WANDB_ENTITY")
    CMD+=("--wandb-project" "$WANDB_PROJECT")
    
    if [ ! -z "$OFFLINE" ]; then
      CMD+=("$OFFLINE")
    fi
  fi
  
  # Run the experiment
  echo "Running command: ${CMD[@]}"
  "${CMD[@]}"
  
  # Return the exit status
  return $?
}

# Array to store experiment results
declare -a EXPERIMENTS=()
declare -a RESULTS=()

# Track start time
START_TIME=$(date +%s)

# Run all experiments
for DATASET in "${DATASETS[@]}"; do
  for QUANT_TYPE in "${QUANTIZATIONS[@]}"; do
    for K_VALUE in "${K_VALUES[@]}"; do
      for ENTROPY in "${ENTROPY_VALUES[@]}"; do
        for POLICY in "${POLICY_TYPES[@]}"; do
          # Skip invalid combinations (e.g., fixed quantization with learned policy)
          if [ "$QUANT_TYPE" = "fixed" ] && [ "$POLICY" = "learned" ]; then
            echo "Skipping invalid combination: fixed quantization with learned policy"
            continue
          fi
          
          # Run the experiment
          EXP_NAME="${DATASET}_${QUANT_TYPE}_k${K_VALUE}_entropy${ENTROPY}_${POLICY}"
          
          EXPERIMENTS+=("$EXP_NAME")
          
          run_experiment "$DATASET" "$QUANT_TYPE" "$K_VALUE" "$ENTROPY" "$POLICY"
          
          if [ $? -eq 0 ]; then
            RESULTS+=("$EXP_NAME: Success")
          else
            RESULTS+=("$EXP_NAME: Failed")
          fi
        done
      done
    done
  done
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(( (TOTAL_TIME % 3600) / 60 ))
SECONDS=$((TOTAL_TIME % 60))

# Run analysis if requested
if [ "$RUN_ANALYSIS" = true ]; then
  echo
  echo "======================================================================"
  echo "Running analysis on all experiments..."
  echo "======================================================================"
  
  # Create dataset-specific subdirectories in the results directory
  for DATASET in "${DATASETS[@]}"; do
    # Run analysis for this dataset
    CMD=("./scripts/run_all_analysis.sh")
    CMD+=("--results-dir" "$RESULTS_DIR")
    CMD+=("--output-dir" "${RESULTS_DIR}/analysis")
    CMD+=("--datasets" "$DATASET")
    
    # Add W&B flags if enabled
    if [ "$USE_WANDB" = true ]; then
      CMD+=("--wandb-project" "$WANDB_PROJECT")
      CMD+=("--wandb-entity" "$WANDB_ENTITY")
      
      if [ ! -z "$OFFLINE" ]; then
        CMD+=("--offline")
      fi
    else
      CMD+=("--no-wandb")
    fi
    
    echo "Running command: ${CMD[@]}"
    "${CMD[@]}"
  done
fi

# Print summary
echo
echo "======================================================================"
echo "Experiment Summary:"
echo "======================================================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo

# Print individual results
echo "Experiment Results:"
for i in "${!EXPERIMENTS[@]}"; do
  echo "${RESULTS[$i]}"
done

echo
echo "All experiments and analysis completed!"
echo "Results saved to: $RESULTS_DIR"
if [ "$USE_WANDB" = true ]; then
  echo "All results logged to W&B project: $WANDB_PROJECT"
  
  if [ ! -z "$OFFLINE" ]; then
    echo "Note: W&B was run in offline mode. Use 'wandb sync' to upload results when online."
  fi
fi 