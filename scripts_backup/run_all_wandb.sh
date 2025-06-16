#!/bin/bash

# Script to run all experiments with full Weights & Biases integration

# Default parameters
RESULTS_DIR="results/wandb_ablations"
EPOCHS=10
BATCH_SIZE=32
USE_WANDB=true
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
OFFLINE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-dir)
      RESULTS_DIR="$2"
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
    --offline)
      OFFLINE="offline=true"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Array to store experiment names
EXPERIMENT_NAMES=()

# Function to run a single experiment
run_experiment() {
  local EXP_NAME=$1
  local ARGS=("${@:2}")
  
  echo
  echo "======================================================================"
  echo "Starting experiment: $EXP_NAME"
  echo "======================================================================"
  
  # Add to experiment names list
  EXPERIMENT_NAMES+=("$EXP_NAME")
  
  # Create experiment directory
  local EXP_DIR="$RESULTS_DIR/$EXP_NAME"
  mkdir -p "$EXP_DIR"
  mkdir -p "$EXP_DIR/logs"
  mkdir -p "$EXP_DIR/checkpoints"
  mkdir -p "$EXP_DIR/evaluation"
  mkdir -p "$EXP_DIR/plots"
  
  # Build command
  local CMD=("python" "scripts/train.py")
  CMD+=("results_dir=$EXP_DIR")
  
  # Add common settings
  CMD+=("training.epochs=$EPOCHS")
  CMD+=("training.batch_size=$BATCH_SIZE")
  # Add learning_rate with its correct name
  CMD+=("training.learning_rate=0.001")
  
  # Add experiment-specific args
  CMD+=("${ARGS[@]}")
  
  # Add wandb settings if enabled
  if [ "$USE_WANDB" = true ]; then
    CMD+=("use_wandb=true")
    CMD+=("wandb_project=$WANDB_PROJECT")
    CMD+=("wandb_entity=$WANDB_ENTITY")
    CMD+=("+wandb_name=$EXP_NAME")
    
    if [ ! -z "$OFFLINE" ]; then
      CMD+=("$OFFLINE")
    fi
  fi
  
  # Run the training
  echo "Running command: ${CMD[@]}"
  "${CMD[@]}"
  
  local TRAIN_EXIT_CODE=$?
  if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    return 1
  fi
  
  # Run evaluation
  local BEST_CHECKPOINT="$EXP_DIR/checkpoints/best_model.pt"
  local CONFIG_PATH="$EXP_DIR/config.yaml"
  
  if [ -f "$BEST_CHECKPOINT" ] && [ -f "$CONFIG_PATH" ]; then
    echo "Running evaluation for experiment: $EXP_NAME"
    python scripts/evaluate.py \
      --checkpoint_path "$BEST_CHECKPOINT" \
      --config_path "$CONFIG_PATH" \
      --output_dir "$EXP_DIR/evaluation"
    
    local EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
      echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
      return 1
    fi
  else
    echo "Skipping evaluation: checkpoint or config not found"
    return 1
  fi
  
  echo "Experiment $EXP_NAME completed successfully!"
  return 0
}

# Dictionary to store experiment results
declare -a RESULTS_STATUS=()
declare -a RESULTS_NAMES=()

# Function to log experiment results
log_experiment_result() {
  local EXP_NAME=$1
  local STATUS=$2
  
  RESULTS_NAMES+=("$EXP_NAME")
  RESULTS_STATUS+=("$STATUS")
  
  echo "Experiment $EXP_NAME: $STATUS"
}

# Run adaptive quantization experiments
for K in 5 10 20; do
  EXP_NAME="adaptive_k${K}"
  
  run_experiment "$EXP_NAME" \
    "quantization=adaptive_config" \
    "quantization.k_initial=${K}" \
    "quantization.k_max=$((K*2))" \
    "dataset=synthetic_config"
  
  if [ $? -eq 0 ]; then
    log_experiment_result "$EXP_NAME" "Success"
  else
    log_experiment_result "$EXP_NAME" "Failed"
  fi
done

# Run fixed quantization experiments
for K in 5 10 20; do
  EXP_NAME="fixed_k${K}"
  
  run_experiment "$EXP_NAME" \
    "quantization=fixed_config" \
    "quantization.k_initial=${K}" \
    "dataset=synthetic_config"
  
  if [ $? -eq 0 ]; then
    log_experiment_result "$EXP_NAME" "Success"
  else
    log_experiment_result "$EXP_NAME" "Failed"
  fi
done

# Run learned policy experiment
EXP_NAME="learned_policy"
run_experiment "$EXP_NAME" \
  "quantization=adaptive_config" \
  "quantization.k_initial=10" \
  "quantization.k_max=50" \
  "policy=learned_config" \
  "dataset=synthetic_config"

if [ $? -eq 0 ]; then
  log_experiment_result "$EXP_NAME" "Success"
else
  log_experiment_result "$EXP_NAME" "Failed"
fi

# Run plot generation script
echo
echo "Generating comparative plots..."
PLOT_CMD=("python" "scripts/plot_ablation_results.py")
PLOT_CMD+=("--results_dir" "$RESULTS_DIR")
PLOT_CMD+=("--output_dir" "$RESULTS_DIR/plots")
PLOT_CMD+=("--experiments")

# Add each experiment name individually
for EXP_NAME in "${EXPERIMENT_NAMES[@]}"; do
  PLOT_CMD+=("$EXP_NAME")
done

if [ "$USE_WANDB" = true ]; then
  PLOT_CMD+=("--use_wandb")
  PLOT_CMD+=("--wandb_project" "$WANDB_PROJECT") 
  PLOT_CMD+=("--wandb_entity" "$WANDB_ENTITY")
  PLOT_CMD+=("--wandb_name" "ablation_plots_$(date +%Y%m%d_%H%M%S)")
  
  if [ ! -z "$OFFLINE" ]; then
    PLOT_CMD+=("--offline")
  fi
fi

echo "Running command: ${PLOT_CMD[@]}"
"${PLOT_CMD[@]}"

# Print summary
echo
echo "======================================================================"
echo "Experiment Summary:"
echo "======================================================================"
for i in "${!RESULTS_NAMES[@]}"; do
  echo "${RESULTS_NAMES[$i]}: ${RESULTS_STATUS[$i]}"
done 