#!/bin/bash
# =============================================================================
# Temporal GFN Ablation Studies Runner - Simplified Version
# =============================================================================
# This script runs a single ablation experiment with adaptive quantization
# to verify everything is working correctly.
#
# Author: Nadhir Vincent Hassen

# Exit only on critical errors
set +e

# Check if the environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No active virtual environment detected."
    echo "Please activate the environment first:"
    echo "  source venv/bin/activate  # For venv"
    echo "  conda activate temporal_gfn  # For conda"
    exit 1
fi

# Default configuration
BASE_NAME="ablation_study"
BASE_RESULTS_DIR="results/ablations"
USE_WANDB=true
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="temporal-gfn-ablations"
WANDB_OFFLINE=false
GPU_ID=0
DATASET="eeg"
EPOCHS=5  # Reduced for testing
CONTEXT_LENGTH=96
PREDICTION_HORIZON=24

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --name)
            BASE_NAME="$2"
            shift
            shift
            ;;
        --results-dir)
            BASE_RESULTS_DIR="$2"
            shift
            shift
            ;;
        --disable-wandb)
            USE_WANDB=false
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --name NAME               Base name for all ablation experiments"
            echo "  --results-dir DIR         Base results directory (default: results/ablations)"
            echo "  --disable-wandb           Disable W&B logging (enabled by default)"
            echo "  --dataset DATASET         Dataset to use (default: eeg)"
            echo "  --epochs N                Number of training epochs (default: 5)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Prepare W&B arguments if enabled
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="use_wandb=true wandb_entity=$WANDB_ENTITY wandb_project=$WANDB_PROJECT"
    
    if [ "$WANDB_OFFLINE" = true ]; then
        WANDB_ARGS="$WANDB_ARGS offline=true"
    fi
fi

# Print header
function print_header() {
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

# Create results directory
mkdir -p "$BASE_RESULTS_DIR"

# Define experiment name
EXPERIMENT_NAME="${BASE_NAME}_adaptive_k10"
EXPERIMENT_DIR="$BASE_RESULTS_DIR/$EXPERIMENT_NAME"

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR/checkpoints"
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/evaluation"

# Save configuration
CONFIG_FILE="$EXPERIMENT_DIR/config.yaml"
cat > "$CONFIG_FILE" << EOF
experiment_name: $EXPERIMENT_NAME
dataset: $DATASET
model_type: transformer
quantization:
  type: adaptive
  k_initial: 10
  adaptive: true
policy_type: uniform
entropy_bonus: 0.01
epochs: $EPOCHS
timestamp: $(date '+%Y-%m-%d %H:%M:%S')
EOF

print_header "Running adaptive quantization (K=10) experiment"

# Build training command with proper Hydra format
TRAIN_CMD="python scripts/train.py \
  --config-name base_config \
  dataset=${DATASET}_config \
  model=transformer_config \
  quantization=adaptive_config \
  quantization.k_initial=10 \
  quantization.adaptive=true \
  policy=uniform_config \
  policy.backward_policy_type=uniform \
  training=base_config \
  training.epochs=$EPOCHS \
  training.batch_size=32 \
  training.learning_rate=0.001 \
  gfn.lambda_entropy=0.01 \
  $WANDB_ARGS \
  results_dir=$EXPERIMENT_DIR"

echo "Training command: $TRAIN_CMD"
echo "Starting training..."
eval "$TRAIN_CMD"

print_header "Training completed"
echo "Results saved to: $EXPERIMENT_DIR" 