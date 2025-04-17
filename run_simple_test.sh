#!/bin/bash
# Simple script to run a test experiment

# Create results directory
RESULTS_DIR="results/eeg_simple_test"
mkdir -p "$RESULTS_DIR"

# Create a modified config
cat > "$RESULTS_DIR/config.yaml" << EOF
# Base configuration for Temporal GFN model

# Set defaults list for Hydra
defaults:
  - _self_
  - dataset: eeg_config
  - model: transformer_config
  - quantization: adaptive_config
  - policy: uniform_config
  - training: base_config

# Model configuration
model:
  d_model: 64  # Smaller model for testing
  nhead: 4
  d_hid: 128
  nlayers: 2
  dropout: 0.1
  uniform_init: true

# GFN configuration
gfn:
  Z_init: 0.0
  Z_lr: 0.01
  lambda_entropy: 0.01

# Quantization configuration
quantization:
  adaptive: true
  k_initial: 5
  k_max: 20
  vmin: -10.0
  vmax: 10.0
  lambda_adapt: 0.9
  epsilon_adapt: 0.02
  delta_adapt: 5
  update_interval: 1000

# Policy configuration
policy:
  backward_policy_type: uniform
  rand_action_prob: 0.01

# Dataset configuration
dataset:
  type: eeg
  context_length: 24  # Shorter context for testing
  prediction_horizon: 8
  scaler_type: mean
  stride: 1
  sample_stride: 5
  return_indices: false

# Training configuration
training:
  epochs: 3
  batch_size: 16
  lr: 0.001  # Fixed to use 'lr' instead of 'learning_rate'
  weight_decay: 0.0001
  use_lr_scheduler: false
  grad_clip_norm: 1.0
  num_workers: 2

# Validation configuration
validation:
  enabled: true

# Evaluation configuration
evaluation:
  batch_size: 16
  num_samples: 10
  quantiles: [0.1, 0.5, 0.9]
  seasonality: 1
  save_forecasts: false

# Logging configuration
logging:
  results_dir: ${results_dir}
  log_interval: 10
  checkpoint_interval: 1

# Default results directory
results_dir: ${results_dir}

# W&B configuration
use_wandb: false
wandb_entity: nadhirvincenthassen
wandb_project: temporal-gfn
gpu: 0
seed: 42
EOF

# Run the training script
python scripts/train.py \
  --config-name config \
  --config-path "$RESULTS_DIR" \
  results_dir="$RESULTS_DIR"

echo "Experiment completed! Check results in $RESULTS_DIR" 