# Temporal GFlowNet Ablation Studies

This directory contains scripts for generating, evaluating, and visualizing results from ablation studies.

## Scripts Overview

### `generate_synthetic_results.py`

Generates synthetic data for testing visualization scripts. This is useful when real experiment data is not available.

Usage:

```
python scripts/generate_synthetic_results.py
```

This script:

- Creates synthetic metrics data for 5 ablation configurations (adaptive_k10, adaptive_k20, fixed_k10, fixed_k20, learned_policy)
- Generates training curves with realistic patterns
- Saves data in the expected directory structure at `results/synthetic_data/`

### `plot_results.py`

The original script for visualizing results from wandb-based ablation studies.

Usage:

```
python scripts/plot_results.py --results_dir <path_to_results> --output_dir <path_to_output>
```

### `plot_synthetic_results.py`

An enhanced script specifically designed to visualize synthetic results from ablation studies.

Usage:

```
python scripts/plot_synthetic_results.py --results_dir <path_to_results> --output_dir <path_to_output> [--experiments exp1 exp2 ...]
```

Arguments:

- `--results_dir`: Directory containing experiment results (default: './results/synthetic_data')
- `--output_dir`: Directory to save plots (default: './results/synthetic_plots')
- `--experiments`: Optional list of specific experiments to include

This script generates:

1. Overall metrics comparison plots
2. Per-experiment metrics plots
3. Per-category metrics plots (adaptive vs fixed vs learned policy)
4. Per-horizon metrics plots
5. Training curves for loss, reward, entropy, and quantization bins (k)

## Directory Structure

For visualization to work correctly, your experiments should follow this structure:

```
results/
  experiment_name/
    config.yaml              # Configuration file
    evaluation/
      metrics.json           # Evaluation metrics
    logs/
      training_curves.json   # Training curves data
```

## Metrics Format

The `metrics.json` file should have the following structure:

```json
{
  "overall": {
    "wql": 0.24,
    "crps": 0.17,
    "mase": 1.10
  },
  "per_horizon": {
    "wql": [0.24, 0.25, 0.26, ...],
    "crps": [0.17, 0.18, 0.19, ...],
    "mase": [1.10, 1.16, 1.22, ...]
  },
  "metadata": {
    "time": "2025-04-16 16:33:10",
    "config": {
      "quantization": "adaptive",
      "k": 10,
      "policy_type": "uniform",
      "entropy_bonus": 0.01
    }
  }
}
```

## Training Curves Format

The `training_curves.json` file should have the following structure:

```json
{
  "loss": [2.59, 2.28, 2.07, ...],
  "reward": [0.39, 0.58, 0.43, ...],
  "entropy": [1.0, 1.0, 1.0, ...],
  "k": [10.0, 10.0, 10.0, ...],
  "epochs": [1, 2, 3, ...]
}
```
