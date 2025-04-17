# Temporal GFlowNet Ablation Study Tools

This repository contains several scripts for running and visualizing ablation studies with the Temporal GFlowNet model for time series forecasting.

## Overview of Scripts

1. **`scripts/run_complete_ablation_study.sh`**: Master script that coordinates running experiments and analyzing the results.
2. **`scripts/run_ablation_analysis.sh`**: Focused script for analyzing existing ablation results.
3. **`scripts/visualize_wandb_ablations.py`**: Python script for visualizing results from wandb ablation studies.

## Running a Complete Ablation Study

The complete ablation study script will run multiple experiments with different configurations and then analyze the results.

```bash
# Basic usage with default parameters
./scripts/run_complete_ablation_study.sh

# Running with custom configurations
./scripts/run_complete_ablation_study.sh \
  --datasets synthetic \
  --quantizations adaptive,fixed \
  --k-values 5,10,20 \
  --policy-types uniform,learned \
  --entropy-value 0.01 \
  --epochs 50 \
  --batch-size 64 \
  --results-dir results/my_ablation_study \
  --gpu 0

# Skipping the experiments and just running analysis
./scripts/run_complete_ablation_study.sh \
  --skip-experiments \
  --results-dir results/existing_ablation_study

# Running in W&B offline mode
./scripts/run_complete_ablation_study.sh \
  --offline
```

## Analyzing Existing Ablation Results

If you already have ablation results and just want to generate visualizations, you can use the analysis script:

```bash
# Basic usage with default parameters
./scripts/run_ablation_analysis.sh

# Customizing the analysis
./scripts/run_ablation_analysis.sh \
  --results-dir results/wandb_ablations \
  --output-dir results/new_ablation_plots \
  --datasets "synthetic eeg" \
  --metrics "wql crps mase"
```

## Visualizing W&B Ablation Results Directly

For more direct control over the visualization process, you can use the Python script:

```bash
# Basic usage
python scripts/visualize_wandb_ablations.py

# Customizing the visualization
python scripts/visualize_wandb_ablations.py \
  --results_dir ./results/wandb_ablations \
  --output_dir ./results/custom_ablation_plots \
  --datasets synthetic eeg \
  --metrics wql crps mase \
  --generate_configs_plot
```

## Expected Output

After running the ablation study, you will get:

1. **Summary Tables**:

   - `experiment_summary.csv`: CSV file with key parameters of each experiment
   - `experiment_summary.md`: Markdown version of the summary table

2. **Configuration Visualizations**:

   - `configuration_heatmap.png`: Heatmap comparing numeric parameters across experiments
   - Individual parameter comparison plots (e.g., `config_quantization_k_initial.png`)

3. **Metrics Visualizations** (if metrics data is available):

   - Comparison plots for each metric (e.g., `quantization_wql_comparison.png`)
   - Training curves (if requested and available)

4. **README.md**:
   - A comprehensive README in the results directory summarizing the study

## Examples

### Example 1: Quick Ablation on Synthetic Dataset

```bash
./scripts/run_complete_ablation_study.sh \
  --datasets synthetic \
  --quantizations adaptive \
  --k-values 10,20 \
  --policy-types uniform \
  --epochs 20 \
  --results-dir results/quick_ablation
```

### Example 2: Analyzing Existing Results

```bash
./scripts/run_ablation_analysis.sh \
  --results-dir results/wandb_ablations \
  --output-dir results/enhanced_ablation_plots
```

## Troubleshooting

If you encounter issues:

1. Check that the scripts are executable: `chmod +x scripts/*.sh`
2. Verify that the directories exist and have the expected structure
3. Look at the log files in the `logs/` directory of your results folder
4. Make sure you have all required Python packages installed
