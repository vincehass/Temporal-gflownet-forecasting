# Essential Ablation Scripts

This repository has been cleaned to focus on ablation studies for Temporal GFlowNet forecasting.

## Core Ablation Scripts

### Main Ablation Runners
- `run_complete_ablation_study.sh` - Master script for comprehensive ablation studies
- `run_integrated_ablation_study.sh` - Integrated ablation runner with W&B
- `run_single_ablation.py` - Run individual ablation experiments
- `direct_ablation.py` - Direct ablation experiment runner
- `run_ablation.sh` - Simplified ablation runner

### Visualization & Analysis
- `plot_ablation_results.py` - Core ablation result visualization
- `visualize_wandb_ablations.py` - W&B-specific ablation visualization
- `enhanced_ablation_viz.py` - Enhanced ablation visualization
- `run_ablation_analysis.sh` - Analysis of ablation results
- `compare_ablation_results.py` - Cross-ablation comparison

### Core Functionality
- `train.py` - Model training script
- `evaluate.py` - Model evaluation script
- `test_functionality.py` - Functionality testing
- `quantization_analysis.py` - Quantization method analysis

## Usage

Start with the complete ablation study:
```bash
./scripts/run_complete_ablation_study.sh
```

Or run individual components:
```bash
python scripts/run_single_ablation.py --help
./scripts/run_ablation_analysis.sh --help
```

## Data Module

The missing data module has been created at `src/temporal_gfn/data/` with:
- `dataset.py` - Dataset classes
- `scaling.py` - Data scaling utilities  
- `windowing.py` - Windowing utilities
