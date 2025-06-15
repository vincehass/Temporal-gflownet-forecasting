#!/usr/bin/env python3
"""
Repository Cleanup Script for Temporal GFlowNet Forecasting
Identifies essential vs unnecessary files and creates the missing data module structure.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set

# Essential scripts to keep (ablation-focused)
ESSENTIAL_SCRIPTS = {
    # Core ablation runners
    'run_complete_ablation_study.sh',
    'run_integrated_ablation_study.sh', 
    'run_single_ablation.py',
    'direct_ablation.py',
    'direct_ablation.sh',
    'run_ablation.sh',
    'run_all_ablations.sh',
    
    # Visualization and analysis
    'plot_ablation_results.py',
    'visualize_wandb_ablations.py',
    'enhanced_ablation_viz.py',
    'run_ablation_analysis.sh',
    'compare_ablation_results.py',
    'run_ablation_visualization.py',
    
    # Core functionality
    'train.py',
    'evaluate.py',
    'test_functionality.py',
    
    # Essential supporting scripts
    'run_eeg_experiments.py',  # Has ablation functionality
    'quantization_analysis.py',  # Core to methodology
}

# Scripts that are redundant or unnecessary
REDUNDANT_SCRIPTS = {
    # Duplicate functionality
    'run_experiments.py',  # Superseded by ablation-specific scripts
    'run_experiments.sh',  # Superseded by ablation-specific scripts
    'run_simple_experiment.py',  # Basic functionality covered elsewhere
    'run_eeg_experiment.py',  # Single experiment version
    
    # Visualization redundancy
    'visualize_results.sh',  # Covered by ablation visualizations
    'visualize_forecasts.py',  # Covered by ablation visualizations
    'plot_results.py',  # Covered by ablation plotting
    'plot_synthetic_results.py',  # Non-essential
    'plot_synthetic_results_wandb.py',  # Non-essential
    'plot_consolidated_results.py',  # Redundant with ablation analysis
    
    # W&B utilities (keep minimal set)
    'wandb_tensorboard_integration.py',  # Complex integration not needed
    'test_wandb.py',  # Test file
    'test_wandb_direct.py',  # Test file
    'direct_wandb_test.py',  # Test file
    'ensure_wandb.py',  # Utility
    
    # Synthetic data generation (keep minimal)
    'generate_synthetic_data.py',  # Basic version
    'generate_synthetic_results.py',  # Covered elsewhere
    'generate_and_visualize_samples.py',  # Complex, not essential
    'simulate_samples.py',  # Not essential
    'log_synthetic_to_wandb.py',  # Not essential
    'log_experiment_to_wandb.py',  # Covered by core scripts
    
    # Workflow scripts (keep essential ones)
    'run_visualization_pipeline.sh',  # Complex pipeline
    'run_with_wandb.sh',  # Basic wrapper
    'run_single_wandb_viz.sh',  # Single visualization
    'automated_wandb_workflow.sh',  # Automated workflow
    'batch_log_to_wandb.sh',  # Batch processing
    
    # Analysis scripts (keep essential ones)
    'run_analysis.py',  # Basic analysis
    'run_all_analysis.sh',  # Comprehensive analysis
    'run_all_wandb.sh',  # W&B specific
    'compare_synthetic_results.py',  # Synthetic comparison
    'compare_forecasts.py',  # Forecast comparison
    'compare_quantile_forecasts.py',  # Quantile comparison
    
    # Utility scripts
    'create_simple_config.py',  # Config utility
}

def create_missing_data_module():
    """Create the missing data module structure."""
    data_dir = Path("src/temporal_gfn/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    (data_dir / "__init__.py").write_text("""
\"\"\"
Data module for Temporal GFlowNet.
\"\"\"

from .dataset import TimeSeriesDataset, SyntheticTimeSeriesDataset, create_dataloader
from .scaling import MeanScaler, StandardScaler
from .windowing import create_windows

__all__ = [
    'TimeSeriesDataset',
    'SyntheticTimeSeriesDataset', 
    'create_dataloader',
    'MeanScaler',
    'StandardScaler',
    'create_windows'
]
""")
    
    # Create dataset.py
    (data_dir / "dataset.py").write_text("""
\"\"\"
Dataset classes for time series data.
\"\"\"

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any

class TimeSeriesDataset(Dataset):
    \"\"\"Basic time series dataset.\"\"\"
    
    def __init__(self, data: np.ndarray, context_length: int = 96, 
                 prediction_horizon: int = 24):
        self.data = data
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return max(0, len(self.data) - self.context_length - self.prediction_horizon + 1)
        
    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_length]
        target = self.data[idx + self.context_length:idx + self.context_length + self.prediction_horizon]
        return torch.FloatTensor(context), torch.FloatTensor(target)

class SyntheticTimeSeriesDataset(TimeSeriesDataset):
    \"\"\"Synthetic time series dataset for testing.\"\"\"
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 200,
                 context_length: int = 96, prediction_horizon: int = 24):
        # Generate synthetic data
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, seq_length)
        data = []
        for _ in range(num_samples):
            # Sine wave with noise
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0, 0.1, seq_length)
            series = amplitude * np.sin(freq * t + phase) + noise
            data.append(series)
        
        data = np.array(data)
        super().__init__(data.reshape(-1), context_length, prediction_horizon)

def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                     shuffle: bool = True, **kwargs) -> DataLoader:
    \"\"\"Create a DataLoader for the dataset.\"\"\"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
""")
    
    # Create scaling.py
    (data_dir / "scaling.py").write_text("""
\"\"\"
Scaling utilities for time series data.
\"\"\"

import numpy as np
from typing import Optional

class MeanScaler:
    \"\"\"Scale data by subtracting mean.\"\"\"
    
    def __init__(self):
        self.mean = None
        
    def fit(self, data: np.ndarray) -> 'MeanScaler':
        self.mean = np.mean(data)
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Scaler not fitted")
        return data - self.mean
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Scaler not fitted")
        return data + self.mean

class StandardScaler:
    \"\"\"Standard scaling (z-score normalization).\"\"\"
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray) -> 'StandardScaler':
        self.mean = np.mean(data)
        self.std = np.std(data)
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return data * self.std + self.mean
""")
    
    # Create windowing.py
    (data_dir / "windowing.py").write_text("""
\"\"\"
Windowing utilities for time series data.
\"\"\"

import numpy as np
from typing import Tuple, List

def create_windows(data: np.ndarray, window_size: int, 
                  stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Create sliding windows from time series data.\"\"\"
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def create_context_target_windows(data: np.ndarray, context_length: int,
                                prediction_horizon: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Create context and target windows for forecasting.\"\"\"
    contexts = []
    targets = []
    
    total_length = context_length + prediction_horizon
    for i in range(0, len(data) - total_length + 1, stride):
        context = data[i:i + context_length]
        target = data[i + context_length:i + total_length]
        contexts.append(context)
        targets.append(target)
    
    return np.array(contexts), np.array(targets)
""")
    
    print("✓ Created missing data module structure")

def backup_original_scripts():
    """Create backup of original scripts directory."""
    backup_dir = Path("scripts_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree("scripts", backup_dir)
    print(f"✓ Created backup at {backup_dir}")

def identify_files_to_remove() -> List[Path]:
    """Identify files that should be removed."""
    scripts_dir = Path("scripts")
    files_to_remove = []
    
    for script_file in scripts_dir.glob("*"):
        if script_file.name in REDUNDANT_SCRIPTS:
            files_to_remove.append(script_file)
    
    return files_to_remove

def remove_redundant_files(files_to_remove: List[Path], dry_run: bool = True):
    """Remove redundant files."""
    if not files_to_remove:
        print("No redundant files identified for removal.")
        return
    
    if dry_run:
        print("\n=== DRY RUN: Files that would be removed ===")
        for file_path in files_to_remove:
            print(f"  - {file_path}")
        print(f"\nTotal: {len(files_to_remove)} files")
    else:
        print("\n=== Removing redundant files ===")
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"  ✓ Removed {file_path}")
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path}: {e}")

def create_essential_scripts_list():
    """Create a README listing essential scripts."""
    readme_content = """# Essential Ablation Scripts

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
"""
    
    Path("ESSENTIAL_SCRIPTS.md").write_text(readme_content)
    print("✓ Created ESSENTIAL_SCRIPTS.md")

def main():
    """Main cleanup function."""
    print("🧹 Temporal GFlowNet Repository Cleanup")
    print("=" * 50)
    
    # Create missing data module
    print("\n1. Creating missing data module...")
    create_missing_data_module()
    
    # Create backup
    print("\n2. Creating backup of scripts...")
    backup_original_scripts()
    
    # Identify redundant files
    print("\n3. Identifying redundant files...")
    files_to_remove = identify_files_to_remove()
    
    # Show what would be removed (dry run)
    print("\n4. Preview of cleanup...")
    remove_redundant_files(files_to_remove, dry_run=True)
    
    # Create essential scripts documentation
    print("\n5. Creating documentation...")
    create_essential_scripts_list()
    
    print("\n" + "=" * 50)
    print("✓ Cleanup analysis complete!")
    print("\nNext steps:")
    print("1. Review the files listed above")
    print("2. Run with --execute to perform actual cleanup")
    print("3. Test essential scripts with: python test_essential_scripts.py")
    
    # Ask for confirmation to proceed
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        response = input("\nProceed with actual file removal? (y/N): ")
        if response.lower() == 'y':
            print("\n6. Executing cleanup...")
            remove_redundant_files(files_to_remove, dry_run=False)
            print("✓ Cleanup completed!")
        else:
            print("Cleanup cancelled.")
    else:
        print("\nTo execute cleanup, run: python cleanup_repository.py --execute")

if __name__ == "__main__":
    main() 