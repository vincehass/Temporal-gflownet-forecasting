#!/usr/bin/env python3
"""
Script for visualizing results from wandb ablation studies.

This script loads data from the results/wandb_ablations directory,
processes configuration files and any available metrics,
and generates comparison plots for different ablation experiments.

Usage:
    python scripts/visualize_wandb_ablations.py [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional, Union, Any

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
COLORS = sns.color_palette("Set2", 10)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize wandb ablation study results')
    parser.add_argument('--results_dir', type=str, default='./results/wandb_ablations',
                        help='Directory containing wandb ablation study results')
    parser.add_argument('--output_dir', type=str, default='./results/wandb_ablations/plots',
                        help='Directory to save plots')
    parser.add_argument('--datasets', type=str, nargs='+', default=['synthetic'],
                        help='Datasets to include in plots (if available)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot (if available)')
    parser.add_argument('--generate_configs_plot', action='store_true',
                        help='Generate plots comparing configuration parameters across experiments')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str) -> List[str]:
    """Find all experiment directories in the results directory."""
    # Exclude the plots directory
    return [d for d in glob.glob(os.path.join(results_dir, "*")) 
            if os.path.isdir(d) and not os.path.basename(d) == "plots"]

def load_config(experiment_dir: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    config_file = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_file):
        print(f"No config file found in {experiment_dir}")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}

def load_metrics_file(metrics_file: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        return {}

def load_metrics(experiment_dir: str, dataset: str) -> Dict[str, Any]:
    """Load metrics for a specific experiment and dataset."""
    metrics_file = os.path.join(experiment_dir, 'evaluation', f'{dataset}_metrics.json')
    
    # Check if metrics file exists
    if not os.path.exists(metrics_file):
        return {}
    
    return load_metrics_file(metrics_file)

def extract_experiment_params(experiment_dir: str) -> Dict[str, str]:
    """Extract parameters from experiment directory name."""
    base_name = os.path.basename(experiment_dir)
    
    # Extract common parameters
    params = {'experiment': base_name}
    
    if 'adaptive' in base_name:
        params['quant_type'] = 'adaptive'
        k_match = re.search(r'k(\d+)', base_name)
        if k_match:
            params['k_value'] = k_match.group(1)
    elif 'fixed' in base_name:
        params['quant_type'] = 'fixed'
        k_match = re.search(r'k(\d+)', base_name)
        if k_match:
            params['k_value'] = k_match.group(1)
    
    if 'learned_policy' in base_name:
        params['policy'] = 'learned'
    elif 'uniform_policy' in base_name:
        params['policy'] = 'uniform'
    
    # If this is a specific named policy experiment and no policy type was extracted
    if 'policy' not in params and 'learned_policy' == base_name:
        params['policy'] = 'learned'
        
    return params

def get_experiment_label(params: Dict[str, str], config: Dict[str, Any]) -> str:
    """Generate a readable label for the experiment based on its parameters."""
    label_parts = []
    
    # Extract from directory name params
    if 'quant_type' in params:
        quant_label = f"{params['quant_type'].capitalize()}"
        if 'k_value' in params:
            quant_label += f" (k={params['k_value']})"
        label_parts.append(quant_label)
    
    if 'policy' in params:
        label_parts.append(f"Policy={params['policy']}")
    
    # Extract from config if available
    if config and not label_parts:
        if 'quantization' in config:
            quant_config = config['quantization']
            quant_type = 'Adaptive' if quant_config.get('adaptive', False) else 'Fixed'
            k_value = quant_config.get('k_initial', 'N/A')
            label_parts.append(f"{quant_type} (k={k_value})")
        
        if 'policy' in config:
            policy_type = config.get('policy', {}).get('backward_policy_type', 'N/A')
            label_parts.append(f"Policy={policy_type}")
    
    # If we still have no parts, use the experiment name
    if not label_parts and 'experiment' in params:
        return params['experiment'].replace('_', ' ').title()
    
    return " - ".join(label_parts)

def load_all_configs(experiment_dirs: List[str]) -> pd.DataFrame:
    """Load configurations from all experiments into a DataFrame."""
    rows = []
    
    for experiment_dir in experiment_dirs:
        params = extract_experiment_params(experiment_dir)
        config = load_config(experiment_dir)
        
        # Skip if no config found
        if not config:
            continue
            
        # Extract the basic info
        row = {
            'experiment': os.path.basename(experiment_dir),
            'experiment_label': get_experiment_label(params, config)
        }
        
        # Add experiment parameters
        for param_name, param_value in params.items():
            row[param_name] = param_value
        
        # Add flattened config parameters
        def flatten_config(config_dict, prefix=''):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    flatten_config(value, f"{prefix}{key}.")
                else:
                    row[f"{prefix}{key}"] = value
        
        flatten_config(config)
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_config_comparison(configs_df: pd.DataFrame, output_dir: str):
    """Create comparison plots for important configuration parameters."""
    if configs_df.empty:
        print("No configuration data available for plotting")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameters to plot
    param_groups = {
        'quantization': ['quantization.k_initial', 'quantization.k_max', 'quantization.adaptive', 'quantization.ste'],
        'model': ['model.d_model', 'model.nhead', 'model.nlayers', 'model.dropout'],
        'training': ['training.epochs', 'training.batch_size', 'training.learning_rate'],
        'gfn': ['gfn.lambda_entropy', 'gfn.Z_init', 'gfn.Z_lr']
    }
    
    for group_name, params in param_groups.items():
        # Filter parameters that exist in the DataFrame
        available_params = [p for p in params if p in configs_df.columns]
        
        if not available_params:
            continue
            
        # Create a plot for each parameter in the group
        for param in available_params:
            param_name = param.split('.')[-1]  # Get the last part of the parameter name
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by experiment label
            sorted_df = configs_df.sort_values('experiment_label')
            
            # Check if parameter is numeric or categorical
            if pd.api.types.is_numeric_dtype(sorted_df[param].dtype):
                # Numeric parameter - use bar plot
                sns.barplot(x='experiment_label', y=param, data=sorted_df, ax=ax)
                
                # Add values on top of bars
                for i, v in enumerate(sorted_df[param]):
                    ax.text(i, v + 0.01 * max(sorted_df[param]), 
                            f"{v}", ha='center', va='bottom')
            else:
                # Categorical parameter - create countplot with hue
                counts = sorted_df.groupby(['experiment_label', param]).size().unstack(fill_value=0)
                counts.plot.bar(stacked=True, ax=ax)
                ax.legend(title=param_name)
            
            # Set title and labels
            ax.set_title(f"{param_name.upper()} Comparison")
            ax.set_xlabel('')
            ax.set_ylabel(param_name)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"config_{group_name}_{param_name}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def generate_summary_table(configs_df: pd.DataFrame, output_dir: str):
    """Generate a summary table of all experiments and their key configurations."""
    if configs_df.empty:
        return
    
    # Select important columns for the summary
    important_cols = ['experiment_label']
    
    # Add columns if they exist
    possible_cols = [
        'quant_type', 'k_value', 'policy',
        'quantization.adaptive', 'quantization.k_initial', 
        'policy.backward_policy_type',
        'gfn.lambda_entropy',
        'training.epochs', 'training.batch_size'
    ]
    
    for col in possible_cols:
        if col in configs_df.columns:
            important_cols.append(col)
    
    # Create summary table
    summary_df = configs_df[important_cols].copy()
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'experiment_summary.csv'), index=False)
    
    # Generate a markdown table for README
    with open(os.path.join(output_dir, 'experiment_summary.md'), 'w') as f:
        f.write("# Ablation Study Experiments Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

def plot_configuration_heatmap(configs_df: pd.DataFrame, output_dir: str):
    """Create a heatmap visualization of experiment configurations."""
    if configs_df.empty:
        return
    
    # Identify numerical columns for the heatmap
    numeric_cols = [col for col in configs_df.columns 
                   if pd.api.types.is_numeric_dtype(configs_df[col].dtype) and col != 'experiment']
    
    if not numeric_cols:
        return
    
    # Scale the numeric values to [0,1] range for better visualization
    scaled_df = configs_df.copy()
    for col in numeric_cols:
        col_min = configs_df[col].min()
        col_max = configs_df[col].max()
        if col_max > col_min:
            scaled_df[col] = (configs_df[col] - col_min) / (col_max - col_min)
    
    # Create heatmap
    plt.figure(figsize=(12, len(configs_df) * 0.8))
    
    # Use scaled values for the heatmap
    heatmap_data = scaled_df.set_index('experiment_label')[numeric_cols]
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=configs_df.set_index('experiment_label')[numeric_cols], 
               fmt='.2g', cmap='YlGnBu', linewidths=0.5)
    
    plt.title('Configuration Parameters Across Experiments')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"configuration_heatmap.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiment_comparison(experiment_dirs: List[str], output_dir: str, 
                             metrics: List[str] = None, datasets: List[str] = None):
    """Generate comparison plots based on the configurations."""
    # Load all configs
    configs_df = load_all_configs(experiment_dirs)
    
    if configs_df.empty:
        print("No configuration data found for experiments")
        return
    
    # Generate summary table
    generate_summary_table(configs_df, output_dir)
    
    # Plot configuration comparison
    print("Generating configuration comparison plots...")
    plot_config_comparison(configs_df, output_dir)
    
    # Plot configuration heatmap
    print("Generating configuration heatmap...")
    plot_configuration_heatmap(configs_df, output_dir)
    
    # Check if we have metrics files
    has_metrics = False
    for exp_dir in experiment_dirs:
        for dataset in datasets or ['synthetic']:
            metrics_file = os.path.join(exp_dir, 'evaluation', f'{dataset}_metrics.json')
            if os.path.exists(metrics_file):
                has_metrics = True
                break
    
    # If we have metrics, we could add plots here
    if has_metrics and metrics:
        print("Metrics files found - plotting metrics...")
        # Add code for metrics plotting if needed
    else:
        print("No metrics files found - skipping metrics plots")
    
    print(f"All plots saved to {output_dir}")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.results_dir)
    
    if not experiment_dirs:
        print(f"No experiment directories found in: {args.results_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Generate plots
    plot_experiment_comparison(
        experiment_dirs=experiment_dirs,
        output_dir=args.output_dir,
        metrics=args.metrics,
        datasets=args.datasets
    )

if __name__ == "__main__":
    main() 