#!/usr/bin/env python3
"""
Script for visualizing results from ablation studies.
This script loads metrics from different experiments, creates comparison plots,
and visualizes training curves from tensorboard logs.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
import re
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List, Tuple, Optional, Union, Any

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
COLORS = sns.color_palette("Set2", 10)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot ablation study results')
    parser.add_argument('--results_dir', type=str, default='./results/ablations',
                        help='Directory containing ablation study results')
    parser.add_argument('--output_dir', type=str, default='./results/ablation_plots',
                        help='Directory to save plots')
    parser.add_argument('--study_type', type=str, default='all',
                        help='Type of ablation study to plot (e.g., "quantization", "entropy", "all")')
    parser.add_argument('--datasets', type=str, nargs='+', default=['electricity', 'traffic', 'ETTm1', 'ETTh1'],
                        help='Datasets to include in plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot')
    parser.add_argument('--plot_training_curves', action='store_true',
                        help='Whether to plot training curves from tensorboard logs')
    parser.add_argument('--wandb_dir', type=str, default='./results/wandb_ablations',
                        help='Directory containing Weights & Biases logs')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str, study_type: str = 'all') -> List[str]:
    """Find all experiment directories for the given study type."""
    if study_type == 'all':
        pattern = os.path.join(results_dir, 'ablation_study_*')
    else:
        pattern = os.path.join(results_dir, f'ablation_study_{study_type}*')
    
    return sorted(glob.glob(pattern))

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
    return load_metrics_file(metrics_file)

def extract_experiment_params(experiment_dir: str) -> Dict[str, str]:
    """Extract parameters from experiment directory name."""
    base_name = os.path.basename(experiment_dir)
    
    # Common parameters to extract
    params = {}
    
    # Extract quantization type and parameters
    quant_match = re.search(r'quantization_(\w+)(?:_k(\d+))?', base_name)
    if quant_match:
        params['quant_type'] = quant_match.group(1)
        if quant_match.group(2):
            params['k_value'] = quant_match.group(2)
    
    # Extract entropy parameters
    entropy_match = re.search(r'entropy_(\d+\.\d+)', base_name)
    if entropy_match:
        params['entropy'] = entropy_match.group(1)
    
    # Extract policy type
    policy_match = re.search(r'policy_(\w+)', base_name)
    if policy_match:
        params['policy'] = policy_match.group(1)
        
    # If no parameters were extracted, use the directory name as the experiment name
    if not params:
        params['experiment'] = base_name
    
    return params

def get_experiment_label(params: Dict[str, str]) -> str:
    """Generate a readable label for the experiment based on its parameters."""
    if 'quant_type' in params:
        label = f"{params['quant_type'].capitalize()}"
        if 'k_value' in params:
            label += f" (k={params['k_value']})"
        return label
    elif 'entropy' in params:
        return f"Entropy={params['entropy']}"
    elif 'policy' in params:
        return f"Policy={params['policy']}"
    elif 'experiment' in params:
        # Clean up the experiment name
        name = params['experiment'].replace('ablation_study_', '')
        return name.replace('_', ' ').title()
    else:
        return "Unknown"

def load_all_metrics(experiment_dirs: List[str], datasets: List[str]) -> pd.DataFrame:
    """Load metrics from all experiments and datasets into a DataFrame."""
    rows = []
    
    for experiment_dir in experiment_dirs:
        params = extract_experiment_params(experiment_dir)
        experiment_label = get_experiment_label(params)
        
        for dataset in datasets:
            metrics = load_metrics(experiment_dir, dataset)
            
            if not metrics:
                continue
                
            # Extract the main metrics we care about
            row = {
                'experiment': os.path.basename(experiment_dir),
                'experiment_label': experiment_label,
                'dataset': dataset,
            }
            
            # Add all metrics from the metrics file
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metrics (like per-quantile WQL)
                    for sub_name, sub_value in metric_value.items():
                        row[f"{metric_name}_{sub_name}"] = sub_value
                else:
                    row[metric_name] = metric_value
                    
            # Add experiment parameters
            for param_name, param_value in params.items():
                row[param_name] = param_value
                
            rows.append(row)
    
    return pd.DataFrame(rows)

def load_tensorboard_logs(experiment_dir: str, dataset: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load tensorboard logs for a specific experiment and dataset."""
    log_dir = os.path.join(experiment_dir, 'logs', dataset)
    
    if not os.path.exists(log_dir):
        return {}
        
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        return {}
        
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Extract scalar values
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = [(event.step, event.value) for event in events]
        
    return scalars

def plot_metrics_comparison(metrics_df: pd.DataFrame, metric: str, datasets: List[str], 
                           output_dir: str, study_type: str):
    """Plot comparison of a specific metric across experiments for each dataset."""
    if not metrics_df.empty:
        # Filter by metric availability
        valid_df = metrics_df[metrics_df[metric].notna()]
        
        if valid_df.empty:
            print(f"No valid data for metric: {metric}")
            return
            
        # Create a figure with subplots for each dataset
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 6), sharey=True)
        
        if n_datasets == 1:
            axes = [axes]
            
        for i, dataset in enumerate(datasets):
            dataset_df = valid_df[valid_df['dataset'] == dataset]
            
            if dataset_df.empty:
                axes[i].text(0.5, 0.5, f"No data for {dataset}", 
                             ha='center', va='center', fontsize=12)
                axes[i].set_title(dataset)
                continue
                
            # Sort by experiment label
            dataset_df = dataset_df.sort_values('experiment_label')
            
            # Create barplot
            sns.barplot(x='experiment_label', y=metric, data=dataset_df, ax=axes[i], 
                       palette=COLORS[:len(dataset_df)])
            
            # Set title and rotate x-labels
            axes[i].set_title(dataset)
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Format y-axis
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            
            # Add values on top of bars
            for j, p in enumerate(axes[i].patches):
                value = dataset_df.iloc[j][metric]
                axes[i].annotate(f"{value:.3f}", 
                                (p.get_x() + p.get_width()/2., p.get_height()),
                                ha='center', va='bottom', fontsize=9)
        
        # Add overall title and y-label
        fig.suptitle(f"{metric.upper()} Comparison", fontsize=16)
        fig.text(0.04, 0.5, f"{metric.upper()}", va='center', rotation='vertical', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, left=0.1)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{study_type}_{metric}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_training_curves(experiment_dirs: List[str], datasets: List[str], output_dir: str, study_type: str):
    """Plot training curves from tensorboard logs for each experiment and dataset."""
    # Define metrics to plot from tensorboard logs
    tb_metrics = ['train/loss', 'val/loss', 'train/wql', 'val/wql']
    
    for dataset in datasets:
        # Collect data from all experiments for this dataset
        experiment_data = {}
        
        for experiment_dir in experiment_dirs:
            params = extract_experiment_params(experiment_dir)
            experiment_label = get_experiment_label(params)
            
            logs = load_tensorboard_logs(experiment_dir, dataset)
            
            if logs:
                experiment_data[experiment_label] = logs
        
        if not experiment_data:
            print(f"No tensorboard logs found for dataset: {dataset}")
            continue
            
        # Plot each metric
        for metric in tb_metrics:
            if not any(metric in data for data in experiment_data.values()):
                continue
                
            plt.figure(figsize=(10, 6))
            
            for i, (exp_label, logs) in enumerate(experiment_data.items()):
                if metric in logs:
                    steps, values = zip(*logs[metric])
                    plt.plot(steps, values, label=exp_label, color=COLORS[i % len(COLORS)], 
                            linewidth=2, marker='o', markersize=4, markevery=max(1, len(steps)//10))
            
            plt.title(f"{dataset} - {metric}")
            plt.xlabel('Steps')
            plt.ylabel(metric.split('/')[-1].upper())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            clean_metric = metric.replace('/', '_')
            plt.savefig(os.path.join(output_dir, f"{study_type}_{dataset}_{clean_metric}_curve.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def plot_quantile_metrics(metrics_df: pd.DataFrame, datasets: List[str], output_dir: str, study_type: str):
    """Plot quantile-based metrics (like WQL per quantile) across experiments."""
    # Check if we have quantile-specific metrics
    quantile_columns = [col for col in metrics_df.columns if col.startswith('wql_') and col != 'wql']
    
    if not quantile_columns:
        return
        
    quantile_values = sorted([float(col.split('_')[1]) for col in quantile_columns])
    
    for dataset in datasets:
        dataset_df = metrics_df[metrics_df['dataset'] == dataset]
        
        if dataset_df.empty:
            continue
            
        plt.figure(figsize=(12, 6))
        
        for i, row in dataset_df.iterrows():
            exp_label = row['experiment_label']
            
            # Extract quantile values for this experiment
            quantile_scores = [row[f'wql_{q}'] for q in quantile_values if f'wql_{q}' in row]
            
            if quantile_scores:
                plt.plot(quantile_values[:len(quantile_scores)], quantile_scores, 
                        marker='o', linewidth=2, label=exp_label)
        
        plt.title(f"{dataset} - Quantile Loss by Probability Level")
        plt.xlabel('Quantile')
        plt.ylabel('Quantile Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{study_type}_{dataset}_quantile_loss.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_dataset_comparison(metrics_df: pd.DataFrame, metrics: List[str], output_dir: str, study_type: str):
    """Plot comparison of metrics across datasets."""
    if metrics_df.empty:
        return
        
    for metric in metrics:
        valid_df = metrics_df[metrics_df[metric].notna()]
        
        if valid_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Group by experiment_label and dataset, then calculate mean
        grouped = valid_df.groupby(['experiment_label', 'dataset'])[metric].mean().reset_index()
        
        # Create the plot
        sns.barplot(x='dataset', y=metric, hue='experiment_label', data=grouped)
        
        plt.title(f'{metric.upper()} across datasets')
        plt.xlabel('Dataset')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{study_type}_dataset_comparison_{metric}.png'), dpi=300)
        plt.close()

def plot_quantization_metrics(experiment_dirs: List[str], datasets: List[str], output_dir: str):
    """
    Plot detailed quantization metrics from TensorBoard logs.
    
    This function creates visualizations specifically for quantization ablation analysis:
    - K progression over time for adaptive vs fixed
    - Bin usage efficiency (unique action ratio)
    - Normalized bin entropy
    - Unique sequences generated
    - Relationship between bin count and metrics
    """
    # Prepare data structure for collecting metrics
    quant_metrics = defaultdict(lambda: defaultdict(list))
    
    # Collect all quantization metrics from tensorboard logs
    for experiment_dir in experiment_dirs:
        params = extract_experiment_params(experiment_dir)
        experiment_label = get_experiment_label(params)
        
        for dataset in datasets:
            logs = load_tensorboard_logs(experiment_dir, dataset)
            
            # Extract key quantization metrics
            metrics_to_extract = [
                'quant/unique_action_ratio',
                'quant/unique_sequences',
                'quant/bin_entropy',
                'quantization/k',
                'quantization/var_mean_ratio',
                'quantization/eta_e',
                'quantization/delta_t'
            ]
            
            for metric in metrics_to_extract:
                if metric in logs:
                    # Store step, value pairs with experiment label
                    for step, value in logs[metric]:
                        quant_metrics[metric][experiment_label].append((step, value))
    
    # K progression plot
    if 'quantization/k' in quant_metrics and quant_metrics['quantization/k']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quantization/k'].items():
            steps, k_values = zip(*values)
            plt.plot(steps, k_values, label=exp_label, linewidth=2, marker='o', markersize=4)
        
        plt.title('Quantization Bins (K) Progression Over Training')
        plt.xlabel('Training Step')
        plt.ylabel('Number of Bins (K)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_k_progression.png'), dpi=300)
        plt.close()
    
    # Bin usage efficiency plot
    if 'quant/unique_action_ratio' in quant_metrics and quant_metrics['quant/unique_action_ratio']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quant/unique_action_ratio'].items():
            steps, ratios = zip(*values)
            plt.plot(steps, ratios, label=exp_label, linewidth=2, marker='o', markersize=4)
        
        plt.title('Bin Usage Efficiency (Unique Actions / Total Bins)')
        plt.xlabel('Training Step')
        plt.ylabel('Efficiency Ratio')
        plt.ylim(0, 1.05)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_bin_efficiency.png'), dpi=300)
        plt.close()
    
    # Bin entropy plot
    if 'quant/bin_entropy' in quant_metrics and quant_metrics['quant/bin_entropy']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quant/bin_entropy'].items():
            steps, entropy = zip(*values)
            plt.plot(steps, entropy, label=exp_label, linewidth=2, marker='o', markersize=4)
        
        plt.title('Normalized Bin Usage Entropy')
        plt.xlabel('Training Step')
        plt.ylabel('Entropy (higher = more uniform bin usage)')
        plt.ylim(0, 1.05)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_bin_entropy.png'), dpi=300)
        plt.close()
    
    # Unique sequences plot
    if 'quant/unique_sequences' in quant_metrics and quant_metrics['quant/unique_sequences']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quant/unique_sequences'].items():
            steps, seq_counts = zip(*values)
            plt.plot(steps, seq_counts, label=exp_label, linewidth=2, marker='o', markersize=4)
        
        plt.title('Unique Trajectories Generated')
        plt.xlabel('Training Step')
        plt.ylabel('Count of Unique Sequences')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_unique_sequences.png'), dpi=300)
        plt.close()
    
    # Delta_t plot (when K changes)
    if 'quantization/delta_t' in quant_metrics and quant_metrics['quantization/delta_t']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quantization/delta_t'].items():
            # Filter out zero values (no change)
            non_zero_values = [(step, val) for step, val in values if val != 0]
            if non_zero_values:
                steps, deltas = zip(*non_zero_values)
                plt.scatter(steps, deltas, label=exp_label, s=100, alpha=0.7)
        
        plt.title('K Change Events (Delta_t)')
        plt.xlabel('Training Step')
        plt.ylabel('Change Direction (+1 = increase, -1 = decrease)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_delta_t.png'), dpi=300)
        plt.close()
    
    # Learning indicator plot
    if 'quantization/eta_e' in quant_metrics and quant_metrics['quantization/eta_e']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quantization/eta_e'].items():
            steps, indicators = zip(*values)
            plt.plot(steps, indicators, label=exp_label, linewidth=2, marker='o', markersize=4)
            
            # Also plot the threshold line if this is adaptive quantization
            if 'adaptive' in exp_label.lower():
                # Typical threshold value (could extract from config if available)
                threshold = 0.02
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                           label='Adaptation Threshold' if exp_label == list(quant_metrics['quantization/eta_e'].keys())[0] else None)
        
        plt.title('Learning Progress Indicator (eta_e)')
        plt.xlabel('Training Step')
        plt.ylabel('Indicator Value (lower = more stability)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_learning_indicator.png'), dpi=300)
        plt.close()
    
    # Variance-to-mean ratio plot
    if 'quantization/var_mean_ratio' in quant_metrics and quant_metrics['quantization/var_mean_ratio']:
        plt.figure(figsize=(10, 6))
        
        for exp_label, values in quant_metrics['quantization/var_mean_ratio'].items():
            steps, ratios = zip(*values)
            plt.plot(steps, ratios, label=exp_label, linewidth=2, marker='o', markersize=4)
        
        plt.title('Reward Variance-to-Mean Ratio')
        plt.xlabel('Training Step')
        plt.ylabel('Ratio (higher = more variability)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantization_var_mean_ratio.png'), dpi=300)
        plt.close()

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.results_dir, args.study_type)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {args.results_dir} for study type '{args.study_type}'")
        return
        
    print(f"Found {len(experiment_dirs)} experiment directories.")
    
    # Load metrics into DataFrame
    metrics_df = load_all_metrics(experiment_dirs, args.datasets)
    
    if metrics_df.empty:
        print("No metrics found. Check that evaluation has been run.")
        return
        
    print(f"Loaded metrics for {len(metrics_df)} experiment-dataset combinations.")
    
    # Create plots
    for metric in args.metrics:
        plot_metrics_comparison(metrics_df, metric, args.datasets, args.output_dir, args.study_type)
        
    plot_dataset_comparison(metrics_df, args.metrics, args.output_dir, args.study_type)
    
    # Plot quantile metrics if WQL is in the metrics
    if 'wql' in args.metrics:
        plot_quantile_metrics(metrics_df, args.datasets, args.output_dir, args.study_type)
    
    # Plot training curves if requested
    if args.plot_training_curves:
        plot_training_curves(experiment_dirs, args.datasets, args.output_dir, args.study_type)
    
    # Plot quantization-specific metrics for quantization studies
    if args.study_type in ['all', 'quantization']:
        plot_quantization_metrics(experiment_dirs, args.datasets, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 