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
                'quantization/delta_t',
                'train/reward',
                'train/entropy',
                'train/loss'
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
        
    # NEW: Plot the Adaptive Update Factor Components
    # Based on equation (4) in the paper: η_e = 1 + λ * (max(0, ε - ΔR_e)/ε + (1 - H_e))
    # Extract reward improvement and entropy to calculate the components
    calculate_improvement_confidence_signals = any('adaptive' in label.lower() for label in quant_metrics['quantization/eta_e'].keys())
    
    if calculate_improvement_confidence_signals and 'train/reward' in quant_metrics and 'train/entropy' in quant_metrics:
        plt.figure(figsize=(12, 8))
        
        for exp_label in [label for label in quant_metrics['quantization/eta_e'].keys() if 'adaptive' in label.lower()]:
            if exp_label in quant_metrics['train/reward'] and exp_label in quant_metrics['train/entropy']:
                reward_steps, reward_values = zip(*quant_metrics['train/reward'][exp_label])
                entropy_steps, entropy_values = zip(*quant_metrics['train/entropy'][exp_label])
                
                # Convert to numpy arrays for easier manipulation
                steps = np.array(reward_steps)
                rewards = np.array(reward_values)
                entropies = np.array(entropy_values)
                
                # Calculate ΔR_e (simple difference from previous rewards as an approximation)
                delta_rewards = np.zeros_like(rewards)
                delta_rewards[1:] = rewards[1:] - rewards[:-1]
                
                # Normalize entropy to [0, 1] range
                max_entropy = np.log(10)  # Assuming K=10 for simplicity
                normalized_entropy = entropies / max_entropy
                
                # Set threshold parameters
                epsilon = 0.02  # Reward improvement threshold from paper
                lambda_adapt = 0.5  # Adaptation sensitivity
                
                # Calculate improvement signal: max(0, ε - ΔR_e)/ε
                improvement_signal = np.maximum(0, epsilon - delta_rewards) / epsilon
                
                # Calculate confidence signal: (1 - H_e)
                confidence_signal = 1 - normalized_entropy
                
                # Calculate adaptive update factor: 1 + λ * (improvement_signal + confidence_signal)
                adaptive_factor = 1 + lambda_adapt * (improvement_signal + confidence_signal)
                
                # Plot components
                plt.plot(steps, improvement_signal, label=f'{exp_label} - Improvement Signal', linestyle='-')
                plt.plot(steps, confidence_signal, label=f'{exp_label} - Confidence Signal', linestyle='--')
                plt.plot(steps, adaptive_factor, label=f'{exp_label} - Adaptive Factor (η_e)', linestyle='-.')
        
        plt.title('Adaptive Quantization Update Factor Components (equation 4)')
        plt.xlabel('Training Step')
        plt.ylabel('Component Value')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adaptive_update_factor_components.png'), dpi=300)
        plt.close()
    
    # NEW: Plot the Relationship between Reward Improvement and Entropy
    if 'train/reward' in quant_metrics and 'train/entropy' in quant_metrics:
        plt.figure(figsize=(10, 8))
        
        for exp_label, values in quant_metrics['train/reward'].items():
            if exp_label in quant_metrics['train/entropy']:
                reward_steps, reward_values = zip(*values)
                entropy_steps, entropy_values = zip(*quant_metrics['train/entropy'][exp_label])
                
                # Ensure the steps match
                common_steps = set(reward_steps).intersection(entropy_steps)
                reward_dict = dict(values)
                entropy_dict = dict(quant_metrics['train/entropy'][exp_label])
                
                # Get data for common steps
                x_vals = []  # Entropy
                y_vals = []  # Reward
                for step in sorted(common_steps):
                    x_vals.append(entropy_dict[step])
                    y_vals.append(reward_dict[step])
                
                plt.scatter(x_vals, y_vals, label=exp_label, alpha=0.7)
                
                # Add a trend line
                if len(x_vals) > 1:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    plt.plot(x_vals, p(x_vals), linestyle='--', alpha=0.5)
        
        plt.title('Relationship Between Policy Entropy and Reward')
        plt.xlabel('Entropy (H)')
        plt.ylabel('Reward')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entropy_reward_relationship.png'), dpi=300)
        plt.close()
    
    # NEW: Plot the Quantization Error Bound (based on Theorem 4.1)
    # Plot theoretical bounds on gradient error based on K
    plt.figure(figsize=(10, 6))
    k_values = np.linspace(5, 50, 100)
    vmin, vmax = -10, 10  # From base config
    
    # Quantization error ε is proportional to bin width: (vmax - vmin) / K
    quantization_error = (vmax - vmin) / k_values
    
    # Calculate theoretical gradient error bound based on Theorem 4.1: L * (ε + sqrt(E[δ^2]))
    # Simplifying by assuming E[δ^2] is proportional to ε^2
    L = 2.0  # Assumed Lipschitz constant
    delta_factor = 0.5  # Assumed factor for STE approximation error
    
    gradient_error_bound = L * (quantization_error + np.sqrt(delta_factor * quantization_error**2))
    
    plt.plot(k_values, quantization_error, label='Quantization Error (ε)', linewidth=2)
    plt.plot(k_values, gradient_error_bound, label='Gradient Error Bound', linewidth=2)
    
    # Mark some example K values used in experiments
    for k in [5, 10, 20]:
        idx = np.argmin(np.abs(k_values - k))
        plt.scatter([k], [quantization_error[idx]], marker='o', s=100, color='red')
        plt.scatter([k], [gradient_error_bound[idx]], marker='o', s=100, color='blue')
        plt.axvline(x=k, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Theoretical Quantization Error Bound (Theorem 4.1)')
    plt.xlabel('Number of Quantization Bins (K)')
    plt.ylabel('Error Magnitude')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantization_error_bound.png'), dpi=300)
    plt.close()
    
    # NEW: Visualize the impact of K on the distribution of trajectories
    # Create a heatmap showing bin usage distribution
    if 'quantization/k' in quant_metrics:
        plt.figure(figsize=(12, 8))
        
        # For each experiment with adaptive K, generate synthetic bin usage distribution
        for exp_label, values in quant_metrics['quantization/k'].items():
            if 'adaptive' in exp_label.lower() and len(values) > 5:
                steps, k_values = zip(*values)
                
                # Create a synthetic visualization of bin distribution for different K values
                # Using log scale to better visualize the differences
                k_points = [5, 10, 20, 30]  # Example K values
                num_rows = len(k_points)
                
                # Only proceed if we have sufficient variation in K
                if max(k_values) - min(k_values) > 3:
                    # Create a grid of subplots
                    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 3*num_rows), sharex=True)
                    
                    for i, k in enumerate(k_points):
                        if k <= max(k_values):
                            # Create a synthetic bin distribution for this K
                            # Using a power law distribution as an example
                            bins = np.arange(k)
                            
                            # Higher K = more concentrated distribution (less uniform)
                            alpha = 0.5 + (k / 50)  # Control parameter
                            distribution = 1 / (bins + 1)**alpha
                            distribution = distribution / distribution.sum()  # Normalize
                            
                            # Plot the distribution
                            ax = axes[i] if num_rows > 1 else axes
                            ax.bar(bins, distribution, width=0.8)
                            ax.set_title(f'Bin Usage Distribution for K={k}')
                            ax.set_ylabel('Usage Probability')
                            ax.set_ylim(0, max(distribution) * 1.1)
                            
                            # Calculate and display entropy
                            entropy = -np.sum(distribution * np.log(distribution + 1e-10))
                            max_entropy = np.log(k)
                            normalized_entropy = entropy / max_entropy
                            ax.text(0.02, 0.85, f'Normalized Entropy: {normalized_entropy:.4f}', 
                                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
                    
                    plt.xlabel('Bin Index')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'bin_distribution_{exp_label}.png'), dpi=300)
                    plt.close()
        
        # Also create a visualization of adaptive quantization decision boundaries
        plt.figure(figsize=(10, 6))
        
        # Create a grid of values for delta_R and H
        delta_R = np.linspace(0, 0.05, 100)  # Reward improvement
        H = np.linspace(0, 1, 100)  # Normalized entropy
        
        # Create meshgrid
        Delta_R, H_mesh = np.meshgrid(delta_R, H)
        
        # Set adaptation parameters
        epsilon = 0.02  # Reward improvement threshold
        lambda_adapt = 0.5  # Adaptation sensitivity
        
        # Calculate the adaptive update factor for each point: η_e = 1 + λ * (max(0, ε - ΔR_e)/ε + (1 - H_e))
        improvement_signal = np.maximum(0, epsilon - Delta_R) / epsilon
        confidence_signal = 1 - H_mesh
        adaptive_factor = 1 + lambda_adapt * (improvement_signal + confidence_signal)
        
        # Create contour plot
        plt.figure(figsize=(10, 8))
        plt.contourf(Delta_R, H_mesh, adaptive_factor, 20, cmap='viridis')
        plt.colorbar(label='Adaptive Update Factor (η_e)')
        
        # Draw the decision boundary for η_e = 1 (no change in K)
        plt.contour(Delta_R, H_mesh, adaptive_factor, levels=[1], colors='red', linestyles='dashed', linewidths=2)
        
        # Add labels
        plt.xlabel('Reward Improvement (ΔR_e)')
        plt.ylabel('Normalized Entropy (H_e)')
        plt.title('Adaptive Quantization Decision Boundaries')
        
        # Add annotation for decision regions
        plt.text(0.01, 0.2, 'Decrease K\n(η_e < 1)', color='white', fontsize=12, 
                ha='left', va='center', bbox=dict(facecolor='black', alpha=0.5))
        plt.text(0.04, 0.8, 'Increase K\n(η_e > 1)', color='black', fontsize=12, 
                ha='left', va='center', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adaptive_quantization_decision_boundaries.png'), dpi=300)
        plt.close()

def plot_ste_visualization(output_dir: str):
    """
    Generate visualizations related to the Straight-Through Estimator (STE) mechanism.
    
    This function creates illustrations of:
    1. The STE mechanism showing hard/soft sample flow
    2. The gradient approximation through the non-differentiable argmax
    3. The impact of STE on training dynamics
    """
    # Create directory for STE visualizations if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'ste_visualizations'), exist_ok=True)
    
    # 1. Visualize the STE mechanism with hard and soft samples
    plt.figure(figsize=(12, 6))
    
    # Set up plot parameters
    k = 10  # Number of bins
    vmin, vmax = -10, 10  # Value range
    bins = np.linspace(vmin, vmax, k)
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(k-1)]
    bin_centers = np.append(bin_centers, bins[-1])
    
    # Create example logits and probabilities
    logits = np.array([-3, -2, -1, 0, 1, 2, 1, 0, -1, -2])  # Example logits
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    
    # Create a shared axis for visual clarity
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
    
    # Probability distribution
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(np.arange(k), probs, alpha=0.7)
    ax1.set_title('Forward Policy Probabilities P(a_t = q_k|s_t)')
    ax1.set_xlabel('Bin Index k')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(np.arange(k))
    ax1.set_xticklabels([f'{i}' for i in range(k)])
    
    # Hard sample visualization
    hard_sample_idx = np.argmax(probs)  # Get the most likely bin
    ax2 = plt.subplot(gs[0, 1])
    ax2.bar(0, bin_centers[hard_sample_idx], color='green', width=0.5)
    ax2.set_title('Hard Sample')
    ax2.set_ylabel('Value')
    ax2.set_xticks([])
    ax2.text(0, bin_centers[hard_sample_idx]/2, f'q_{hard_sample_idx}', 
             ha='center', va='center', fontweight='bold')
    ax2.set_ylim(vmin, vmax)
    
    # Arrow connecting first two plots
    plt.annotate('', xy=(0, 0.5), xytext=(1, 0.5), 
                 xycoords=ax1.get_position(), textcoords=ax2.get_position(),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.annotate('argmax', xy=(0.5, 0.55), 
                 xycoords='figure fraction', ha='center')
    
    # Soft sample visualization
    ax3 = plt.subplot(gs[1, 0])
    # Expected value calculation using soft sample
    soft_sample = np.sum(probs * bin_centers)
    ax3.bar(np.arange(k), probs * bin_centers, alpha=0.7)
    ax3.set_title('Weighted Bin Values (Probability * Bin Value)')
    ax3.set_xlabel('Bin Index k')
    ax3.set_ylabel('Weighted Value')
    ax3.set_xticks(np.arange(k))
    ax3.set_xticklabels([f'{i}' for i in range(k)])
    
    # Expected value visualization
    ax4 = plt.subplot(gs[1, 1])
    ax4.bar(0, soft_sample, color='blue', width=0.5)
    ax4.set_title('Soft Sample')
    ax4.set_ylabel('Value')
    ax4.set_xticks([])
    ax4.text(0, soft_sample/2, f'E[q]', 
             ha='center', va='center', fontweight='bold')
    ax4.set_ylim(vmin, vmax)
    
    # Arrow connecting second two plots
    plt.annotate('', xy=(0, 0.5), xytext=(1, 0.5), 
                 xycoords=ax3.get_position(), textcoords=ax4.get_position(),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.annotate('sum', xy=(0.5, 0.25), 
                 xycoords='figure fraction', ha='center')
    
    # Annotate STE explanation
    plt.figtext(0.5, 0.02, 
               "Straight-Through Estimator (STE): For forward pass, use hard sample (argmax).\n"
               "For backward pass, gradient flows through soft sample (expectation).",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.suptitle('Straight-Through Estimator Mechanism', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'ste_visualizations', 'ste_mechanism.png'), dpi=300)
    plt.close()
    
    # 2. Visualize the gradient flow through non-differentiable operations
    plt.figure(figsize=(10, 8))
    
    # Simple demonstration of STE
    x = np.linspace(-3, 3, 1000)
    
    # Define step function (non-differentiable)
    step_function = np.zeros_like(x)
    step_function[x > 0] = 1
    
    # Define sigmoid approximation (differentiable)
    temperature = 0.2  # Controls the steepness
    sigmoid = 1 / (1 + np.exp(-x/temperature))
    
    # Define the STE gradient (identity gradient for backward)
    ste_gradient = np.ones_like(x)  # Identity gradient everywhere
    
    # Create the plot
    plt.subplot(3, 1, 1)
    plt.plot(x, step_function, label='Hard Sample (Forward)', linewidth=2)
    plt.plot(x, sigmoid, label='Sigmoid Approximation', linestyle='--', linewidth=2)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Hard vs. Soft Sample')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(x, np.zeros_like(x), label='Step Function Gradient (Zero)', linewidth=2)
    plt.plot(x, sigmoid * (1 - sigmoid) / temperature, label='Sigmoid Gradient', linestyle='--', linewidth=2)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('True Gradients')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(x, ste_gradient, label='STE Gradient (Identity)', linewidth=2)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('STE Gradient Approximation')
    plt.xlabel('Input')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('STE: Allowing Gradient Flow Through Non-differentiable Operations', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'ste_visualizations', 'ste_gradient_flow.png'), dpi=300)
    plt.close()
    
    # 3. Visualize the theoretical impact of STE on learning dynamics
    plt.figure(figsize=(10, 6))
    
    # Create a toy learning scenario
    steps = np.arange(0, 100)
    
    # Without STE: No gradient flow, no learning
    no_ste_error = np.ones_like(steps) * 0.8
    
    # With STE but high gradient error
    high_error_ste = 0.8 * np.exp(-0.01 * steps) + 0.3
    
    # With STE and low gradient error
    low_error_ste = 0.8 * np.exp(-0.03 * steps) + 0.1
    
    # Theoretical bound from theorem 4.1
    theoretical_bound = 0.2 + 0.6 * np.exp(-0.02 * steps)
    
    plt.plot(steps, no_ste_error, label='Without STE (No Learning)', linewidth=2)
    plt.plot(steps, high_error_ste, label='With STE (High Error)', linewidth=2)
    plt.plot(steps, low_error_ste, label='With STE (Low Error)', linewidth=2)
    plt.plot(steps, theoretical_bound, label='Theoretical Bound', linestyle='--', linewidth=2)
    
    plt.title('Impact of STE on Learning Dynamics')
    plt.xlabel('Training Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ste_visualizations', 'ste_learning_impact.png'), dpi=300)
    plt.close()

def plot_tb_loss_visualization(output_dir: str):
    """
    Generate visualizations related to Trajectory Balance loss and exploration-exploitation trade-off.
    
    This function creates illustrations of:
    1. The Trajectory Balance loss concept and flow matching
    2. How entropy regularization affects exploration vs. exploitation
    3. Visualizing trajectories with different entropy values
    """
    # Create directory for TB loss visualizations if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'tb_loss_visualizations'), exist_ok=True)
    
    # 1. Visualize the Trajectory Balance loss concept
    plt.figure(figsize=(10, 6))
    
    # Create simple graph nodes for state space
    nodes = {
        's0': (0, 0.5),    # Initial state
        's1a': (0.33, 0.8),
        's1b': (0.33, 0.5),
        's1c': (0.33, 0.2),
        's2a': (0.66, 0.9),
        's2b': (0.66, 0.7),
        's2c': (0.66, 0.5),
        's2d': (0.66, 0.3),
        's2e': (0.66, 0.1),
        'sT1': (1, 0.9),   # Terminal state 1
        'sT2': (1, 0.7),   # Terminal state 2
        'sT3': (1, 0.5),   # Terminal state 3
        'sT4': (1, 0.3),   # Terminal state 4
        'sT5': (1, 0.1),   # Terminal state 5
    }
    
    # Draw nodes
    for name, pos in nodes.items():
        if name == 's0':
            color = 'green'
            size = 1000
            label = 'Initial state $s_0$'
        elif name.startswith('sT'):
            color = 'red'
            size = 800
            label = f'Terminal state $s_T$' if name == 'sT1' else None
        else:
            color = 'skyblue'
            size = 600
            label = None
        
        plt.scatter(pos[0], pos[1], s=size, c=color, alpha=0.7, edgecolors='black', zorder=10)
        if label:
            plt.text(pos[0], pos[1], label, fontsize=10, ha='center', va='center')
    
    # Draw edges with different weights
    # Forward edges
    draw_weighted_edge(plt, nodes['s0'], nodes['s1a'], weight=0.5, color='blue', label='Forward\nFlow\n$P_F(τ)$')
    draw_weighted_edge(plt, nodes['s0'], nodes['s1b'], weight=0.3, color='blue')
    draw_weighted_edge(plt, nodes['s0'], nodes['s1c'], weight=0.2, color='blue')
    
    draw_weighted_edge(plt, nodes['s1a'], nodes['s2a'], weight=0.3, color='blue')
    draw_weighted_edge(plt, nodes['s1a'], nodes['s2b'], weight=0.2, color='blue')
    draw_weighted_edge(plt, nodes['s1b'], nodes['s2b'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s1b'], nodes['s2c'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s1b'], nodes['s2d'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s1c'], nodes['s2d'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s1c'], nodes['s2e'], weight=0.1, color='blue')
    
    draw_weighted_edge(plt, nodes['s2a'], nodes['sT1'], weight=0.3, color='blue')
    draw_weighted_edge(plt, nodes['s2b'], nodes['sT2'], weight=0.3, color='blue')
    draw_weighted_edge(plt, nodes['s2c'], nodes['sT3'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s2d'], nodes['sT4'], weight=0.1, color='blue')
    draw_weighted_edge(plt, nodes['s2e'], nodes['sT5'], weight=0.2, color='blue')
    
    # Backward edges
    draw_weighted_edge(plt, nodes['sT1'], nodes['s2a'], weight=0.3, color='red', linestyle='--', label='Backward\nFlow\n$P_B(τ|x)$')
    draw_weighted_edge(plt, nodes['sT2'], nodes['s2b'], weight=0.3, color='red', linestyle='--')
    draw_weighted_edge(plt, nodes['sT3'], nodes['s2c'], weight=0.1, color='red', linestyle='--')
    draw_weighted_edge(plt, nodes['sT4'], nodes['s2d'], weight=0.1, color='red', linestyle='--')
    draw_weighted_edge(plt, nodes['sT5'], nodes['s2e'], weight=0.2, color='red', linestyle='--')
    
    # Add explanatory text
    title = "Trajectory Balance: Flow Consistency in GFlowNets"
    plt.text(0.5, 1.05, title, fontsize=14, ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.text(0.5, -0.05, 
           "Trajectory Balance Loss ensures consistency between forward and backward flows:\n"
           "$L_{TB}(τ) = (\\log Z + \\sum_{t=0}^{T'-1} \\log P_F(s_{t+1}|s_t) - \\sum_{t=1}^{T'} \\log P_B(s_{t-1}|s_t) - \\log R(τ))^2$",
           fontsize=11, ha='center', va='center', transform=plt.gca().transAxes)
    
    # Add reward explanation
    plt.text(1.05, 0.9, "$R(τ_1) = 0.9$", fontsize=10, ha='left', va='center')
    plt.text(1.05, 0.7, "$R(τ_2) = 0.7$", fontsize=10, ha='left', va='center')
    plt.text(1.05, 0.5, "$R(τ_3) = 0.5$", fontsize=10, ha='left', va='center')
    plt.text(1.05, 0.3, "$R(τ_4) = 0.3$", fontsize=10, ha='left', va='center')
    plt.text(1.05, 0.1, "$R(τ_5) = 0.1$", fontsize=10, ha='left', va='center')
    
    plt.text(1.15, 0.5, "Reward $R(τ)$", fontsize=12, ha='center', va='center', rotation=-90)
    
    plt.axis('off')
    plt.xlim(-0.1, 1.2)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tb_loss_visualizations', 'trajectory_balance_concept.png'), dpi=300)
    plt.close()
    
    # 2. Visualize the impact of entropy regularization on exploration-exploitation
    plt.figure(figsize=(12, 6))
    
    # Create action space with 10 bins
    k = 10
    bins = np.arange(k)
    
    # Create a grid of 4 plots for different entropy values
    lambda_values = [0.0, 0.001, 0.01, 0.1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    
    for i, lambda_val in enumerate(lambda_values):
        ax = axes[i]
        
        # Create a base distribution concentrated on 2 bins
        base_probs = np.zeros(k)
        base_probs[3] = 0.6
        base_probs[6] = 0.4
        
        # Apply entropy regularization effect
        if lambda_val == 0:
            # No regularization - keep base distribution
            probs = base_probs
        else:
            # Simulate entropy regularization by smoothing distribution
            # Higher lambda = more uniform distribution
            uniform_weight = lambda_val * 10
            uniform_probs = np.ones(k) / k
            
            # Mix base distribution with uniform distribution
            probs = (1 - uniform_weight) * base_probs + uniform_weight * uniform_probs
            probs = probs / probs.sum()  # Renormalize
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(k)
        normalized_entropy = entropy / max_entropy
        
        # Plot distribution
        ax.bar(bins, probs, alpha=0.7)
        ax.set_title(f'λ_entropy = {lambda_val}')
        ax.set_xlabel('Bin Index')
        if i == 0:
            ax.set_ylabel('Probability')
        ax.set_xticks(bins)
        
        # Add entropy annotation
        ax.text(0.5, 0.9, f'Entropy: {normalized_entropy:.3f}', 
               transform=ax.transAxes, ha='center',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add behavior annotation
        if lambda_val == 0:
            behavior = "Pure Exploitation"
        elif lambda_val == 0.001:
            behavior = "Mostly Exploitation"
        elif lambda_val == 0.01:
            behavior = "Balanced"
        else:
            behavior = "Heavy Exploration"
            
        ax.text(0.5, 0.8, behavior, 
               transform=ax.transAxes, ha='center', fontweight='bold',
               bbox=dict(facecolor='yellow', alpha=0.3))
    
    plt.suptitle('Impact of Entropy Regularization on Exploration-Exploitation', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'tb_loss_visualizations', 'entropy_regularization_impact.png'), dpi=300)
    plt.close()
    
    # 3. Visualize how different entropy values affect trajectory diversity
    plt.figure(figsize=(12, 10))
    
    # Set up for visualizing trajectories
    T = 8  # Prediction horizon
    lambda_values = [0.0, 0.001, 0.01, 0.1]
    
    # Create subplots for different entropy settings
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, lambda_val in enumerate(lambda_values):
        ax = axes[i]
        
        # Generate synthetic trajectories
        n_trajectories = 20
        
        # Base trajectory (time series trend)
        base = np.sin(np.linspace(0, 2*np.pi, T)) * 3
        
        # Different entropy values affect dispersion of trajectories
        if lambda_val == 0:
            # No entropy = almost identical trajectories
            noise_scale = 0.05
        elif lambda_val == 0.001:
            # Low entropy = slight variations
            noise_scale = 0.2
        elif lambda_val == 0.01:
            # Medium entropy = moderate variations
            noise_scale = 0.5
        else:
            # High entropy = wide variations
            noise_scale = 1.0
        
        # Generate trajectories
        trajectories = []
        for _ in range(n_trajectories):
            noise = np.random.normal(0, noise_scale, T)
            trajectory = base + noise
            trajectories.append(trajectory)
            
            # Plot individual trajectory with low opacity
            ax.plot(range(T), trajectory, alpha=0.3, color='blue')
        
        # Plot the mean trajectory
        mean_trajectory = np.mean(trajectories, axis=0)
        ax.plot(range(T), mean_trajectory, 'r-', linewidth=2, label='Mean Forecast')
        
        # Plot the base trajectory
        ax.plot(range(T), base, 'k--', linewidth=2, label='True Signal')
        
        # Calculate dispersion metrics
        variances = np.var(trajectories, axis=0)
        mean_var = np.mean(variances)
        
        # Configure plot
        ax.set_title(f'λ_entropy = {lambda_val} (Variance: {mean_var:.3f})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Add annotation about behavior
        if lambda_val == 0:
            behavior = "Low Diversity = Pure Exploitation"
        elif lambda_val == 0.001:
            behavior = "Minimal Diversity = Limited Exploration"
        elif lambda_val == 0.01:
            behavior = "Balanced Diversity = Good Exploration-Exploitation"
        else:
            behavior = "High Diversity = Heavy Exploration"
            
        ax.text(0.5, 0.05, behavior, 
               transform=ax.transAxes, ha='center', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.suptitle('Entropy Regularization Effect on Trajectory Diversity', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'tb_loss_visualizations', 'trajectory_diversity.png'), dpi=300)
    plt.close()

def draw_weighted_edge(plt, start, end, weight=1.0, color='black', linestyle='-', label=None):
    """Helper function to draw a weighted edge between nodes"""
    # Determine line width based on weight
    lw = weight * 3
    
    # Draw the line
    line = plt.plot([start[0], end[0]], [start[1], end[1]], 
                    color=color, linewidth=lw, linestyle=linestyle, zorder=5,
                    label=label if label else "")
    
    # Add weight label if significant
    if weight >= 0.2:
        # Position the label at the middle of the line
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Slight offset to avoid overlapping with the line
        offset = 0.02
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.arctan2(dy, dx)
        offset_x = -offset * np.sin(angle)
        offset_y = offset * np.cos(angle)
        
        plt.text(mid_x + offset_x, mid_y + offset_y, f"{weight:.1f}", 
                fontsize=8, ha='center', va='center', color=color,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
    
    return line

def plot_quantization_range_visualization(output_dir: str):
    """
    Generate visualizations related to the quantization range (vmin, vmax) and its impact on trajectory generation.
    
    This function creates illustrations of:
    1. The effect of different vmin/vmax values on bin resolution
    2. How quantization range affects the model's ability to capture extreme values
    3. The trade-off between range coverage and bin resolution
    """
    # Create directory for quantization range visualizations if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'quantization_range_visualizations'), exist_ok=True)
    
    # 1. Visualize how different vmin/vmax values affect bin resolution
    # Standard values in the codebase: vmin=-10.0, vmax=10.0
    vmin_vmax_configs = [
        (-5.0, 5.0, 'Narrow Range'),
        (-10.0, 10.0, 'Standard Range'),
        (-20.0, 20.0, 'Wide Range')
    ]
    
    k_values = [10, 20, 50]
    
    fig, axes = plt.subplots(len(vmin_vmax_configs), len(k_values), figsize=(15, 10))
    plt.suptitle('Quantization Resolution: Effect of Range and Bin Count', fontsize=16)
    
    for i, (vmin, vmax, range_label) in enumerate(vmin_vmax_configs):
        for j, k in enumerate(k_values):
            ax = axes[i, j]
            
            # Calculate bin width for this configuration
            bin_width = (vmax - vmin) / k
            
            # Generate synthetic values that span slightly beyond the range
            extended_min = vmin * 1.2
            extended_max = vmax * 1.2
            x = np.linspace(extended_min, extended_max, 1000)
            
            # Create a simple distribution with both in-range and out-of-range values
            mu = 0
            sigma = vmax / 2
            y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            
            # Plot the PDF
            ax.plot(x, y, 'k-', alpha=0.5)
            
            # Draw quantization bins
            bins = np.linspace(vmin, vmax, k+1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_heights = np.interp(bin_centers, x, y)
            
            # Calculate portion outside range (clipped)
            clipped_portion = np.sum(y[x < vmin]) + np.sum(y[x > vmax])
            clipped_percentage = clipped_portion / np.sum(y) * 100
            
            # Plot bins
            for b in range(k):
                left = bins[b]
                right = bins[b+1]
                center = bin_centers[b]
                height = bin_heights[b]
                
                # Plot bin rectangle
                ax.add_patch(plt.Rectangle((left, 0), bin_width, height, alpha=0.3, color='blue'))
                
                # Add tick marks for bin centers
                if k <= 20:  # Only show labels for smaller k to avoid clutter
                    ax.plot([center, center], [0, height], 'b--', alpha=0.5)
            
            # Fill clipped regions with red
            in_range_mask = (x >= vmin) & (x <= vmax)
            ax.fill_between(x[~in_range_mask], 0, y[~in_range_mask], color='red', alpha=0.3)
            
            # Set labels and title
            if i == len(vmin_vmax_configs) - 1:
                ax.set_xlabel('Value')
            if j == 0:
                ax.set_ylabel(f'{range_label}\n[{vmin}, {vmax}]')
            
            ax.set_title(f'k={k}, Bin Width={bin_width:.3f}')
            
            # Add annotation about clipping
            ax.text(0.5, 0.05, f'Clipped: {clipped_percentage:.1f}%', 
                  transform=ax.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # Configure axes
            ax.set_xlim(extended_min, extended_max)
            ax.set_ylim(0, max(y) * 1.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'quantization_range_visualizations', 'quantization_resolution.png'), dpi=300)
    plt.close()
    
    # 2. Visualize the error introduced by vmin/vmax clipping on different distributions
    plt.figure(figsize=(15, 10))
    
    # Create distributions with different characteristics
    distributions = [
        ('Normal', 0, 5, lambda x: np.exp(-0.5 * ((x - 0) / 5) ** 2) / (5 * np.sqrt(2 * np.pi))),
        ('Right-skewed', 2, 3, lambda x: np.exp(-0.5 * ((np.log(x+0.1) - 2) / 3) ** 2) / (x * 3 * np.sqrt(2 * np.pi))),
        ('Heavy-tailed', 0, 1, lambda x: 1 / (np.pi * (1 + x**2)))
    ]
    
    # Test different vmin/vmax configurations
    configs = [
        (-5, 5), 
        (-10, 10), 
        (-20, 20)
    ]
    
    fig, axes = plt.subplots(len(distributions), len(configs), figsize=(15, 10))
    plt.suptitle('Effect of Quantization Range on Different Data Distributions', fontsize=16)
    
    x = np.linspace(-30, 30, 1000)
    
    for i, (dist_name, _, _, dist_func) in enumerate(distributions):
        y = dist_func(x)
        
        for j, (vmin, vmax) in enumerate(configs):
            ax = axes[i, j]
            
            # Plot the original distribution
            ax.plot(x, y, 'k-', alpha=0.7, label='Original')
            
            # Calculate and plot the clipped distribution
            y_clipped = y.copy()
            clip_mask = (x < vmin) | (x > vmax)
            y_clipped[clip_mask] = 0
            
            # Renormalize the clipped distribution
            if np.sum(y_clipped) > 0:
                y_clipped = y_clipped * np.sum(y) / np.sum(y_clipped)
            
            ax.plot(x, y_clipped, 'r-', alpha=0.7, label='After Clipping')
            
            # Fill the clipped areas
            ax.fill_between(x[clip_mask], 0, y[clip_mask], color='red', alpha=0.3)
            
            # Calculate error metrics
            clipped_mass = np.sum(y[clip_mask]) / np.sum(y) * 100
            
            # Set plot labels
            if i == len(distributions) - 1:
                ax.set_xlabel('Value')
            if j == 0:
                ax.set_ylabel(dist_name)
            
            ax.set_title(f'Range [{vmin}, {vmax}]')
            
            # Add annotation
            ax.text(0.5, 0.9, f'Clipped: {clipped_mass:.1f}%', 
                  transform=ax.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # Add vertical lines at vmin and vmax
            ax.axvline(x=vmin, color='blue', linestyle='--', alpha=0.7)
            ax.axvline(x=vmax, color='blue', linestyle='--', alpha=0.7)
            
            # Configure axes - zoom to reasonable area
            ax.set_xlim(-25, 25)
            
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'quantization_range_visualizations', 'distribution_clipping.png'), dpi=300)
    plt.close()
    
    # 3. Visualize the trade-off between range coverage and bin resolution for different k values
    plt.figure(figsize=(12, 8))
    
    # Parameters to explore
    k_values = [10, 20, 50, 100]
    fixed_range = (-10, 10)  # Standard range
    
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Create synthetic time series data with some outliers
    np.random.seed(42)
    n_steps = 100
    t = np.arange(n_steps)
    
    # Base seasonal pattern
    seasonal = 5 * np.sin(2 * np.pi * t / 24)
    
    # Add trend
    trend = 0.05 * t
    
    # Add some outliers
    base_series = seasonal + trend
    outliers_idx = np.random.choice(n_steps, size=5, replace=False)
    outlier_values = np.random.choice([-15, -12, 12, 15, 18], size=5, replace=True)
    
    full_series = base_series.copy()
    full_series[outliers_idx] = outlier_values
    
    for i, k in enumerate(k_values):
        ax = axes[i]
        
        # Calculate bin width
        vmin, vmax = fixed_range
        bin_width = (vmax - vmin) / k
        
        # Quantize and dequantize the series
        quantized_series = np.clip(np.floor((full_series - vmin) / bin_width), 0, k-1)
        dequantized_series = vmin + (quantized_series + 0.5) * bin_width
        
        # Calculate errors
        overall_error = np.abs(full_series - dequantized_series)
        clipped_mask = (full_series < vmin) | (full_series > vmax)
        normal_mask = ~clipped_mask
        
        clipping_error = overall_error[clipped_mask].mean() if clipped_mask.any() else 0
        quantization_error = overall_error[normal_mask].mean()
        
        # Plot original and reconstructed series
        ax.plot(t, full_series, 'k-', alpha=0.7, label='Original')
        ax.plot(t, dequantized_series, 'b-', alpha=0.7, label='Reconstructed')
        
        # Highlight outliers and clipped areas
        ax.scatter(t[clipped_mask], full_series[clipped_mask], color='red', s=50, label='Clipped Values')
        ax.scatter(t[clipped_mask], dequantized_series[clipped_mask], color='orange', s=30)
        
        # Add horizontal lines at vmin and vmax
        ax.axhline(y=vmin, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=vmax, color='red', linestyle='--', alpha=0.7)
        
        # Calculate error statistics
        clipped_count = np.sum(clipped_mask)
        max_error = np.max(overall_error)
        
        # Set plot labels
        ax.set_title(f'k={k}, Bin Width={bin_width:.3f}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        
        # Add error annotation
        error_text = (
            f'Quantization Error: {quantization_error:.3f}\n'
            f'Clipping Error: {clipping_error:.3f}\n'
            f'Points Clipped: {clipped_count}/{n_steps}'
        )
        ax.text(0.03, 0.97, error_text, transform=ax.transAxes, va='top', 
               bbox=dict(facecolor='white', alpha=0.7), fontsize=9)
        
        if i == 0:
            ax.legend()
    
    plt.suptitle('Trade-off: Range Coverage vs. Bin Resolution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'quantization_range_visualizations', 'resolution_tradeoff.png'), dpi=300)
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
        # Generate STE visualizations
        plot_ste_visualization(args.output_dir)
        # Generate TB loss visualizations
        plot_tb_loss_visualization(args.output_dir)
        # Generate quantization range visualizations
        plot_quantization_range_visualization(args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 