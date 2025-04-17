#!/usr/bin/env python
"""
Enhanced script to visualize results from ablation studies for quantization methods.

This script loads metrics and training curves from different experiment configurations 
and generates publication-quality comparative visualizations, focusing on readability,
design consistency, and statistical insight.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy import stats

# Set plot styles for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Define a consistent color palette for experiments
COLORS = {
    'adaptive_k5': '#1f77b4',   # blue
    'adaptive_k10': '#ff7f0e',  # orange
    'adaptive_k20': '#2ca02c',  # green
    'fixed_k10': '#d62728',     # red
    'fixed_k20': '#9467bd',     # purple
    'learned_policy': '#8c564b', # brown
    'default': '#e377c2'        # pink (for any others)
}

def get_color(exp_name: str) -> str:
    """Get a consistent color for an experiment name."""
    return COLORS.get(exp_name, COLORS['default'])

def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics from {metrics_file}: {e}")
        return {}

def load_training_curves(training_curves_file: str) -> Dict:
    """Load training curves from a JSON file."""
    try:
        with open(training_curves_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training curves from {training_curves_file}: {e}")
        return {}

def plot_overall_metrics_comparison(metrics_data: Dict, output_dir: str, fig_format: str = 'pdf') -> None:
    """
    Generate a single figure with bar plots for all overall metrics.
    
    Args:
        metrics_data: Dictionary mapping experiment names to their metrics
        output_dir: Directory to save the plots
        fig_format: Format to save the figures (pdf, png, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract overall metrics
    overall_metrics = {}
    all_metrics = set()
    for exp_name, metrics in metrics_data.items():
        if 'overall' in metrics:
            for metric_name, value in metrics['overall'].items():
                all_metrics.add(metric_name)
                if metric_name not in overall_metrics:
                    overall_metrics[metric_name] = []
                overall_metrics[metric_name].append({
                    'Experiment': exp_name,
                    'Value': value
                })
    
    # Skip if no metrics found
    if not overall_metrics:
        print("No overall metrics found.")
        return
    
    # Create a figure with subplots for each metric
    metric_list = sorted(all_metrics)
    n_metrics = len(metric_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    # Handle case with only one metric
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(metric_list):
        if metric_name in overall_metrics:
            df = pd.DataFrame(overall_metrics[metric_name])
            # Sort by value for better visualization
            df = df.sort_values('Value')
            
            # Plot bars with custom colors
            bars = axes[i].bar(
                range(len(df)), 
                df['Value'], 
                color=[get_color(exp) for exp in df['Experiment']]
            )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01 * max(df['Value']),
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=10
                )
            
            # Set labels and title
            axes[i].set_title(f'{metric_name.upper()}')
            axes[i].set_xticks(range(len(df)))
            axes[i].set_xticklabels(df['Experiment'], rotation=45, ha='right')
            axes[i].set_ylabel(metric_name.upper())
            
            # Add a grid for readability
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Create custom legend outside the plots
    handles = [mpatches.Patch(color=get_color(exp), label=exp) 
              for exp in sorted(set([item['Experiment'] for items in overall_metrics.values() 
                                    for item in items]))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(handles)), frameon=True)
    
    # Add more space at the bottom for the legend
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'overall_metrics.{fig_format}'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create individual plots for each metric
    for metric_name, data in overall_metrics.items():
        df = pd.DataFrame(data)
        df = df.sort_values('Value')
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            range(len(df)), 
            df['Value'], 
            color=[get_color(exp) for exp in df['Experiment']]
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01 * max(df['Value']),
                f'{height:.3f}',
                ha='center', 
                va='bottom', 
                fontsize=10
            )
        
        plt.title(f'{metric_name.upper()} Comparison Across Experiments')
        plt.xticks(range(len(df)), df['Experiment'], rotation=45, ha='right')
        plt.ylabel(metric_name.upper())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'overall_{metric_name}.{fig_format}'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_per_horizon_metrics(metrics_data: Dict, output_dir: str, fig_format: str = 'pdf') -> None:
    """
    Plot per-horizon metrics across experiments.
    
    Args:
        metrics_data: Dictionary mapping experiment names to their metrics
        output_dir: Directory to save the plots
        fig_format: Format to save the figures (pdf, png, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which metrics are available
    available_metrics = set()
    for exp_data in metrics_data.values():
        if 'per_horizon' in exp_data:
            available_metrics.update(exp_data['per_horizon'].keys())
    
    if not available_metrics:
        print("No per-horizon metrics found.")
        return
    
    # Create a single figure with all metrics
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 5))
    
    # Handle case with only one metric
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(sorted(available_metrics)):
        # Collect data for this metric
        horizon_data = []
        for exp_name, metrics in metrics_data.items():
            if 'per_horizon' in metrics and metric_name in metrics['per_horizon']:
                for h, value in enumerate(metrics['per_horizon'][metric_name]):
                    horizon_data.append({
                        'Experiment': exp_name,
                        'Horizon': h + 1,
                        'Value': value
                    })
        
        if horizon_data:
            df = pd.DataFrame(horizon_data)
            
            # Plot each experiment with consistent colors
            for exp_name in df['Experiment'].unique():
                exp_data = df[df['Experiment'] == exp_name]
                axes[i].plot(
                    exp_data['Horizon'], 
                    exp_data['Value'],
                    marker='o',
                    color=get_color(exp_name),
                    label=exp_name,
                    linewidth=2
                )
            
            axes[i].set_title(f'{metric_name.upper()} by Forecast Horizon')
            axes[i].set_xlabel('Forecast Horizon')
            axes[i].set_ylabel(metric_name.upper())
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            # Add light shading for confidence intervals
            for exp_name in df['Experiment'].unique():
                exp_data = df[df['Experiment'] == exp_name]
                x = exp_data['Horizon'].values
                y = exp_data['Value'].values
                
                # Simple confidence interval based on linear regression
                if len(x) > 2:  # Only if we have enough points
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    y_pred = intercept + slope * x
                    
                    # Calculate standard error of the prediction
                    std = np.sqrt(np.sum((y - y_pred)**2) / (len(y) - 2))
                    
                    # Plot confidence interval
                    axes[i].fill_between(
                        x,
                        y_pred - std,
                        y_pred + std,
                        alpha=0.2,
                        color=get_color(exp_name)
                    )
    
    # Create legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(labels)), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the combined figure
    plt.savefig(os.path.join(output_dir, f'per_horizon_metrics.{fig_format}'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create individual plots for each metric
    for metric_name in available_metrics:
        horizon_data = []
        for exp_name, metrics in metrics_data.items():
            if 'per_horizon' in metrics and metric_name in metrics['per_horizon']:
                for h, value in enumerate(metrics['per_horizon'][metric_name]):
                    horizon_data.append({
                        'Experiment': exp_name,
                        'Horizon': h + 1,
                        'Value': value
                    })
        
        if horizon_data:
            df = pd.DataFrame(horizon_data)
            
            plt.figure(figsize=(10, 6))
            
            # Plot each experiment
            for exp_name in df['Experiment'].unique():
                exp_data = df[df['Experiment'] == exp_name]
                plt.plot(
                    exp_data['Horizon'], 
                    exp_data['Value'],
                    marker='o',
                    color=get_color(exp_name),
                    label=exp_name,
                    linewidth=2
                )
                
                # Add confidence intervals
                if len(exp_data) > 2:
                    x = exp_data['Horizon'].values
                    y = exp_data['Value'].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    y_pred = intercept + slope * x
                    std = np.sqrt(np.sum((y - y_pred)**2) / (len(y) - 2))
                    plt.fill_between(
                        x,
                        y_pred - std,
                        y_pred + std,
                        alpha=0.2,
                        color=get_color(exp_name)
                    )
            
            plt.title(f'{metric_name.upper()} by Forecast Horizon')
            plt.xlabel('Forecast Horizon')
            plt.ylabel(metric_name.upper())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Experiment')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f'per_horizon_{metric_name}.{fig_format}'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def plot_training_curves(curves_data: Dict, output_dir: str, fig_format: str = 'pdf') -> None:
    """
    Create enhanced plots for training curves.
    
    Args:
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save the plots
        fig_format: Format to save the figures (pdf, png, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all available metrics
    all_metrics = set()
    for curves in curves_data.values():
        all_metrics.update([k for k in curves.keys() if k != 'epochs'])
    
    # Filter out metrics with missing data
    valid_metrics = []
    for metric in all_metrics:
        if any(metric in curves for curves in curves_data.values()):
            valid_metrics.append(metric)
    
    if not valid_metrics:
        print("No valid training metrics found.")
        return
    
    # Create a grid of plots for all metrics
    n_cols = min(2, len(valid_metrics))
    n_rows = (len(valid_metrics) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(12, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, metric in enumerate(sorted(valid_metrics)):
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        for exp_name, curves in curves_data.items():
            if metric in curves and 'epochs' in curves:
                # Plot with consistent color and style
                ax.plot(
                    curves['epochs'], 
                    curves[metric], 
                    label=exp_name,
                    color=get_color(exp_name),
                    marker='o', 
                    markersize=4,
                    linestyle='-',
                    linewidth=2,
                    alpha=0.8
                )
                
                # Add smoothed trend line
                if len(curves['epochs']) > 5:
                    x = np.array(curves['epochs'])
                    y = np.array(curves[metric])
                    
                    # Simple moving average for smoothing
                    window = min(5, len(y) // 4)
                    if window > 1:
                        weights = np.ones(window) / window
                        y_smooth = np.convolve(y, weights, mode='valid')
                        x_smooth = x[window-1:]
                        
                        ax.plot(
                            x_smooth, 
                            y_smooth, 
                            color=get_color(exp_name),
                            linestyle='--',
                            linewidth=1.5,
                            alpha=0.6
                        )
        
        # Set title and labels
        metric_label = metric.capitalize()
        if metric == 'k':
            metric_label = 'Quantization Level (k)'
            
        ax.set_title(f'Training {metric_label}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Y-axis formatting for better readability
        if metric == 'loss':
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        
        # Add legend only if there's more than one experiment
        if len(curves_data) > 1:
            ax.legend(title='Experiment')
    
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig(os.path.join(output_dir, f'training_curves.{fig_format}'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create individual plots for each metric
    for metric in valid_metrics:
        plt.figure(figsize=(10, 6))
        
        for exp_name, curves in curves_data.items():
            if metric in curves and 'epochs' in curves:
                plt.plot(
                    curves['epochs'], 
                    curves[metric], 
                    label=exp_name,
                    color=get_color(exp_name),
                    marker='o', 
                    markersize=4,
                    alpha=0.8
                )
                
                # Add smoothed trend line
                if len(curves['epochs']) > 5:
                    x = np.array(curves['epochs'])
                    y = np.array(curves[metric])
                    
                    # Simple moving average for smoothing
                    window = min(5, len(y) // 4)
                    if window > 1:
                        weights = np.ones(window) / window
                        y_smooth = np.convolve(y, weights, mode='valid')
                        x_smooth = x[window-1:]
                        
                        plt.plot(
                            x_smooth, 
                            y_smooth, 
                            color=get_color(exp_name),
                            linestyle='--',
                            linewidth=1.5,
                            alpha=0.6
                        )
        
        # Format and label
        metric_label = metric.capitalize()
        if metric == 'k':
            metric_label = 'Quantization Level (k)'
            
        plt.title(f'Training {metric_label} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if metric == 'loss':
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
        
        plt.legend(title='Experiment')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'training_{metric}.{fig_format}'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_correlation_heatmap(metrics_data: Dict, curves_data: Dict, output_dir: str, fig_format: str = 'pdf') -> None:
    """
    Create a correlation heatmap to show relationships between training dynamics and final metrics.
    
    Args:
        metrics_data: Dictionary mapping experiment names to their metrics
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save the plots
        fig_format: Format to save the figures (pdf, png, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data that can be correlated
    experiment_features = {}
    
    for exp_name in set(metrics_data.keys()) & set(curves_data.keys()):
        features = {}
        
        # Extract final metrics
        if 'overall' in metrics_data[exp_name]:
            for metric, value in metrics_data[exp_name]['overall'].items():
                features[f'final_{metric}'] = value
        
        # Extract training curve statistics
        if 'epochs' in curves_data[exp_name]:
            # Number of epochs
            features['num_epochs'] = len(curves_data[exp_name]['epochs'])
            
            # Training curve statistics
            for metric in ['loss', 'reward', 'entropy', 'k']:
                if metric in curves_data[exp_name]:
                    values = curves_data[exp_name][metric]
                    features[f'mean_{metric}'] = np.mean(values)
                    features[f'max_{metric}'] = np.max(values)
                    features[f'min_{metric}'] = np.min(values)
                    features[f'final_{metric}'] = values[-1]
                    
                    # Compute trend (slope of linear regression)
                    if len(values) > 2:
                        x = np.array(curves_data[exp_name]['epochs'])
                        y = np.array(values)
                        slope, _, _, _, _ = stats.linregress(x, y)
                        features[f'{metric}_trend'] = slope
        
        if features:
            experiment_features[exp_name] = features
    
    if not experiment_features:
        print("Not enough data for correlation analysis.")
        return
    
    # Convert to dataframe
    df = pd.DataFrame(experiment_features).T
    
    # Filter out columns with all NaN or single unique value
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.nunique() > 1]
    
    if df.shape[1] < 2:
        print("Not enough features for correlation analysis.")
        return
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask,
        cmap="RdBu_r",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f"
    )
    
    plt.title('Correlation between Training Dynamics and Final Metrics')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'correlation_heatmap.{fig_format}'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced visualization for ablation studies')
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_ablation_plots',
                        help='Directory to save the plots')
    parser.add_argument('--experiments', type=str, nargs='+', 
                        default=None,
                        help='Experiment folders to include in comparison (default: all folders in results_dir)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg', 'jpg'],
                        help='Output format for plots')
    
    # Add wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    parser.add_argument('--wandb_name', type=str, default='enhanced_visualization',
                        help='W&B run name')
    parser.add_argument('--offline', action='store_true',
                        help='Run W&B in offline mode')
    
    args = parser.parse_args()
    
    # Initialize wandb if specified
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            # Set offline mode if requested
            if hasattr(args, 'offline') and args.offline:
                os.environ["WANDB_MODE"] = "offline"
                print("Running W&B in offline mode")
                
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config={
                    "results_dir": args.results_dir,
                    "output_dir": args.output_dir,
                    "format": args.format
                }
            )
            use_wandb = True
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            use_wandb = False
    else:
        use_wandb = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of experiments
    if args.experiments is None:
        try:
            args.experiments = [d for d in os.listdir(args.results_dir) 
                              if os.path.isdir(os.path.join(args.results_dir, d))]
        except Exception as e:
            print(f"Error listing directory {args.results_dir}: {e}")
            return
    
    if not args.experiments:
        print(f"No experiments found in {args.results_dir}")
        return
        
    print(f"Found experiments: {args.experiments}")
    
    # Load metrics data
    metrics_data = {}
    curves_data = {}
    
    print(f"Loading data from experiments...")
    for exp_name in args.experiments:
        exp_dir = os.path.join(args.results_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f"Warning: Experiment directory {exp_dir} not found, skipping.")
            continue
            
        metrics_file = os.path.join(exp_dir, 'evaluation', 'metrics.json')
        if os.path.exists(metrics_file):
            print(f"Loading metrics from {metrics_file}")
            metrics_data[exp_name] = load_metrics(metrics_file)
        else:
            print(f"Warning: Metrics file not found for {exp_name}")
            
        curves_file = os.path.join(exp_dir, 'logs', 'training_curves.json')
        if os.path.exists(curves_file):
            print(f"Loading training curves from {curves_file}")
            curves_data[exp_name] = load_training_curves(curves_file)
        else:
            print(f"Warning: Training curves file not found for {exp_name}")
    
    # Generate plots
    if metrics_data:
        print(f"Generating overall metrics comparison plots...")
        plot_overall_metrics_comparison(metrics_data, args.output_dir, args.format)
        
        print(f"Generating per-horizon metrics plots...")
        plot_per_horizon_metrics(metrics_data, args.output_dir, args.format)
    else:
        print("No metrics data found.")
        
    if curves_data:
        print(f"Generating training curves plots...")
        plot_training_curves(curves_data, args.output_dir, args.format)
    else:
        print("No training curves data found.")
        
    if metrics_data and curves_data:
        print(f"Generating correlation heatmap...")
        plot_correlation_heatmap(metrics_data, curves_data, args.output_dir, args.format)
    
    print(f"Enhanced plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 