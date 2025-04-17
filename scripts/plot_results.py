#!/usr/bin/env python
"""
Visualization script for comparing results from different ablation studies.
Uses wandb for visualization instead of TensorBoard.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, List, Any, Optional

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_metrics(experiment_dir: str) -> Dict[str, Any]:
    """
    Load metrics from experiment results.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing metrics
    """
    metrics_path = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics file not found at {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_wandb_logs(experiment_dir: str) -> Optional[pd.DataFrame]:
    """
    Load wandb logs from experiment results.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        DataFrame containing wandb logs
    """
    # Check for wandb logs (typically in a CSV export or similar)
    wandb_export_path = os.path.join(experiment_dir, 'wandb_export.csv')
    if os.path.exists(wandb_export_path):
        return pd.read_csv(wandb_export_path)
    
    # If no exported file, check for wandb run ID
    wandb_id_path = os.path.join(experiment_dir, 'wandb_id.txt')
    if os.path.exists(wandb_id_path):
        with open(wandb_id_path, 'r') as f:
            run_id = f.read().strip()
            print(f"Found wandb run ID: {run_id}, but no exported data. Please export wandb data manually.")
    
    print(f"Warning: No wandb logs found at {experiment_dir}")
    return None


def plot_metrics_comparison(experiments: Dict[str, Dict[str, Any]], output_dir: str, 
                           upload_to_wandb: bool = False, wandb_run=None):
    """
    Plot metrics comparison between experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to metrics
        output_dir: Directory to save plots
        upload_to_wandb: Whether to upload plots to wandb
        wandb_run: wandb run object if uploading
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract overall metrics for each experiment
    metric_names = ['wql_mean', 'crps_mean', 'mase_mean']
    metrics_data = []
    
    for exp_name, metrics in experiments.items():
        if metrics is None:
            continue
        
        exp_metrics = {
            'experiment': exp_name
        }
        
        for metric in metric_names:
            if metric in metrics['overall']:
                exp_metrics[metric] = metrics['overall'][metric]
        
        metrics_data.append(exp_metrics)
    
    if not metrics_data:
        print("No metrics data available for plotting")
        return
    
    # Create dataframe
    df = pd.DataFrame(metrics_data)
    
    # Melt the dataframe for easier plotting
    df_melted = pd.melt(df, id_vars=['experiment'], var_name='metric', value_name='value')
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 5))
    if len(metric_names) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        # Skip if metric not available
        if metric not in df.columns:
            continue
        
        # Extract data for this metric
        metric_df = df_melted[df_melted['metric'] == metric]
        
        # Plot
        sns.barplot(x='experiment', y='value', data=metric_df, ax=axes[i])
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel('')
        
        # Add values on top of bars
        for j, v in enumerate(metric_df['value']):
            axes[i].text(j, v + 0.01, f"{v:.4f}", ha='center')
        
        # Set y-axis to start from 0
        axes[i].set_ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'metrics_comparison.png')
    fig.savefig(fig_path, bbox_inches='tight')
    
    # Upload to wandb if requested
    if upload_to_wandb and wandb_run is not None:
        wandb_run.log({"metrics_comparison": wandb.Image(fig_path)})
        
        # Also log the metrics as a table
        wandb_run.log({"metrics_table": wandb.Table(dataframe=df)})
        
        # Log individual metrics for easier comparison
        for exp_name, metrics in experiments.items():
            if metrics is None:
                continue
            
            # Log metrics with experiment name prefix
            for metric in metric_names:
                if metric in metrics['overall']:
                    metric_key = f"{exp_name}/{metric}"
                    wandb_run.summary[metric_key] = metrics['overall'][metric]
    
    plt.close(fig)


def plot_training_curves(experiments: Dict[str, pd.DataFrame], output_dir: str,
                        upload_to_wandb: bool = False, wandb_run=None):
    """
    Plot training curves comparison between experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to wandb logs
        output_dir: Directory to save plots
        upload_to_wandb: Whether to upload plots to wandb
        wandb_run: wandb run object if uploading
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        ('train/loss', 'Training Loss'),
        ('val/loss', 'Validation Loss'),
        ('train/log_reward', 'Training Reward'),
        ('quantization/k', 'Number of Bins (K)'),
    ]
    
    # Check if any logs are available
    if not any(log is not None for log in experiments.values()):
        print("No training logs available for plotting")
        return
    
    # Create a figure for each metric
    for metric_name, metric_title in metrics:
        # Check if metric exists in any experiment
        exists = False
        for exp_name, df in experiments.items():
            if df is not None and metric_name in df.columns:
                exists = True
                break
        
        if not exists:
            print(f"Metric {metric_name} not found in any experiment")
            continue
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot each experiment
        for exp_name, df in experiments.items():
            if df is None or metric_name not in df.columns:
                continue
            
            # Plot
            if 'step' in df.columns:
                plt.plot(df['step'], df[metric_name], label=exp_name)
            elif 'epoch' in df.columns:
                plt.plot(df['epoch'], df[metric_name], label=exp_name)
            else:
                plt.plot(df.index, df[metric_name], label=exp_name)
        
        # Add labels and title
        plt.xlabel('Step/Epoch')
        plt.ylabel('Value')
        plt.title(metric_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{metric_name.replace('/', '_')}.png")
        plt.savefig(fig_path, bbox_inches='tight')
        
        # Upload to wandb if requested
        if upload_to_wandb and wandb_run is not None:
            wandb_run.log({f"curve_{metric_name.replace('/', '_')}": wandb.Image(fig_path)})
        
        plt.close()


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot results from ablation studies')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./results/plots', help='Directory to save plots')
    parser.add_argument('--experiments', type=str, nargs='+', default=None, help='Experiments to include (default: all)')
    parser.add_argument('--use_wandb', action='store_true', help='Upload plots to wandb')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen', help='W&B entity name')
    parser.add_argument('--wandb_name', type=str, default='results_comparison', help='W&B run name')
    args = parser.parse_args()
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            job_type="visualization",
            config=vars(args)
        )
    
    # Get experiment directories
    if args.experiments is None:
        # Use all directories in results_dir
        args.experiments = [d for d in os.listdir(args.results_dir) 
                           if os.path.isdir(os.path.join(args.results_dir, d))]
    
    print(f"Including experiments: {args.experiments}")
    
    # Load metrics and wandb logs for each experiment
    metrics = {}
    logs = {}
    
    for exp_name in args.experiments:
        exp_dir = os.path.join(args.results_dir, exp_name)
        
        # Load metrics
        metrics[exp_name] = load_metrics(exp_dir)
        
        # Load wandb logs
        logs[exp_name] = load_wandb_logs(exp_dir)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics, args.output_dir, args.use_wandb, wandb_run)
    
    # Plot training curves
    plot_training_curves(logs, args.output_dir, args.use_wandb, wandb_run)
    
    print(f"Plots generated and saved to {args.output_dir}")
    
    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main() 