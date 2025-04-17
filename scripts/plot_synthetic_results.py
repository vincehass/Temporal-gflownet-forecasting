#!/usr/bin/env python
"""
Script to visualize synthetic results from ablation studies.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from matplotlib.gridspec import GridSpec

def load_metrics(results_dir: str, experiment_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load metrics from all experiments in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_filter: List of experiment names to include, if None, include all
        
    Returns:
        Dictionary of metrics by experiment name
    """
    metrics_by_exp = {}
    
    # List all directories in the results directory
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    if experiment_filter:
        exp_dirs = [d for d in exp_dirs if d in experiment_filter]
    
    print(f"Including experiments: {exp_dirs}")
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(results_dir, exp_name)
        metrics_path = os.path.join(exp_dir, "evaluation", "metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metrics_by_exp[exp_name] = metrics
                
                # Add experiment name for reference
                if "metadata" not in metrics:
                    metrics["metadata"] = {}
                metrics["metadata"]["experiment"] = exp_name
        else:
            print(f"Warning: No metrics found at {metrics_path}")
    
    return metrics_by_exp

def load_training_curves(results_dir: str, experiment_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load training curves from all experiments in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_filter: List of experiment names to include, if None, include all
        
    Returns:
        Dictionary of training curves by experiment name
    """
    curves_by_exp = {}
    
    # List all directories in the results directory
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    if experiment_filter:
        exp_dirs = [d for d in exp_dirs if d in experiment_filter]
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(results_dir, exp_name)
        curves_path = os.path.join(exp_dir, "logs", "training_curves.json")
        
        if os.path.exists(curves_path):
            with open(curves_path, 'r') as f:
                curves = json.load(f)
                curves_by_exp[exp_name] = curves
        else:
            print(f"Warning: No training curves found at {curves_path}")
    
    return curves_by_exp

def plot_metrics_comparison(metrics_by_exp: Dict[str, Any], output_dir: str) -> None:
    """
    Plot comparison of metrics across different experiments.
    
    Args:
        metrics_by_exp: Dictionary of metrics by experiment name
        output_dir: Directory to save plots
    """
    if not metrics_by_exp:
        print("No metrics available for plotting")
        return
    
    metrics = ['wql', 'crps', 'mase']
    
    # Prepare data for plotting
    data = []
    for exp_name, metrics_data in metrics_by_exp.items():
        for metric in metrics:
            if metric in metrics_data['overall']:
                data.append({
                    'experiment': exp_name,
                    'metric': metric.upper(),
                    'value': metrics_data['overall'][metric]
                })
    
    if not data:
        print("No overall metrics data found")
        return
    
    df = pd.DataFrame(data)
    
    # Create a figure for overall metrics comparison
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(x='experiment', y='value', hue='metric', data=df)
    plt.title('Comparison of Metrics Across Experiments')
    plt.ylabel('Value')
    plt.xlabel('Experiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    # Create individual plots for each metric
    for metric in metrics:
        metric_data = df[df['metric'] == metric.upper()]
        if not metric_data.empty:
            plt.figure(figsize=(8, 5))
            chart = sns.barplot(x='experiment', y='value', data=metric_data)
            plt.title(f'Comparison of {metric.upper()} Across Experiments')
            plt.ylabel(f'{metric.upper()} Value')
            plt.xlabel('Experiment')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'overall_{metric}.png'), dpi=300)
            plt.close()
    
    # Create plots by experiment type (adaptive vs fixed, k value)
    experiment_categories = {}
    for exp_name in metrics_by_exp.keys():
        if 'adaptive' in exp_name:
            category = 'adaptive'
        elif 'fixed' in exp_name:
            category = 'fixed'
        else:
            category = 'other'
        
        if category not in experiment_categories:
            experiment_categories[category] = []
        experiment_categories[category].append(exp_name)
    
    # Plot by category
    for category, exps in experiment_categories.items():
        category_data = df[df['experiment'].isin(exps)]
        if not category_data.empty:
            plt.figure(figsize=(10, 6))
            chart = sns.barplot(x='experiment', y='value', hue='metric', data=category_data)
            plt.title(f'Metrics for {category.capitalize()} Experiments')
            plt.ylabel('Value')
            plt.xlabel('Experiment')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'metrics_comparison_{category}.png'), dpi=300)
            plt.close()
    
    # Plot by policy type
    policy_categories = {}
    for exp_name in metrics_by_exp.keys():
        if 'learned' in exp_name:
            category = 'learned'
        else:
            category = 'uniform'
        
        if category not in policy_categories:
            policy_categories[category] = []
        policy_categories[category].append(exp_name)
    
    # Create a figure comparing metrics by policy type
    plt.figure(figsize=(12, 6))
    # Reshape the data for easier plotting
    policy_data = []
    for policy, exps in policy_categories.items():
        for exp_name in exps:
            for metric in metrics:
                if metric in metrics_by_exp[exp_name]['overall']:
                    policy_data.append({
                        'policy': policy,
                        'experiment': exp_name,
                        'metric': metric.upper(),
                        'value': metrics_by_exp[exp_name]['overall'][metric]
                    })
    
    policy_df = pd.DataFrame(policy_data)
    if not policy_df.empty:
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(x='policy', y='value', hue='metric', data=policy_df)
        plt.title('Metrics by Policy Type')
        plt.ylabel('Value')
        plt.xlabel('Policy Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison_by_policy.png'), dpi=300)
        plt.close()
    
    # Create a figure comparing all experiments
    plt.figure(figsize=(15, 8))
    g = sns.catplot(
        data=df, kind="bar",
        x="experiment", y="value", hue="metric",
        height=6, aspect=1.5
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.set_axis_labels("Experiment", "Value")
    g.fig.suptitle('Metrics Comparison by Experiment', y=1.02)
    g.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_by_experiment.png'), dpi=300)
    plt.close()

def plot_per_horizon_metrics(metrics_by_exp: Dict[str, Any], output_dir: str) -> None:
    """
    Plot metrics per prediction horizon.
    
    Args:
        metrics_by_exp: Dictionary of metrics by experiment name
        output_dir: Directory to save plots
    """
    if not metrics_by_exp:
        print("No metrics available for plotting")
        return
    
    metrics = ['wql', 'crps', 'mase']
    
    # Check if per_horizon metrics exist
    has_per_horizon = any('per_horizon' in metrics_data and 
                           any(metric in metrics_data['per_horizon'] for metric in metrics)
                           for metrics_data in metrics_by_exp.values())
    
    if not has_per_horizon:
        print("No per-horizon metrics found")
        return
    
    # Create a figure for per-horizon metrics
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(len(metrics), 1, figure=fig)
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i, 0])
        
        for exp_name, metrics_data in metrics_by_exp.items():
            if ('per_horizon' in metrics_data and 
                metric in metrics_data['per_horizon'] and 
                metrics_data['per_horizon'][metric]):
                
                values = metrics_data['per_horizon'][metric]
                horizons = list(range(1, len(values) + 1))
                
                ax.plot(horizons, values, marker='o', linewidth=2, label=exp_name)
        
        ax.set_title(f'{metric.upper()} by Forecast Horizon')
        ax.set_xlabel('Horizon')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_horizon_metrics.png'), dpi=300)
    plt.close()
    
    # Create individual plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for exp_name, metrics_data in metrics_by_exp.items():
            if ('per_horizon' in metrics_data and 
                metric in metrics_data['per_horizon'] and 
                metrics_data['per_horizon'][metric]):
                
                values = metrics_data['per_horizon'][metric]
                horizons = list(range(1, len(values) + 1))
                
                plt.plot(horizons, values, marker='o', linewidth=2, label=exp_name)
        
        plt.title(f'{metric.upper()} by Forecast Horizon')
        plt.xlabel('Horizon')
        plt.ylabel(f'{metric.upper()} Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_horizon_{metric}.png'), dpi=300)
        plt.close()

def plot_training_curves(curves_by_exp: Dict[str, Any], output_dir: str) -> None:
    """
    Plot training curves.
    
    Args:
        curves_by_exp: Dictionary of training curves by experiment name
        output_dir: Directory to save plots
    """
    if not curves_by_exp:
        print("No training logs available for plotting")
        return
    
    metrics = ['loss', 'reward', 'entropy', 'k']
    
    # Create a figure for training curves
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(len(metrics), 1, figure=fig)
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i, 0])
        
        for exp_name, curves_data in curves_by_exp.items():
            if metric in curves_data and curves_data[metric]:
                values = curves_data[metric]
                epochs = curves_data.get('epochs', list(range(1, len(values) + 1)))
                
                ax.plot(epochs, values, linewidth=2, label=exp_name)
        
        ax.set_title(f'{metric.capitalize()} During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # Create individual plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for exp_name, curves_data in curves_by_exp.items():
            if metric in curves_data and curves_data[metric]:
                values = curves_data[metric]
                epochs = curves_data.get('epochs', list(range(1, len(values) + 1)))
                
                plt.plot(epochs, values, linewidth=2, label=exp_name)
        
        plt.title(f'{metric.capitalize()} During Training')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_{metric}.png'), dpi=300)
        plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize results from ablation studies')
    parser.add_argument('--results_dir', type=str, default='./results/synthetic_data',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./results/synthetic_plots',
                        help='Directory to save plots')
    parser.add_argument('--experiments', type=str, nargs='*',
                        help='List of experiments to include')
    args = parser.parse_args()
    
    # Load metrics from all experiments
    metrics_by_exp = load_metrics(args.results_dir, args.experiments)
    
    # Load training curves from all experiments
    curves_by_exp = load_training_curves(args.results_dir, args.experiments)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_by_exp, args.output_dir)
    
    # Plot per-horizon metrics
    plot_per_horizon_metrics(metrics_by_exp, args.output_dir)
    
    # Plot training curves
    plot_training_curves(curves_by_exp, args.output_dir)
    
    print(f"Plots generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main() 