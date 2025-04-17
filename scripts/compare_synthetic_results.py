#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to compare results from different synthetic data experiments.

This script loads metrics from all experiments in the synthetic_data directory
and creates comparison plots for overall metrics, per-horizon metrics, and per-series metrics.
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
from typing import Dict, List, Tuple, Optional, Any, Union

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot comparison of synthetic results')
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Directory containing the experiment results')
    parser.add_argument('--output_dir', type=str, default='results/synthetic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 6],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str) -> List[str]:
    """Find all experiment directories in the results directory."""
    return [d for d in os.listdir(results_dir) 
            if os.path.isdir(os.path.join(results_dir, d))]

def load_metrics(experiment_dir: str) -> Dict:
    """Load metrics from a single experiment directory."""
    metrics_path = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Warning: No metrics file found at {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def get_experiment_label(exp_name: str) -> str:
    """Generate a readable label for the experiment."""
    components = exp_name.split('_')
    if 'fixed' in exp_name:
        return f"Fixed k={components[-1]}"
    elif 'adaptive' in exp_name:
        return f"Adaptive k={components[-1]}"
    elif 'learned' in exp_name:
        return "Learned Policy"
    else:
        return exp_name

def load_all_metrics(results_dir: str) -> pd.DataFrame:
    """Load metrics from all experiments into a DataFrame."""
    exp_dirs = find_experiment_dirs(results_dir)
    
    all_metrics = []
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        metrics = load_metrics(full_path)
        
        if metrics is None:
            continue
        
        # Extract experiment information
        exp_label = get_experiment_label(exp_dir)
        
        # Extract overall metrics
        row = {
            'experiment': exp_dir,
            'label': exp_label
        }
        
        # Add overall metrics
        for metric, value in metrics['overall'].items():
            row[metric] = value
        
        # Add metadata if available
        if 'metadata' in metrics and 'config' in metrics['metadata']:
            for key, value in metrics['metadata']['config'].items():
                row[f"config_{key}"] = value
        
        all_metrics.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    return df

def plot_overall_metrics(df: pd.DataFrame, metrics: List[str], 
                         figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot overall metrics comparison."""
    # Set up the figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Sort by experiment label
    df = df.sort_values('label')
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            print(f"Warning: Metric {metric} not found in results")
            continue
        
        ax = axes[i]
        sns.barplot(x='label', y=metric, data=df, ax=ax)
        ax.set_title(f'Overall {metric.upper()}')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=dpi)
    plt.close()

def plot_per_horizon_metrics(results_dir: str, exp_dirs: List[str], metrics: List[str],
                             figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot per-horizon metrics for all experiments."""
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Load and plot data for each experiment
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        metrics_data = load_metrics(full_path)
        
        if metrics_data is None or 'per_horizon' not in metrics_data:
            continue
        
        exp_label = get_experiment_label(exp_dir)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric not in metrics_data['per_horizon']:
                continue
            
            horizons = range(1, len(metrics_data['per_horizon'][metric]) + 1)
            axes[i].plot(horizons, metrics_data['per_horizon'][metric], 
                         marker='o', label=exp_label)
            axes[i].set_title(f'{metric.upper()} by Forecast Horizon')
            axes[i].set_xlabel('Forecast Horizon')
            axes[i].set_ylabel(metric.upper())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_horizon_metrics.png'), dpi=dpi)
    plt.close()

def plot_per_series_metrics(results_dir: str, exp_dirs: List[str], metrics: List[str],
                           figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot per-series metrics distributions for all experiments."""
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Prepare data for boxplots
    boxplot_data = {metric: [] for metric in metrics}
    labels = []
    
    # Load data for each experiment
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        metrics_data = load_metrics(full_path)
        
        if metrics_data is None or 'per_series' not in metrics_data:
            continue
        
        exp_label = get_experiment_label(exp_dir)
        labels.append(exp_label)
        
        # Collect data for each metric
        for metric in metrics:
            if metric not in metrics_data['per_series']:
                boxplot_data[metric].append([])
            else:
                boxplot_data[metric].append(metrics_data['per_series'][metric])
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Check if we have data to plot
        if not any(boxplot_data[metric]):
            continue
        
        # Create boxplot
        ax.boxplot(boxplot_data[metric], labels=labels)
        ax.set_title(f'{metric.upper()} Distribution Across Series')
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_series_metrics.png'), dpi=dpi)
    plt.close()

def plot_config_comparison(df: pd.DataFrame, metrics: List[str], 
                          figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot metrics grouped by configuration parameters."""
    # Check if we have configuration data
    config_cols = [col for col in df.columns if col.startswith('config_')]
    
    if not config_cols:
        return
    
    # Plot for each configuration parameter
    for config_param in config_cols:
        param_name = config_param.replace('config_', '')
        
        # Skip if parameter has only one value
        if df[config_param].nunique() <= 1:
            continue
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            
            ax = axes[i]
            sns.barplot(x=config_param, y=metric, data=df, ax=ax)
            ax.set_title(f'{metric.upper()} by {param_name}')
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_by_{param_name}.png'), dpi=dpi)
        plt.close()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all metrics
    df = load_all_metrics(args.results_dir)
    
    if df.empty:
        print(f"No metrics found in {args.results_dir}")
        return
    
    # Get experiment directories
    exp_dirs = find_experiment_dirs(args.results_dir)
    
    # Plot overall metrics comparison
    plot_overall_metrics(df, args.metrics, tuple(args.figsize), args.output_dir, args.dpi)
    
    # Plot per-horizon metrics
    plot_per_horizon_metrics(args.results_dir, exp_dirs, args.metrics, 
                            tuple(args.figsize), args.output_dir, args.dpi)
    
    # Plot per-series metrics
    plot_per_series_metrics(args.results_dir, exp_dirs, args.metrics,
                           tuple(args.figsize), args.output_dir, args.dpi)
    
    # Plot configuration comparisons
    plot_config_comparison(df, args.metrics, tuple(args.figsize), args.output_dir, args.dpi)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 