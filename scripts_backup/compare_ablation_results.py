#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to compare results from different ablation studies.

This script loads metrics from all experiments in the wandb_ablations directory
and creates comparison plots for various metrics.
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
import re

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot comparison of ablation results')
    parser.add_argument('--results_dir', type=str, default='results/wandb_ablations',
                        help='Directory containing the experiment results')
    parser.add_argument('--output_dir', type=str, default='results/ablation_plots',
                        help='Directory to save the plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot')
    parser.add_argument('--dataset_filter', type=str, default=None,
                        help='Filter experiments by dataset')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
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

def extract_params_from_config(config_path: str) -> Dict:
    """Extract important parameters from config file."""
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        params = {}
        # Extract relevant parameters
        if 'dataset' in config:
            params['dataset'] = config['dataset'].get('name', 'unknown')
        
        if 'model' in config:
            model_params = config['model']
            params['d_model'] = model_params.get('d_model', None)
            params['nlayers'] = model_params.get('nlayers', None)
        
        if 'quantization' in config:
            quant_params = config['quantization']
            params['quantization_type'] = quant_params.get('type', None)
            params['vmin'] = quant_params.get('vmin', None)
            params['vmax'] = quant_params.get('vmax', None)
            params['k_initial'] = quant_params.get('k_initial', None)
            params['k_max'] = quant_params.get('k_max', None)
        
        if 'training' in config:
            train_params = config['training']
            params['learning_rate'] = train_params.get('learning_rate', None)
            params['batch_size'] = train_params.get('batch_size', None)
            params['epochs'] = train_params.get('epochs', None)
        
        return params
    
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def get_experiment_label(exp_name: str, params: Dict = None) -> str:
    """Generate a readable label for the experiment."""
    # First try to extract information from experiment name
    if 'fixed_k' in exp_name:
        match = re.search(r'fixed_k(\d+)', exp_name)
        if match:
            k_value = match.group(1)
            return f"Fixed k={k_value}"
    
    elif 'adaptive_k' in exp_name:
        match = re.search(r'adaptive_k(\d+)', exp_name)
        if match:
            k_value = match.group(1)
            return f"Adaptive k={k_value}"
    
    elif 'learned_policy' in exp_name:
        return "Learned Policy"
    
    # If we have config params, use those for better labeling
    if params and params.get('quantization_type'):
        quant_type = params['quantization_type']
        
        if quant_type == 'fixed':
            k = params.get('k_initial', 'unknown')
            return f"Fixed k={k}"
        
        elif quant_type == 'adaptive':
            k_initial = params.get('k_initial', 'unknown')
            k_max = params.get('k_max', 'unknown')
            return f"Adaptive k={k_initial}-{k_max}"
        
        elif quant_type == 'learned_policy':
            return "Learned Policy"
    
    # Default fallback
    return exp_name

def load_all_metrics(results_dir: str, dataset_filter: Optional[str] = None) -> pd.DataFrame:
    """Load metrics from all experiments into a DataFrame."""
    exp_dirs = find_experiment_dirs(results_dir)
    
    all_metrics = []
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        
        # Load config parameters first
        config_path = os.path.join(full_path, 'config.yaml')
        params = extract_params_from_config(config_path)
        
        # Apply dataset filter if specified
        if dataset_filter and params.get('dataset') != dataset_filter:
            continue
        
        # Load metrics
        metrics = load_metrics(full_path)
        if metrics is None:
            continue
        
        # Extract experiment information
        exp_label = get_experiment_label(exp_dir, params)
        
        # Extract overall metrics
        row = {
            'experiment': exp_dir,
            'label': exp_label
        }
        
        # Add config parameters
        for key, value in params.items():
            row[key] = value
        
        # Add overall metrics
        if 'overall' in metrics:
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
    
    # Determine sort order based on quantization type
    sort_order = {
        'fixed': 0,
        'adaptive': 1,
        'learned_policy': 2
    }
    
    if 'quantization_type' in df.columns:
        df['sort_key'] = df['quantization_type'].map(sort_order).fillna(999)
        df = df.sort_values(['sort_key', 'k_initial'])
    
    # Set colors based on quantization type
    palette = {
        'fixed': 'blue',
        'adaptive': 'green',
        'learned_policy': 'red'
    }
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            print(f"Warning: Metric {metric} not found in results")
            continue
        
        ax = axes[i]
        
        if 'quantization_type' in df.columns:
            sns.barplot(x='label', y=metric, data=df, ax=ax, 
                        hue='quantization_type', palette=palette, dodge=False)
            ax.legend(title='Quantization Type')
        else:
            sns.barplot(x='label', y=metric, data=df, ax=ax)
        
        ax.set_title(f'Overall {metric.upper()}')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Add dataset info to filename if available
    dataset_name = "all_datasets"
    if 'dataset' in df.columns and df['dataset'].nunique() == 1:
        dataset_name = df['dataset'].iloc[0]
    
    plt.savefig(os.path.join(output_dir, f'overall_metrics_{dataset_name}.png'), dpi=dpi)
    plt.close()

def plot_per_horizon_metrics(results_dir: str, exp_dirs: List[str], metrics: List[str],
                             figsize: Tuple[int, int], output_dir: str, 
                             dataset_filter: Optional[str] = None, dpi: int = 300):
    """Plot per-horizon metrics for all experiments."""
    # Figure out how many metrics we're plotting
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # List to keep track of experiment details for legend
    exp_details = []
    
    # Load and plot data for each experiment
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        
        # Load config to check dataset
        config_path = os.path.join(full_path, 'config.yaml')
        params = extract_params_from_config(config_path)
        
        # Apply dataset filter if specified
        if dataset_filter and params.get('dataset') != dataset_filter:
            continue
        
        # Load metrics
        metrics_data = load_metrics(full_path)
        if metrics_data is None or 'per_horizon' not in metrics_data:
            continue
        
        # Generate label
        exp_label = get_experiment_label(exp_dir, params)
        exp_details.append({
            'label': exp_label,
            'quant_type': params.get('quantization_type', 'unknown')
        })
        
        # Determine line style and color based on quantization type
        line_style = '-'
        marker = 'o'
        color = None
        
        quant_type = params.get('quantization_type')
        if quant_type == 'fixed':
            line_style = '-'
            color = 'blue'
        elif quant_type == 'adaptive':
            line_style = '--'
            color = 'green'
        elif quant_type == 'learned_policy':
            line_style = '-.'
            color = 'red'
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric not in metrics_data['per_horizon']:
                continue
            
            horizons = range(1, len(metrics_data['per_horizon'][metric]) + 1)
            axes[i].plot(horizons, metrics_data['per_horizon'][metric], 
                         marker=marker, linestyle=line_style, label=exp_label, color=color)
            axes[i].set_title(f'{metric.upper()} by Forecast Horizon')
            axes[i].set_xlabel('Forecast Horizon')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3)
    
    # Add legends
    for ax in axes:
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Add dataset info to filename if dataset_filter is specified
    filename = 'per_horizon_metrics'
    if dataset_filter:
        filename += f'_{dataset_filter}'
    
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=dpi)
    plt.close()

def plot_per_series_metrics(results_dir: str, exp_dirs: List[str], metrics: List[str],
                            figsize: Tuple[int, int], output_dir: str, 
                            dataset_filter: Optional[str] = None, dpi: int = 300):
    """Plot per-series metrics distributions for all experiments."""
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Prepare data for boxplots
    boxplot_data = {metric: [] for metric in metrics}
    labels = []
    colors = []
    
    # Load data for each experiment
    for exp_dir in exp_dirs:
        full_path = os.path.join(results_dir, exp_dir)
        
        # Load config to check dataset
        config_path = os.path.join(full_path, 'config.yaml')
        params = extract_params_from_config(config_path)
        
        # Apply dataset filter if specified
        if dataset_filter and params.get('dataset') != dataset_filter:
            continue
        
        # Load metrics
        metrics_data = load_metrics(full_path)
        if metrics_data is None or 'per_series' not in metrics_data:
            continue
        
        # Generate label
        exp_label = get_experiment_label(exp_dir, params)
        labels.append(exp_label)
        
        # Determine color based on quantization type
        quant_type = params.get('quantization_type')
        if quant_type == 'fixed':
            colors.append('blue')
        elif quant_type == 'adaptive':
            colors.append('green')
        elif quant_type == 'learned_policy':
            colors.append('red')
        else:
            colors.append('gray')
        
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
        boxes = ax.boxplot(boxplot_data[metric], labels=labels, patch_artist=True)
        
        # Set box colors
        for box, color in zip(boxes['boxes'], colors):
            box.set(facecolor=color, alpha=0.6)
        
        ax.set_title(f'{metric.upper()} Distribution Across Series')
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add dataset info to filename if dataset_filter is specified
    filename = 'per_series_metrics'
    if dataset_filter:
        filename += f'_{dataset_filter}'
    
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=dpi)
    plt.close()

def plot_parameter_comparison(df: pd.DataFrame, metrics: List[str], 
                             figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot metrics grouped by different parameters."""
    # Parameters to analyze
    params_to_compare = [
        'quantization_type', 'k_initial', 'k_max', 
        'd_model', 'nlayers', 'learning_rate', 'batch_size'
    ]
    
    # Filter to parameters that exist in dataframe and have multiple values
    params_to_compare = [p for p in params_to_compare 
                         if p in df.columns and df[p].nunique() > 1]
    
    if not params_to_compare:
        return
    
    # Plot for each parameter
    for param in params_to_compare:
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            
            ax = axes[i]
            
            # For categorical parameters
            if df[param].dtype == 'object' or df[param].nunique() < 10:
                if 'quantization_type' in df.columns:
                    sns.barplot(x=param, y=metric, hue='quantization_type', data=df, ax=ax)
                    ax.legend(title='Quantization Type')
                else:
                    sns.barplot(x=param, y=metric, data=df, ax=ax)
            # For numeric parameters
            else:
                if 'quantization_type' in df.columns:
                    sns.scatterplot(x=param, y=metric, hue='quantization_type', data=df, ax=ax)
                    ax.legend(title='Quantization Type')
                else:
                    sns.scatterplot(x=param, y=metric, data=df, ax=ax)
            
            ax.set_title(f'{metric.upper()} by {param}')
            ax.set_xlabel(param)
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Add dataset info to filename if available
        dataset_name = "all_datasets"
        if 'dataset' in df.columns and df['dataset'].nunique() == 1:
            dataset_name = df['dataset'].iloc[0]
        
        plt.savefig(os.path.join(output_dir, f'comparison_by_{param}_{dataset_name}.png'), dpi=dpi)
        plt.close()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(args.results_dir)
    if not exp_dirs:
        print(f"No experiment directories found in {args.results_dir}")
        return
    
    # Load all metrics into a DataFrame
    df = load_all_metrics(args.results_dir, args.dataset_filter)
    
    if df.empty:
        print(f"No metrics found in {args.results_dir}")
        return
    
    # Plot overall metrics comparison
    plot_overall_metrics(df, args.metrics, tuple(args.figsize), args.output_dir, args.dpi)
    
    # Plot per-horizon metrics
    plot_per_horizon_metrics(args.results_dir, exp_dirs, args.metrics, 
                            tuple(args.figsize), args.output_dir, args.dataset_filter, args.dpi)
    
    # Plot per-series metrics
    plot_per_series_metrics(args.results_dir, exp_dirs, args.metrics,
                           tuple(args.figsize), args.output_dir, args.dataset_filter, args.dpi)
    
    # Plot parameter comparisons
    plot_parameter_comparison(df, args.metrics, tuple(args.figsize), args.output_dir, args.dpi)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 