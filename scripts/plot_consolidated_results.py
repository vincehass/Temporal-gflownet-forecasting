#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consolidated script to compare results from different experiments and ablation studies.

This script loads metrics from experiments and creates comparison plots for various metrics,
with options for filtering by dataset, experiment type, and visualization styles.
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
import yaml
from matplotlib.colors import LinearSegmentedColormap

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot comparison of experiment results')
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Directory containing the experiment results')
    parser.add_argument('--output_dir', type=str, default='results/consolidated_plots',
                        help='Directory to save the plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot')
    parser.add_argument('--dataset_filter', type=str, default=None,
                        help='Filter experiments by dataset')
    parser.add_argument('--experiment_type', type=str, default=None, choices=['fixed', 'adaptive', 'learned_policy'],
                        help='Filter experiments by type')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-darkgrid',
                        help='Matplotlib style to use')
    parser.add_argument('--colormap', type=str, default='viridis',
                        help='Colormap to use for plots')
    parser.add_argument('--legend_loc', type=str, default='best',
                        help='Legend location')
    parser.add_argument('--comparison_type', type=str, default='all',
                        choices=['all', 'overall', 'horizon', 'series', 'parameters'],
                        help='Type of comparison plots to generate')
    
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
            config = yaml.safe_load(f)
        
        params = {}
        # Extract relevant parameters
        if 'dataset' in config:
            if isinstance(config['dataset'], dict):
                params['dataset'] = config['dataset'].get('name', 'unknown')
            else:
                params['dataset'] = config['dataset']
        
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
            # Check for adaptive flag
            if 'adaptive' in quant_params:
                params['adaptive'] = quant_params.get('adaptive', False)
                if params['adaptive'] and not params['quantization_type']:
                    params['quantization_type'] = 'adaptive'
                elif not params['adaptive'] and not params['quantization_type']:
                    params['quantization_type'] = 'fixed'
        
        if 'policy' in config:
            policy_params = config['policy']
            params['backward_policy_type'] = policy_params.get('backward_policy_type', 'uniform')
            if params['backward_policy_type'] == 'learned' and not params.get('quantization_type'):
                params['quantization_type'] = 'learned_policy'
        
        if 'training' in config:
            train_params = config['training']
            params['learning_rate'] = train_params.get('learning_rate', None)
            params['batch_size'] = train_params.get('batch_size', None)
            params['epochs'] = train_params.get('epochs', None)
        
        if 'gfn' in config:
            gfn_params = config['gfn']
            params['lambda_entropy'] = gfn_params.get('lambda_entropy', None)
            params['Z_init'] = gfn_params.get('Z_init', None)
            params['Z_lr'] = gfn_params.get('Z_lr', None)
        
        return params
    
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
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
    if params:
        # Construct label parts
        parts = []
        
        # Quantization type
        quant_type = params.get('quantization_type')
        if quant_type == 'fixed':
            k = params.get('k_initial', 'unknown')
            parts.append(f"Fixed k={k}")
        
        elif quant_type == 'adaptive':
            k_initial = params.get('k_initial', 'unknown')
            k_max = params.get('k_max', 'unknown')
            parts.append(f"Adaptive k={k_initial}")
            if k_max and k_max != k_initial:
                parts[-1] += f"-{k_max}"
        
        elif quant_type == 'learned_policy':
            parts.append("Learned Policy")
        
        # Backward policy type if different from default
        policy_type = params.get('backward_policy_type')
        if policy_type and policy_type != 'uniform' and quant_type != 'learned_policy':
            parts.append(f"{policy_type.capitalize()} BP")
        
        # Entropy param if specified
        entropy = params.get('lambda_entropy')
        if entropy is not None:
            parts.append(f"λ={entropy}")
        
        if parts:
            return ", ".join(parts)
    
    # Default fallback
    return exp_name

def load_all_metrics(results_dir: str, dataset_filter: Optional[str] = None, 
                    experiment_type: Optional[str] = None) -> pd.DataFrame:
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
        
        # Apply experiment type filter if specified
        if experiment_type and params.get('quantization_type') != experiment_type:
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
            'label': exp_label,
            'path': full_path
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

def setup_plot_style(style: str, colormap: str):
    """Set up the matplotlib style and colormap."""
    try:
        plt.style.use(style)
    except:
        print(f"Warning: Style {style} not found. Using default style.")
    
    # Create custom colorblind-friendly colormap if specified
    if colormap == 'colorblind':
        colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000']
        cmap = LinearSegmentedColormap.from_list('colorblind', colors)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        return cmap
    
    return plt.cm.get_cmap(colormap)

def plot_overall_metrics(df: pd.DataFrame, metrics: List[str], 
                         figsize: Tuple[int, int], output_dir: str, 
                         legend_loc: str = 'best', dpi: int = 300, 
                         colormap=None, dataset_name: str = None):
    """Plot overall metrics comparison."""
    if df.empty:
        print("No data to plot overall metrics")
        return
    
    # Set up the figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Determine sort order based on quantization type
    sort_order = {
        'fixed': 0,
        'adaptive': 1,
        'learned_policy': 2
    }
    
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    if 'quantization_type' in df.columns:
        plot_df['sort_key'] = plot_df['quantization_type'].map(sort_order).fillna(999)
        plot_df = plot_df.sort_values(['sort_key', 'k_initial'])
    else:
        plot_df = plot_df.sort_values('label')
    
    # Set colors based on quantization type
    palette = {
        'fixed': '#0072B2',  # Blue
        'adaptive': '#009E73',  # Green
        'learned_policy': '#D55E00'  # Red-orange
    }
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric not in plot_df.columns:
            print(f"Warning: Metric {metric} not found in results")
            continue
        
        ax = axes[i]
        
        if 'quantization_type' in plot_df.columns and plot_df['quantization_type'].nunique() > 1:
            sns.barplot(x='label', y=metric, data=plot_df, ax=ax, 
                        hue='quantization_type', palette=palette, dodge=False)
            ax.legend(title='Quantization Type', loc=legend_loc)
        else:
            # Use colormap for gradients when there's only one quantization type
            num_exps = len(plot_df)
            colors = [colormap(i/num_exps) for i in range(num_exps)] if colormap else None
            
            # Use barplot with custom colors
            bars = sns.barplot(x='label', y=metric, data=plot_df, ax=ax)
            
            # Apply custom colors if specified
            if colors:
                for j, bar in enumerate(bars.patches):
                    bar.set_color(colors[j % len(colors)])
        
        ax.set_title(f'Overall {metric.upper()}')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        
        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    
    # Add dataset info to filename if available
    if dataset_name is None:
        dataset_name = "all_datasets"
        if 'dataset' in df.columns and df['dataset'].nunique() == 1:
            dataset_name = df['dataset'].iloc[0]
    
    plt.savefig(os.path.join(output_dir, f'overall_metrics_{dataset_name}.png'), dpi=dpi)
    plt.close()

def plot_per_horizon_metrics(df: pd.DataFrame, metrics: List[str],
                             figsize: Tuple[int, int], output_dir: str, 
                             legend_loc: str = 'best', dpi: int = 300,
                             colormap=None, dataset_name: str = None):
    """Plot per-horizon metrics for all experiments."""
    if df.empty:
        print("No data to plot per-horizon metrics")
        return
    
    # Figure out how many metrics we're plotting
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # List to keep track of experiment details for legend
    exp_details = []
    
    # Load and plot data for each experiment
    for idx, row in df.iterrows():
        exp_path = row['path']
        
        # Load metrics
        metrics_data = load_metrics(exp_path)
        if metrics_data is None or 'per_horizon' not in metrics_data:
            continue
        
        # Generate label
        exp_label = row['label']
        exp_details.append({
            'label': exp_label,
            'quant_type': row.get('quantization_type', 'unknown')
        })
        
        # Determine line style and color based on quantization type
        line_style = '-'
        marker = 'o'
        color = None
        
        quant_type = row.get('quantization_type')
        if quant_type == 'fixed':
            line_style = '-'
            color = '#0072B2'  # Blue
        elif quant_type == 'adaptive':
            line_style = '--'
            color = '#009E73'  # Green
        elif quant_type == 'learned_policy':
            line_style = '-.'
            color = '#D55E00'  # Red-orange
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric not in metrics_data['per_horizon']:
                continue
            
            horizons = range(1, len(metrics_data['per_horizon'][metric]) + 1)
            axes[i].plot(horizons, metrics_data['per_horizon'][metric], 
                         marker=marker, linestyle=line_style, 
                         label=exp_label, color=color)
            axes[i].set_title(f'{metric.upper()} by Forecast Horizon')
            axes[i].set_xlabel('Forecast Horizon')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3)
    
    # Add legends
    for ax in axes:
        ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    # Add dataset info to filename if provided
    if dataset_name is None:
        dataset_name = "all_datasets"
        if 'dataset' in df.columns and df['dataset'].nunique() == 1:
            dataset_name = df['dataset'].iloc[0]
    
    filename = f'per_horizon_metrics_{dataset_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.close()

def plot_per_series_metrics(df: pd.DataFrame, metrics: List[str],
                            figsize: Tuple[int, int], output_dir: str, 
                            legend_loc: str = 'best', dpi: int = 300,
                            colormap=None, dataset_name: str = None):
    """Plot per-series metrics distributions for all experiments."""
    if df.empty:
        print("No data to plot per-series metrics")
        return
        
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Prepare data for boxplots
    boxplot_data = {metric: [] for metric in metrics}
    labels = []
    colors = []
    
    # Load data for each experiment
    for idx, row in df.iterrows():
        exp_path = row['path']
        
        # Load metrics
        metrics_data = load_metrics(exp_path)
        if metrics_data is None or 'per_series' not in metrics_data:
            continue
        
        # Generate label
        exp_label = row['label']
        labels.append(exp_label)
        
        # Determine color based on quantization type
        quant_type = row.get('quantization_type')
        if quant_type == 'fixed':
            colors.append('#0072B2')  # Blue
        elif quant_type == 'adaptive':
            colors.append('#009E73')  # Green
        elif quant_type == 'learned_policy':
            colors.append('#D55E00')  # Red-orange
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
        
        # Create boxplot with tick_labels 
        boxes = ax.boxplot(boxplot_data[metric], tick_labels=labels, patch_artist=True)
        
        # Set box colors
        for box, color in zip(boxes['boxes'], colors):
            box.set(facecolor=color, alpha=0.6)
        
        ax.set_title(f'{metric.upper()} Distribution Across Series')
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add dataset info to filename if provided
    if dataset_name is None:
        dataset_name = "all_datasets"
        if 'dataset' in df.columns and df['dataset'].nunique() == 1:
            dataset_name = df['dataset'].iloc[0]
    
    filename = f'per_series_metrics_{dataset_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.close()

def plot_parameter_comparison(df: pd.DataFrame, metrics: List[str], 
                             figsize: Tuple[int, int], output_dir: str, 
                             legend_loc: str = 'best', dpi: int = 300,
                             colormap=None, dataset_name: str = None):
    """Plot metrics grouped by different parameters."""
    if df.empty:
        print("No data to plot parameter comparisons")
        return
        
    # Parameters to analyze
    params_to_compare = [
        'quantization_type', 'k_initial', 'k_max', 'd_model', 'nlayers',
        'learning_rate', 'batch_size', 'lambda_entropy', 'backward_policy_type'
    ]
    
    # Filter to parameters that exist in dataframe and have multiple values
    params_to_compare = [p for p in params_to_compare 
                         if p in df.columns and df[p].nunique() > 1]
    
    if not params_to_compare:
        print("No parameters with multiple values found for comparison plots")
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
                if 'quantization_type' in df.columns and param != 'quantization_type':
                    sns.barplot(x=param, y=metric, hue='quantization_type', data=df, ax=ax)
                    ax.legend(title='Quantization Type', loc=legend_loc)
                else:
                    # Use colormap for gradients
                    bars = sns.barplot(x=param, y=metric, data=df, ax=ax)
                    
                    # Apply custom colors if colormap specified
                    if colormap:
                        num_categories = df[param].nunique()
                        colors = [colormap(i/num_categories) for i in range(num_categories)]
                        for j, bar in enumerate(bars.patches):
                            bar.set_color(colors[j % len(colors)])
            
            # For numeric parameters
            else:
                if 'quantization_type' in df.columns:
                    sns.scatterplot(x=param, y=metric, hue='quantization_type', 
                                   data=df, ax=ax, s=100, alpha=0.7)
                    # Add regression line
                    sns.regplot(x=param, y=metric, data=df, ax=ax, 
                               scatter=False, ci=None, line_kws={"color": "black", "alpha": 0.5})
                    ax.legend(title='Quantization Type', loc=legend_loc)
                else:
                    # Use colormap for points
                    if colormap:
                        sns.scatterplot(x=param, y=metric, data=df, ax=ax, 
                                       palette=colormap, s=100, alpha=0.7)
                    else:
                        sns.scatterplot(x=param, y=metric, data=df, ax=ax, s=100, alpha=0.7)
                    
                    # Add regression line
                    sns.regplot(x=param, y=metric, data=df, ax=ax, 
                               scatter=False, ci=None, line_kws={"color": "black", "alpha": 0.5})
            
            ax.set_title(f'{metric.upper()} by {param.replace("_", " ").title()}')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Add dataset info to filename if provided
        if dataset_name is None:
            dataset_name = "all_datasets"
            if 'dataset' in df.columns and df['dataset'].nunique() == 1:
                dataset_name = df['dataset'].iloc[0]
        
        plt.savefig(os.path.join(output_dir, f'comparison_by_{param}_{dataset_name}.png'), dpi=dpi)
        plt.close()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up plot style
    colormap = setup_plot_style(args.style, args.colormap)
    
    # Load all metrics into a DataFrame
    df = load_all_metrics(args.results_dir, args.dataset_filter, args.experiment_type)
    
    if df.empty:
        print(f"No metrics found in {args.results_dir} with the specified filters")
        return
    
    # Set dataset name for filenames
    dataset_name = None
    if args.dataset_filter:
        dataset_name = args.dataset_filter
    
    # Generate plots based on comparison type
    if args.comparison_type in ['all', 'overall']:
        plot_overall_metrics(df, args.metrics, tuple(args.figsize), args.output_dir, 
                            args.legend_loc, args.dpi, colormap, dataset_name)
    
    if args.comparison_type in ['all', 'horizon']:
        plot_per_horizon_metrics(df, args.metrics, tuple(args.figsize), args.output_dir, 
                                args.legend_loc, args.dpi, colormap, dataset_name)
    
    if args.comparison_type in ['all', 'series']:
        plot_per_series_metrics(df, args.metrics, tuple(args.figsize), args.output_dir, 
                               args.legend_loc, args.dpi, colormap, dataset_name)
    
    if args.comparison_type in ['all', 'parameters']:
        plot_parameter_comparison(df, args.metrics, tuple(args.figsize), args.output_dir, 
                                 args.legend_loc, args.dpi, colormap, dataset_name)
    
    print(f"Plots saved to {args.output_dir}")
    
    # Print summary of experiments analyzed
    print("\nSummary of experiments analyzed:")
    print(f"Total experiments: {len(df)}")
    
    if 'quantization_type' in df.columns:
        print("\nBy quantization type:")
        print(df['quantization_type'].value_counts())
    
    if 'k_initial' in df.columns:
        print("\nBy initial k value:")
        print(df['k_initial'].value_counts())
    
    if 'dataset' in df.columns:
        print("\nBy dataset:")
        print(df['dataset'].value_counts())
    
    # Print summary of metrics
    print("\nMetrics summary:")
    for metric in args.metrics:
        if metric in df.columns:
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {df[metric].mean():.4f}")
            print(f"  Min:  {df[metric].min():.4f}")
            print(f"  Max:  {df[metric].max():.4f}")
            
            # Print top 3 experiments for this metric (lower is better)
            print(f"\nTop 3 experiments by {metric.upper()} (lower is better):")
            top3 = df.sort_values(by=metric).head(3)
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                print(f"  {i}. {row['label']} - {row[metric]:.4f}")

if __name__ == "__main__":
    main() 