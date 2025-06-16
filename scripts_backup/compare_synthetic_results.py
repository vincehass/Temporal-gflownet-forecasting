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
from matplotlib.ticker import MaxNLocator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare results from synthetic data experiments')
    parser.add_argument('--results_dirs', nargs='+', type=str, 
                        default=['results/synthetic_data', 'results/synthetic_data_full'],
                        help='Directories containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/synthetic_plots',
                        help='Directory to save plots')
    parser.add_argument('--metrics', nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot (default: wql crps mase)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 6],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str) -> List[str]:
    """Find all experiment directories in the results directory."""
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory {results_dir} does not exist")
    
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))]
    return exp_dirs

def load_metrics(experiment_dir: str) -> Dict:
    """Load metrics from a single experiment directory."""
    metrics_path = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Warning: No metrics found in {experiment_dir}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def get_experiment_label(exp_name: str, dataset_name: str = "") -> str:
    """Generate a readable label for the experiment."""
    # Parse the experiment name to extract configuration
    parts = exp_name.split('_')
    
    prefix = f"{dataset_name} " if dataset_name else ""
    
    if 'learned' in exp_name and 'policy' in exp_name:
        return f"{prefix}Learned Policy"
    
    if len(parts) >= 2 and parts[0] in ['fixed', 'adaptive']:
        quantization = parts[0].capitalize()
        k_value = parts[1].replace('k', 'K=')
        return f"{prefix}{quantization} {k_value}"
    
    return f"{prefix}{exp_name}"

def load_all_metrics(results_dirs: List[str]) -> pd.DataFrame:
    """Load metrics from all experiments into a DataFrame."""
    data = []
    
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Warning: Results directory {results_dir} does not exist")
            continue
            
        # Extract dataset name from directory path
        dataset_name = os.path.basename(results_dir).replace('_', ' ').title()
        
        exp_dirs = find_experiment_dirs(results_dir)
        
        for exp_dir in exp_dirs:
            full_path = os.path.join(results_dir, exp_dir)
            metrics = load_metrics(full_path)
            
            if metrics is None:
                continue
            
            # Extract configuration from metadata if available
            config = metrics.get('metadata', {}).get('config', {})
            if not config:
                # Try to parse from directory name
                config = {
                    'quantization': 'learned' if 'learned' in exp_dir else 
                               ('fixed' if 'fixed' in exp_dir else 'adaptive'),
                    'k': int(exp_dir.split('k')[-1]) if 'k' in exp_dir else None,
                    'policy_type': 'learned' if 'learned' in exp_dir else 'uniform'
                }
            
            row = {
                'experiment': exp_dir,
                'dataset': dataset_name,
                'label': get_experiment_label(exp_dir, dataset_name),
                'quantization': config.get('quantization', ''),
                'k': config.get('k', ''),
                'policy_type': config.get('policy_type', ''),
                'entropy_bonus': config.get('entropy_bonus', '')
            }
            
            # Add overall metrics
            for metric, value in metrics.get('overall', {}).items():
                row[f'{metric}'] = value
            
            data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def plot_overall_metrics(df: pd.DataFrame, metrics: List[str], 
                         figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot overall metrics comparison."""
    plt.figure(figsize=tuple(figsize))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        
        # Sort by metric value (ascending)
        sorted_df = df.sort_values(by=metric)
        
        # Create bar plot
        sns.barplot(x='label', y=metric, data=sorted_df)
        
        plt.title(f'{metric.upper()} Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_dataset_comparison(df: pd.DataFrame, metrics: List[str], 
                          figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot comparison between datasets for the same experiment types."""
    if 'dataset' not in df.columns or df['dataset'].nunique() <= 1:
        return
    
    # Get the different experiment types
    exp_types = []
    for _, row in df.iterrows():
        exp_type = f"{row['quantization']}_{row['k']}" if row['quantization'] != 'learned' else "learned_policy"
        if exp_type not in exp_types:
            exp_types.append(exp_type)
    
    for metric in metrics:
        plt.figure(figsize=tuple(figsize))
        
        # Create a dataframe with experiment types and their metrics for each dataset
        pivot_df = df.pivot_table(
            index='dataset',
            columns='experiment',
            values=metric
        )
        
        # Plot grouped bar chart
        pivot_df.plot(kind='bar', ax=plt.gca())
        
        plt.title(f'{metric.upper()} Comparison Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel(metric.upper())
        plt.legend(title='Experiment', loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'dataset_comparison_{metric}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

def plot_per_horizon_metrics(results_dirs: List[str], metrics: List[str], 
                             figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot per-horizon metrics for all experiments across all datasets."""
    # Create a flattened list of (dataset_name, results_dir, exp_dir) tuples
    all_experiments = []
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            continue
            
        dataset_name = os.path.basename(results_dir).replace('_', ' ').title()
        exp_dirs = find_experiment_dirs(results_dir)
        
        for exp_dir in exp_dirs:
            all_experiments.append((dataset_name, results_dir, exp_dir))
    
    for metric in metrics:
        plt.figure(figsize=tuple(figsize))
        
        for dataset_name, results_dir, exp_dir in all_experiments:
            full_path = os.path.join(results_dir, exp_dir)
            metrics_data = load_metrics(full_path)
            
            if metrics_data is None or 'per_horizon' not in metrics_data:
                continue
            
            if metric not in metrics_data['per_horizon']:
                continue
            
            horizon_values = metrics_data['per_horizon'][metric]
            horizons = range(1, len(horizon_values) + 1)
            
            plt.plot(horizons, horizon_values, marker='o', 
                     label=get_experiment_label(exp_dir, dataset_name))
        
        plt.title(f'{metric.upper()} by Forecast Horizon')
        plt.xlabel('Horizon')
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize='small')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'per_horizon_{metric}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

def plot_per_series_metrics(results_dirs: List[str], metrics: List[str], 
                           figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot per-series metrics distribution using box plots."""
    # Create a flattened list of (dataset_name, results_dir, exp_dir) tuples
    all_experiments = []
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            continue
            
        dataset_name = os.path.basename(results_dir).replace('_', ' ').title()
        exp_dirs = find_experiment_dirs(results_dir)
        
        for exp_dir in exp_dirs:
            all_experiments.append((dataset_name, results_dir, exp_dir))
    
    for metric in metrics:
        plt.figure(figsize=tuple(figsize))
        
        series_data = []
        labels = []
        
        for dataset_name, results_dir, exp_dir in all_experiments:
            full_path = os.path.join(results_dir, exp_dir)
            metrics_data = load_metrics(full_path)
            
            if metrics_data is None or 'per_series' not in metrics_data:
                continue
            
            if metric not in metrics_data['per_series']:
                continue
            
            series_values = metrics_data['per_series'][metric]
            series_data.append(series_values)
            labels.append(get_experiment_label(exp_dir, dataset_name))
        
        # Create box plot
        plt.boxplot(series_data, labels=labels)
        plt.title(f'{metric.upper()} Distribution Across Series')
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'per_series_{metric}_boxplot.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

def plot_improvement_heatmap(df: pd.DataFrame, metrics: List[str], 
                          figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot heatmap showing improvement between datasets for each experiment type."""
    if 'dataset' not in df.columns or df['dataset'].nunique() <= 1:
        return
    
    datasets = sorted(df['dataset'].unique())
    if len(datasets) != 2:
        return  # This visualization is designed for comparing exactly 2 datasets
    
    base_dataset, compare_dataset = datasets
    
    # Create a mapping from experiment to its type
    exp_to_type = {}
    for _, row in df.iterrows():
        exp_type = f"{row['quantization']}_{row['k']}" if row['quantization'] != 'learned' else "learned_policy"
        exp_to_type[row['experiment']] = exp_type
    
    # Get unique experiment types
    exp_types = sorted(set(exp_to_type.values()))
    
    for metric in metrics:
        # Create a dataframe for the heatmap
        improvements = {}
        
        for exp_type in exp_types:
            # Find matching experiments in both datasets
            base_rows = df[(df['dataset'] == base_dataset) & (df['experiment'].apply(lambda x: exp_to_type.get(x) == exp_type))]
            compare_rows = df[(df['dataset'] == compare_dataset) & (df['experiment'].apply(lambda x: exp_to_type.get(x) == exp_type))]
            
            if len(base_rows) == 0 or len(compare_rows) == 0:
                continue
                
            base_value = base_rows[metric].values[0]
            compare_value = compare_rows[metric].values[0]
            
            # Calculate relative improvement (negative is better for these metrics)
            improvement = (base_value - compare_value) / base_value * 100
            improvements[exp_type] = improvement
        
        if not improvements:
            continue
            
        # Create the heatmap
        plt.figure(figsize=tuple(figsize))
        
        # Convert to dataframe for seaborn
        improvement_df = pd.DataFrame({
            'Experiment Type': list(improvements.keys()),
            'Improvement (%)': list(improvements.values())
        })
        
        # Reshape for heatmap
        heatmap_data = improvement_df.set_index('Experiment Type')['Improvement (%)'].to_frame().T
        
        # Plot heatmap
        ax = sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.1f', 
                         center=0, cbar_kws={'label': 'Improvement (%)'})
        
        plt.title(f'{metric.upper()} Improvement: {compare_dataset} vs {base_dataset}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'improvement_{metric}_heatmap.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

def plot_config_comparison(df: pd.DataFrame, metrics: List[str], 
                          figsize: Tuple[int, int], output_dir: str, dpi: int = 300):
    """Plot metrics grouped by configuration parameters."""
    if 'quantization' in df.columns and 'k' in df.columns:
        # Group by quantization type and k value
        plt.figure(figsize=tuple(figsize))
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i+1)
            
            # Create grouped bar plot
            grouped_df = df.pivot_table(
                index=['quantization', 'dataset'], 
                columns='k', 
                values=metric,
                aggfunc='mean'
            )
            
            grouped_df.plot(kind='bar', ax=plt.gca())
            
            plt.title(f'{metric.upper()} by Configuration')
            plt.ylabel(metric.upper())
            plt.legend(title='K Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'config_comparison.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics from all experiments
    df = load_all_metrics(args.results_dirs)
    
    if df.empty:
        print(f"No metrics found in {args.results_dirs}")
        return
    
    # Plot overall metrics
    plot_overall_metrics(df, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    # Plot dataset comparison
    plot_dataset_comparison(df, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    # Plot per-horizon metrics
    plot_per_horizon_metrics(args.results_dirs, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    # Plot per-series metrics
    plot_per_series_metrics(args.results_dirs, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    # Plot improvement heatmap
    plot_improvement_heatmap(df, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    # Plot configuration comparison
    plot_config_comparison(df, args.metrics, args.figsize, args.output_dir, args.dpi)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 