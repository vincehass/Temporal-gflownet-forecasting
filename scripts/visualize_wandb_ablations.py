#!/usr/bin/env python3
"""
Script for visualizing ablation study results using Weights & Biases (W&B).
This script creates comprehensive W&B dashboards, reports, and visualizations
for comparing different configurations in ablation studies.
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
import re
import wandb
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml

# Set plot style for local visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
COLORS = sns.color_palette("Set2", 10)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create W&B visualizations for ablation studies')
    parser.add_argument('--results_dir', type=str, default='./results/ablations',
                        help='Directory containing ablation study results')
    parser.add_argument('--output_dir', type=str, default='./results/ablation_plots',
                        help='Directory to save local plots (if any)')
    parser.add_argument('--study_type', type=str, default='all',
                        help='Type of ablation study to plot (e.g., "quantization", "entropy", "all")')
    parser.add_argument('--datasets', type=str, nargs='+', default=['electricity', 'traffic', 'ETTm1', 'ETTh1'],
                        help='Datasets to include in plots')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to plot')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str, study_type: str = 'all') -> List[str]:
    """Find all experiment directories for the given study type."""
    if study_type == 'all':
        pattern = os.path.join(results_dir, '*')
    else:
        pattern = os.path.join(results_dir, f'*{study_type}*')
    
    # Filter to only include directories
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return sorted(dirs)

def load_metrics_file(metrics_file: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        return {}

def load_metrics(experiment_dir: str, dataset: str = None) -> Dict[str, Any]:
    """Load metrics for a specific experiment and dataset."""
    # First try dataset-specific metrics
    if dataset:
        metrics_file = os.path.join(experiment_dir, 'evaluation', f'{dataset}_metrics.json')
        metrics = load_metrics_file(metrics_file)
        if metrics:
            return metrics
    
    # If that fails or dataset is None, try general metrics.json
    metrics_file = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    return load_metrics_file(metrics_file)

def load_config(experiment_dir: str) -> Dict[str, Any]:
    """Load configuration for an experiment."""
    config_file = os.path.join(experiment_dir, 'config.yaml')
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}

def extract_experiment_params(experiment_dir: str, config: Dict[str, Any] = None) -> Dict[str, str]:
    """Extract parameters from experiment directory name and config."""
    base_name = os.path.basename(experiment_dir)
    params = {}
    
    # Extract from directory name first
    # Quantization type and parameters
    quant_match = re.search(r'(?:fixed|adaptive)_k(\d+)', base_name)
    if quant_match:
        params['quantization'] = 'fixed' if 'fixed' in base_name else 'adaptive'
        params['k_value'] = quant_match.group(1)
    
    # Policy type
    if 'uniform' in base_name:
        params['policy'] = 'uniform'
    elif 'learned' in base_name:
        params['policy'] = 'learned'
    
    # Extract from config if available
    if config:
        # Get dataset info
        if 'dataset' in config:
            if isinstance(config['dataset'], dict):
                params['dataset'] = config['dataset'].get('name', 'unknown')
            else:
                params['dataset'] = config['dataset']
        
        # Get quantization info
        if 'quantization' in config:
            quant_config = config['quantization']
            if 'adaptive' in quant_config:
                params['quantization'] = 'adaptive' if quant_config['adaptive'] else 'fixed'
            if 'k_initial' in quant_config:
                params['k_value'] = str(quant_config['k_initial'])
        
        # Get policy info
        if 'policy' in config and 'backward_policy_type' in config['policy']:
            params['policy'] = config['policy']['backward_policy_type']
        
        # Get entropy value
        if 'gfn' in config and 'lambda_entropy' in config['gfn']:
            params['entropy'] = str(config['gfn']['lambda_entropy'])
    
    # If nothing found, use the directory name
    if not params:
        params['experiment'] = base_name
    
    return params

def get_experiment_label(params: Dict[str, str]) -> str:
    """Generate a readable label for the experiment based on its parameters."""
    parts = []
    
    if 'quantization' in params:
        parts.append(f"{params['quantization'].capitalize()}")
        if 'k_value' in params:
            parts.append(f"k={params['k_value']}")
    
    if 'policy' in params:
        parts.append(f"Policy={params['policy']}")
    
    if 'entropy' in params:
        parts.append(f"λ={params['entropy']}")
    
    if parts:
        return " ".join(parts)
    elif 'experiment' in params:
        # Clean up the experiment name
        return params['experiment'].replace('_', ' ').title()
    else:
        return "Unknown"

def load_all_metrics(experiment_dirs: List[str], datasets: List[str]) -> pd.DataFrame:
    """Load metrics from all experiments and datasets into a DataFrame."""
    rows = []
    
    for experiment_dir in experiment_dirs:
        # Load config and extract parameters
        config = load_config(experiment_dir)
        params = extract_experiment_params(experiment_dir, config)
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

def get_wandb_runs(experiment_dirs: List[str], wandb_entity: str, wandb_project: str) -> Dict[str, Any]:
    """Get W&B runs for the specified experiments."""
    api = wandb.Api()
    
    try:
        # Get all runs from the project
        all_runs = api.runs(f"{wandb_entity}/{wandb_project}")
        
        # Map experiment directories to W&B runs
        run_map = {}
        for experiment_dir in experiment_dirs:
            exp_name = os.path.basename(experiment_dir)
            # Find matching runs
            matching_runs = [run for run in all_runs if exp_name in run.name]
            if matching_runs:
                run_map[experiment_dir] = matching_runs[0]
        
        return run_map
    except Exception as e:
        print(f"Error accessing W&B API: {e}")
        return {}

def create_quantization_visualizations():
    """Create static visualizations for quantization mechanisms."""
    # 1. Visualize bin distribution for different K values
    k_values = [5, 10, 20]
    vmin, vmax = -10, 10
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5))
    for i, k in enumerate(k_values):
        ax = axes[i]
        bin_width = (vmax - vmin) / k
        bin_centers = np.linspace(vmin + bin_width/2, vmax - bin_width/2, k)
        
        # Create a synthetic distribution (example data)
        x = np.linspace(vmin, vmax, 1000)
        y = np.exp(-0.5 * ((x - 0) / 3) ** 2)
        
        # Plot original distribution
        ax.plot(x, y, 'k-', alpha=0.5, label='Original Distribution')
        
        # Plot quantized bins
        quantized = np.floor((x - vmin) / bin_width) * bin_width + vmin + bin_width/2
        ax.step(x, np.interp(quantized, x, y), 'r-', where='mid', label='Quantized', alpha=0.7)
        
        # Mark bin centers
        for j in range(k):
            ax.axvline(bin_centers[j], color='blue', alpha=0.3, linestyle='--')
        
        ax.set_title(f'K={k}, Bin Width={bin_width:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    return fig

def create_adaptive_quantization_visualization():
    """Create visualization of adaptive quantization decision process."""
    # Create a grid of values for delta_R and H
    delta_R = np.linspace(0, 0.05, 100)  # Reward improvement
    H = np.linspace(0, 1, 100)  # Normalized entropy
    
    # Create meshgrid
    Delta_R, H_mesh = np.meshgrid(delta_R, H)
    
    # Set adaptation parameters
    epsilon = 0.02  # Reward improvement threshold
    lambda_adapt = 0.5  # Adaptation sensitivity
    
    # Calculate the adaptive update factor: η_e = 1 + λ * (max(0, ε - ΔR_e)/ε + (1 - H_e))
    improvement_signal = np.maximum(0, epsilon - Delta_R) / epsilon
    confidence_signal = 1 - H_mesh
    adaptive_factor = 1 + lambda_adapt * (improvement_signal + confidence_signal)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(Delta_R, H_mesh, adaptive_factor, 20, cmap='viridis')
    plt.colorbar(contour, label='Adaptive Update Factor (η_e)')
    
    # Draw the decision boundary for η_e = 1 (no change in K)
    ax.contour(Delta_R, H_mesh, adaptive_factor, levels=[1], colors='red', linestyles='dashed', linewidths=2)
    
    # Add labels
    ax.set_xlabel('Reward Improvement (ΔR_e)')
    ax.set_ylabel('Normalized Entropy (H_e)')
    ax.set_title('Adaptive Quantization Decision Boundaries')
    
    # Add annotation for decision regions
    ax.text(0.01, 0.2, 'Decrease K\n(η_e < 1)', color='white', fontsize=12, 
           ha='left', va='center', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(0.04, 0.8, 'Increase K\n(η_e > 1)', color='black', fontsize=12, 
           ha='left', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_ste_visualization():
    """Create visualization of the Straight-Through Estimator mechanism."""
    # Define parameters
    k = 10  # Number of bins
    vmin, vmax = -10, 10  # Value range
    
    # Create example logits and probabilities
    logits = np.array([-3, -2, -1, 0, 1, 2, 1, 0, -1, -2])
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Calculate bin centers
    bin_width = (vmax - vmin) / k
    bin_centers = np.linspace(vmin + bin_width/2, vmax - bin_width/2, k)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot forward pass (hard sample)
    ax1.bar(np.arange(k), probs, alpha=0.7)
    ax1.set_title('Forward Pass: Probability Distribution')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Probability')
    
    # Indicate the argmax
    hard_idx = np.argmax(probs)
    ax1.bar([hard_idx], [probs[hard_idx]], color='red', alpha=0.7)
    ax1.text(hard_idx, probs[hard_idx] + 0.02, 'argmax\n(Hard Sample)', 
            ha='center', va='bottom', fontweight='bold')
    
    # Plot backward pass (soft sample)
    soft_sample = np.sum(probs * bin_centers)
    ax2.bar(np.arange(k), probs * bin_centers, alpha=0.7)
    ax2.set_title('Backward Pass: Weighted Values (STE)')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Weighted Value')
    ax2.axhline(y=soft_sample, color='red', linestyle='--')
    ax2.text(k/2, soft_sample + 0.5, f'Expectation: {soft_sample:.2f}\n(Soft Sample for Gradients)', 
            ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Straight-Through Estimator (STE) Mechanism', fontsize=14)
    plt.tight_layout()
    return fig

def create_wandb_report(metrics_df: pd.DataFrame, study_type: str, experiment_dirs: List[str],
                      wandb_entity: str, wandb_project: str):
    """Create a comprehensive W&B report for the ablation study."""
    # Initialize W&B
    wandb.init(project=wandb_project, entity=wandb_entity, 
              name=f"ablation_analysis_{study_type}", 
              job_type="visualization")
    
    try:
        # Create visualizations
        # 1. Create bar charts for each metric
        for metric in ['wql', 'crps', 'mase']:
            if metric in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by experiment label
                grouped = metrics_df.groupby('experiment_label')[metric].mean().reset_index()
                grouped = grouped.sort_values(by=metric)
                
                # Create bar plot
                sns.barplot(x='experiment_label', y=metric, data=grouped, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Average {metric.upper()} by Configuration')
                plt.tight_layout()
                
                # Log to W&B
                wandb.log({f"metric_comparison/{metric}": wandb.Image(fig)})
                plt.close(fig)
        
        # 2. Create dataset comparison plots
        for metric in ['wql', 'crps', 'mase']:
            if metric in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by experiment label and dataset
                grouped = metrics_df.groupby(['experiment_label', 'dataset'])[metric].mean().reset_index()
                
                # Create grouped bar plot
                sns.barplot(x='dataset', y=metric, hue='experiment_label', data=grouped, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'{metric.upper()} by Dataset and Configuration')
                plt.tight_layout()
                
                # Log to W&B
                wandb.log({f"dataset_comparison/{metric}": wandb.Image(fig)})
                plt.close(fig)
        
        # 3. Create special visualizations for quantization analysis
        if study_type in ['all', 'quantization']:
            # Create quantization binning visualization
            fig = create_quantization_visualizations()
            wandb.log({"quantization/bin_distribution": wandb.Image(fig)})
            plt.close(fig)
            
            # Create adaptive quantization visualization
            fig = create_adaptive_quantization_visualization()
            wandb.log({"quantization/adaptive_mechanism": wandb.Image(fig)})
            plt.close(fig)
            
            # Create STE visualization
            fig = create_ste_visualization()
            wandb.log({"ste/mechanism": wandb.Image(fig)})
            plt.close(fig)
        
        # 4. Log the metrics table
        wandb.log({"metrics_table": wandb.Table(dataframe=metrics_df)})
        
        # 5. Create a W&B Artifact with all the result data
        artifact = wandb.Artifact(name=f"ablation_results_{study_type}", type="results")
        
        # Add metrics data
        metrics_path = os.path.join(wandb.run.dir, "metrics.csv")
        metrics_df.to_csv(metrics_path)
        artifact.add_file(metrics_path, name="metrics.csv")
        
        # Add experiment configs
        for exp_dir in experiment_dirs:
            config_path = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_path):
                artifact.add_file(config_path, name=f"configs/{os.path.basename(exp_dir)}_config.yaml")
        
        # Log the artifact
        wandb.log_artifact(artifact)
        
        # Create a report URL
        report_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/reports/Ablation-Study-{study_type.capitalize()}--Vmlldzo0NDU2MA"
        print(f"\nView comprehensive report at: {report_url}")
        
    finally:
        # Finish the W&B run
        wandb.finish()

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Create output directory (for any local plots)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.results_dir, args.study_type)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {args.results_dir} for study type '{args.study_type}'")
        return
        
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Load metrics into DataFrame
    metrics_df = load_all_metrics(experiment_dirs, args.datasets)
    
    if metrics_df.empty:
        print("No metrics found. Check that evaluation has been run.")
        return
        
    print(f"Loaded metrics for {len(metrics_df)} experiment-dataset combinations.")
    
    # Get W&B runs (optional, only used for run-level metrics)
    run_map = get_wandb_runs(experiment_dirs, args.wandb_entity, args.wandb_project)
    
    # Create comprehensive W&B report
    create_wandb_report(metrics_df, args.study_type, experiment_dirs, 
                       args.wandb_entity, args.wandb_project)
    
    print(f"W&B visualizations created. View them at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")

if __name__ == "__main__":
    main() 