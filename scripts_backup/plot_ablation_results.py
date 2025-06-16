#!/usr/bin/env python3
"""
Script for visualizing results from ablation studies.
This script loads metrics from different experiments, creates comparison plots,
and visualizes the results using Weights & Biases.
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
import wandb
from typing import Dict, List, Tuple, Optional, Union, Any

# Remove circular import - these functions will be defined in this file
# from plot_ablation_results import (plot_ste_visualization, 
#                                  plot_tb_loss_visualization,
#                                  plot_quantization_range_visualization)

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
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    parser.add_argument('--create_report', action='store_true',
                        help='Create a W&B report summarizing the ablation study')
    
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

def get_wandb_runs(experiment_dirs: List[str], wandb_entity: str, wandb_project: str) -> Dict[str, Any]:
    """Get W&B runs for the specified experiments."""
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    
    # Map experiment directories to W&B runs
    run_map = {}
    for experiment_dir in experiment_dirs:
        exp_name = os.path.basename(experiment_dir)
        # Find matching runs
        matching_runs = [run for run in runs if exp_name in run.name]
        if matching_runs:
            run_map[experiment_dir] = matching_runs[0]
    
    return run_map

def create_wandb_plots(metrics_df: pd.DataFrame, run_map: Dict[str, Any], 
                      metrics: List[str], datasets: List[str], 
                      wandb_entity: str, wandb_project: str,
                      study_type: str, output_dir: str):
    """Create plots in W&B for visualization."""
    # Initialize W&B
    wandb.init(project=wandb_project, entity=wandb_entity, name=f"ablation_analysis_{study_type}", 
              job_type="analysis", config={"study_type": study_type})
    
    try:
        # Log the metrics dataframe
        wandb.run.log({"metrics_table": wandb.Table(dataframe=metrics_df)})
        
        # Create plots for each metric
        for metric in metrics:
            if metric not in metrics_df.columns:
                continue
            
            # Create and log overall metrics comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            valid_df = metrics_df[metrics_df[metric].notna()]
            
            if valid_df.empty:
                continue
                
            # Calculate average metrics by experiment
            grouped = valid_df.groupby(['experiment_label'])[metric].mean().reset_index()
            # Sort by value
            grouped = grouped.sort_values(by=metric)
            
            # Create bar plot
            sns.barplot(x='experiment_label', y=metric, data=grouped, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.title(f'Average {metric.upper()} Across Experiments')
            
            # Log to W&B
            wandb.log({f"overall_{metric}": wandb.Image(fig)})
            plt.close(fig)
            
            # Create and log dataset comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped_by_dataset = valid_df.groupby(['experiment_label', 'dataset'])[metric].mean().reset_index()
            
            # Create grouped bar plot
            sns.barplot(x='dataset', y=metric, hue='experiment_label', data=grouped_by_dataset, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.title(f'{metric.upper()} by Dataset')
            
            # Log to W&B
            wandb.log({f"dataset_comparison_{metric}": wandb.Image(fig)})
            plt.close(fig)
        
        # Log special visualizations for quantization studies
        if study_type in ['all', 'quantization']:
            # Log quantization-specific visualizations
            plot_quantization_metrics_wandb(run_map, datasets)
            
            # Generate STE visualizations
            plot_ste_visualization(output_dir)
            # Upload STE visualizations to W&B
            for img_file in glob.glob(os.path.join(output_dir, 'ste_visualizations', '*.png')):
                wandb.log({os.path.basename(img_file): wandb.Image(img_file)})
            
            # Generate TB loss visualizations
            plot_tb_loss_visualization(output_dir)
            # Upload TB loss visualizations to W&B
            for img_file in glob.glob(os.path.join(output_dir, 'tb_loss_visualizations', '*.png')):
                wandb.log({os.path.basename(img_file): wandb.Image(img_file)})
            
            # Generate quantization range visualizations
            plot_quantization_range_visualization(output_dir)
            # Upload quantization range visualizations to W&B
            for img_file in glob.glob(os.path.join(output_dir, 'quantization_range_visualizations', '*.png')):
                wandb.log({os.path.basename(img_file): wandb.Image(img_file)})
        
        # Create and log a summary report if requested
        if args.create_report:
            create_wandb_report(metrics_df, study_type, wandb_entity, wandb_project)
    
    finally:
        # Finish the W&B run
        wandb.finish()

def plot_quantization_metrics_wandb(run_map: Dict[str, Any], datasets: List[str]):
    """Plot quantization-specific metrics using W&B run data."""
    # Extract quantization metrics from W&B
    quantization_metrics = [
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
    
    for metric in quantization_metrics:
        # Create a Vega-Lite spec for a line chart
        vega_spec = {
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": "step", "type": "quantitative", "title": "Training Step"},
                "y": {"field": "value", "type": "quantitative", "title": metric},
                "color": {"field": "run", "type": "nominal", "title": "Experiment"}
            }
        }
        
        # Log the chart to W&B
        wandb.log({f"plot_{metric}": wandb.plots.vega_lite.WandbVegaLite(vega_spec, {"values": []})})

def create_wandb_report(metrics_df: pd.DataFrame, study_type: str, wandb_entity: str, wandb_project: str):
    """Create a W&B report summarizing the ablation study."""
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    
    # Create a report
    report = wandb.Report(
        title=f"Ablation Study: {study_type.capitalize()}",
        description=f"Analysis of ablation experiments for {study_type}"
    )
    
    # Add sections to the report
    report.add(
        wandb.visualize.CustomChart({
            "panel_type": "Vega2",
            "vega_spec": {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "experiment_label", "type": "nominal"},
                    "y": {"field": "wql", "type": "quantitative"},
                    "color": {"field": "experiment_label", "type": "nominal"}
                }
            },
            "data": metrics_df[['experiment_label', 'wql']].to_dict('records')
        }),
        name="WQL Comparison"
    )
    
    # Add visualizations for other metrics
    for metric in ['crps', 'mase']:
        if metric in metrics_df.columns:
            report.add(
                wandb.visualize.CustomChart({
                    "panel_type": "Vega2",
                    "vega_spec": {
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "experiment_label", "type": "nominal"},
                            "y": {"field": metric, "type": "quantitative"},
                            "color": {"field": "experiment_label", "type": "nominal"}
                        }
                    },
                    "data": metrics_df[['experiment_label', metric]].to_dict('records')
                }),
                name=f"{metric.upper()} Comparison"
            )
    
    # Save the report
    report.save()

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
    
    # Get W&B runs
    run_map = get_wandb_runs(experiment_dirs, args.wandb_entity, args.wandb_project)
    
    # Create W&B visualizations
    create_wandb_plots(metrics_df, run_map, args.metrics, args.datasets, 
                      args.wandb_entity, args.wandb_project,
                      args.study_type, args.output_dir)
    
    print(f"W&B visualizations created. View them at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")

if __name__ == "__main__":
    main() 