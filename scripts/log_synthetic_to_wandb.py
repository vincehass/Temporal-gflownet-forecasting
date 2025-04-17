#!/usr/bin/env python
"""
Script to log synthetic ablation study results to Weights & Biases.
"""
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, Any

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
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics from {metrics_path}: {e}")
        return None

def load_training_curves(experiment_dir: str) -> Dict[str, Any]:
    """
    Load training curves from experiment results.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing training curves
    """
    curves_path = os.path.join(experiment_dir, 'logs', 'training_curves.json')
    if not os.path.exists(curves_path):
        print(f"Warning: Training curves file not found at {curves_path}")
        return None
    
    try:
        with open(curves_path, 'r') as f:
            curves = json.load(f)
        return curves
    except Exception as e:
        print(f"Error loading training curves from {curves_path}: {e}")
        return None

def load_config(experiment_dir: str) -> Dict[str, Any]:
    """
    Load configuration from experiment results.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing configuration
    """
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}

def create_plots(metrics, curves, exp_name: str) -> Dict[str, Any]:
    """
    Create plots for the experiment.
    
    Args:
        metrics: Metrics data
        curves: Training curves data
        exp_name: Experiment name
        
    Returns:
        Dictionary of plot objects
    """
    plots = {}
    
    # Plot training loss
    if curves and 'loss' in curves:
        plt.figure(figsize=(10, 6))
        plt.plot(curves.get('epochs', range(1, len(curves['loss']) + 1)), curves['loss'])
        plt.title(f'Training Loss ({exp_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plots['training_loss'] = plt.gcf()
        plt.close()
    
    # Plot k values evolution
    if curves and 'k' in curves:
        plt.figure(figsize=(10, 6))
        plt.plot(curves.get('epochs', range(1, len(curves['k']) + 1)), curves['k'])
        plt.title(f'Quantization Bins Evolution ({exp_name})')
        plt.xlabel('Epoch')
        plt.ylabel('K')
        plt.grid(True)
        plots['k_evolution'] = plt.gcf()
        plt.close()
    
    # Plot metrics by horizon
    if metrics and 'per_horizon' in metrics:
        metrics_list = ['wql', 'crps', 'mase']
        
        for metric in metrics_list:
            if metric in metrics['per_horizon']:
                plt.figure(figsize=(10, 6))
                values = metrics['per_horizon'][metric]
                horizons = range(1, len(values) + 1)
                plt.plot(horizons, values, marker='o')
                plt.title(f'{metric.upper()} by Forecast Horizon ({exp_name})')
                plt.xlabel('Horizon')
                plt.ylabel(f'{metric.upper()} Value')
                plt.grid(True)
                plots[f'per_horizon_{metric}'] = plt.gcf()
                plt.close()
    
    return plots

def log_experiment_to_wandb(experiment_dir: str, exp_name: str = None):
    """
    Log experiment results to W&B.
    
    Args:
        experiment_dir: Path to experiment directory
        exp_name: Optional experiment name override
    """
    if exp_name is None:
        exp_name = os.path.basename(experiment_dir)
    
    print(f"Loading data for experiment: {exp_name}")
    
    # Load data
    metrics = load_metrics(experiment_dir)
    curves = load_training_curves(experiment_dir)
    config = load_config(experiment_dir)
    
    # Initialize W&B run
    run = wandb.init(
        project="temporal-gfn-eeg",
        name=exp_name,
        config=config,
        reinit=True
    )
    
    # Log config
    if config:
        print(f"Logging configuration...")
        wandb.config.update(config)
    
    # Log overall metrics
    if metrics and 'overall' in metrics:
        print(f"Logging overall metrics...")
        wandb.log(metrics['overall'])
    
    # Log training curves
    if curves:
        print(f"Logging training curves...")
        steps = curves.get('epochs', list(range(1, len(curves.get('loss', [])) + 1)))
        
        for i, step in enumerate(steps):
            step_metrics = {}
            for key in curves:
                if key != 'epochs' and i < len(curves[key]):
                    step_metrics[key] = curves[key][i]
            
            wandb.log(step_metrics, step=step)
    
    # Create and log plots
    if metrics or curves:
        print(f"Creating and logging plots...")
        plots = create_plots(metrics, curves, exp_name)
        
        for name, plot in plots.items():
            wandb.log({name: wandb.Image(plot)})
    
    # Create per horizon table
    if metrics and 'per_horizon' in metrics:
        print(f"Creating per-horizon tables...")
        horizon_metrics = []
        
        for h, horizon in enumerate(range(1, len(metrics['per_horizon'].get('wql', [])) + 1)):
            row = {'horizon': horizon}
            for metric in metrics['per_horizon']:
                if h < len(metrics['per_horizon'][metric]):
                    row[metric] = metrics['per_horizon'][metric][h]
            horizon_metrics.append(row)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(horizon_metrics)
        wandb.log({"per_horizon_metrics": wandb.Table(dataframe=df)})
    
    wandb.finish()
    print(f"Experiment {exp_name} logged to W&B")

def main():
    """Main function to log synthetic ablation studies to W&B."""
    parser = argparse.ArgumentParser(description='Log synthetic ablation studies to W&B')
    parser.add_argument('--results_dir', type=str, default='./results/synthetic_data_full',
                       help='Directory containing experiment results')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment to log (if not provided, will log all experiments)')
    parser.add_argument('--project', type=str, default='temporal-gfn-eeg',
                       help='W&B project name')
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist.")
        return
    
    # Get experiments to log
    if args.experiment:
        experiment_dirs = [os.path.join(args.results_dir, args.experiment)]
        if not os.path.exists(experiment_dirs[0]):
            print(f"Error: Experiment directory {experiment_dirs[0]} does not exist.")
            return
    else:
        experiment_dirs = [os.path.join(args.results_dir, d) for d in os.listdir(args.results_dir)
                          if os.path.isdir(os.path.join(args.results_dir, d)) and d != 'plots']
    
    print(f"Found {len(experiment_dirs)} experiments to log")
    
    # Log each experiment to W&B
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        log_experiment_to_wandb(exp_dir, exp_name)

if __name__ == "__main__":
    main() 