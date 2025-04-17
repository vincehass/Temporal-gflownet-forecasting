#!/usr/bin/env python3
"""
Script to log existing experiment results to Weights & Biases.
This is useful for experiments that were run without W&B integration.
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import glob
import wandb
from typing import Dict, Any, List, Optional
import re


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Log experiment results to W&B')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                      help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                      help='W&B entity name')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                      help='Tags to add to the W&B run')
    parser.add_argument('--datasets', type=str, nargs='+', 
                      default=['electricity', 'traffic', 'ETTm1', 'ETTh1'],
                      help='Datasets to include')
    
    return parser.parse_args()


def load_metrics(experiment_dir: str, dataset: Optional[str] = None) -> Dict[str, Any]:
    """Load metrics for a specific experiment and dataset."""
    # First try dataset-specific metrics
    if dataset:
        metrics_file = os.path.join(experiment_dir, 'evaluation', f'{dataset}_metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
    
    # If dataset-specific metrics not found or dataset is None, try general metrics.json
    metrics_file = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # If no metrics found, return empty dict
    return {}


def load_config(experiment_dir: str) -> Dict[str, Any]:
    """Load configuration for an experiment."""
    config_file = os.path.join(experiment_dir, 'config.yaml')
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}


def load_logs(experiment_dir: str) -> Dict[str, List[float]]:
    """Load training logs from the logs directory."""
    logs = {}
    logs_dir = os.path.join(experiment_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        return logs
    
    # Try to find all log files
    log_files = glob.glob(os.path.join(logs_dir, '*.json'))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_name = os.path.basename(log_file).replace('.json', '')
                logs[log_name] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading log file {log_file}: {e}")
    
    return logs


def extract_experiment_params(experiment_dir: str) -> Dict[str, str]:
    """Extract parameters from experiment directory name."""
    base_name = os.path.basename(experiment_dir)
    params = {}
    
    # Extract quantization type
    if 'fixed' in base_name:
        params['quantization'] = 'fixed'
    elif 'adaptive' in base_name:
        params['quantization'] = 'adaptive'
    
    # Extract k value
    k_match = re.search(r'k(\d+)', base_name)
    if k_match:
        params['k'] = k_match.group(1)
    
    # Extract policy type
    if 'uniform' in base_name:
        params['policy_type'] = 'uniform'
    elif 'learned' in base_name:
        params['policy_type'] = 'learned'
    
    # Extract entropy bonus
    entropy_match = re.search(r'entropy_(\d+\.\d+)', base_name)
    if entropy_match:
        params['entropy_bonus'] = entropy_match.group(1)
    
    return params


def get_run_name(experiment_dir: str, config: Dict[str, Any]) -> str:
    """Generate a descriptive name for the W&B run."""
    base_name = os.path.basename(experiment_dir)
    
    # Try to extract dataset name from config
    dataset = ""
    if config and 'dataset' in config:
        if isinstance(config['dataset'], dict) and 'name' in config['dataset']:
            dataset = config['dataset']['name']
        elif isinstance(config['dataset'], str):
            dataset = config['dataset']
    
    if dataset:
        return f"{dataset}_{base_name}"
    else:
        return base_name


def log_experiment_to_wandb(args: argparse.Namespace) -> None:
    """Log experiment results to W&B."""
    experiment_dir = args.experiment_dir
    
    # Verify the experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return
    
    # Load experiment config
    config = load_config(experiment_dir)
    if not config:
        print(f"Warning: Could not load config from {experiment_dir}")
    
    # Extract experiment parameters
    params = extract_experiment_params(experiment_dir)
    
    # Merge config with extracted parameters
    for key, value in params.items():
        if key not in config:
            config[key] = value
    
    # Generate run name
    run_name = get_run_name(experiment_dir, config)
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=args.tags,
        job_type="experiment",
    )
    
    try:
        # Log metrics for each dataset
        for dataset in args.datasets:
            metrics = load_metrics(experiment_dir, dataset)
            if metrics:
                # Log metrics with dataset prefix
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        # Handle nested metrics (like per-quantile WQL)
                        for sub_name, sub_value in metric_value.items():
                            wandb.log({f"{dataset}/{metric_name}/{sub_name}": sub_value})
                    else:
                        wandb.log({f"{dataset}/{metric_name}": metric_value})
                
                print(f"Logged metrics for dataset {dataset}")
            else:
                print(f"No metrics found for dataset {dataset}")
        
        # Load and log training logs
        logs = load_logs(experiment_dir)
        if logs:
            for log_name, log_values in logs.items():
                # If log_values is a list, log as a sequence
                if isinstance(log_values, list):
                    for i, value in enumerate(log_values):
                        wandb.log({log_name: value, "step": i})
                elif isinstance(log_values, dict):
                    # If log_values is a dict, log each key-value pair
                    for key, values in log_values.items():
                        if isinstance(values, list):
                            for i, value in enumerate(values):
                                wandb.log({f"{log_name}/{key}": value, "step": i})
                        else:
                            wandb.log({f"{log_name}/{key}": values})
            
            print(f"Logged training logs")
        else:
            print("No training logs found")
        
        # Create artifacts for important files
        artifact = wandb.Artifact(name=f"experiment_data_{run_name}", type="experiment")
        
        # Add config file
        config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(config_path):
            artifact.add_file(config_path, name="config.yaml")
        
        # Add metrics files
        metrics_files = glob.glob(os.path.join(experiment_dir, "evaluation", "*.json"))
        for metrics_file in metrics_files:
            artifact.add_file(metrics_file, name=f"metrics/{os.path.basename(metrics_file)}")
        
        # Log the artifact
        wandb.log_artifact(artifact)
        
        print(f"Logged experiment data as artifact")
        
    finally:
        # Finish the run
        wandb.finish()
    
    print(f"Successfully logged experiment to W&B: {run_name}")
    print(f"View the run at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/{wandb.run.id}")


if __name__ == "__main__":
    args = parse_arguments()
    log_experiment_to_wandb(args) 