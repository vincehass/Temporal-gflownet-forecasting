#!/usr/bin/env python
"""
Script to run multiple experiments with full Weights & Biases integration.

This script automates running multiple ablation studies and tracks all results,
metrics, and visualizations in Weights & Biases.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import shutil
import yaml
import wandb
from tqdm import tqdm

def setup_experiment_dir(base_dir, experiment_name):
    """
    Set up experiment directory structure.
    
    Args:
        base_dir: Base results directory
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with paths to experiment directories
    """
    exp_dir = os.path.join(base_dir, experiment_name)
    paths = {
        'exp_dir': exp_dir,
        'logs_dir': os.path.join(exp_dir, 'logs'),
        'checkpoints_dir': os.path.join(exp_dir, 'checkpoints'),
        'evaluation_dir': os.path.join(exp_dir, 'evaluation'),
        'plots_dir': os.path.join(exp_dir, 'plots')
    }
    
    # Create all directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

def run_experiment(exp_name, cmd_overrides, args):
    """
    Run a single experiment with the given configuration.
    
    Args:
        exp_name: Name of the experiment
        cmd_overrides: List of command line overrides for Hydra
        args: Command line arguments
        
    Returns:
        Boolean indicating success/failure
    """
    print(f"\n{'='*80}")
    print(f"Starting experiment: {exp_name}")
    print(f"{'='*80}")
    
    # Set up experiment directories
    exp_paths = setup_experiment_dir(args.results_dir, exp_name)
    
    # Construct the training command with hydra overrides
    train_cmd = [
        "python", "scripts/train.py",
        f"results_dir={exp_paths['exp_dir']}",
        f"logging.results_dir={exp_paths['exp_dir']}",
    ]
    
    # Add all overrides
    train_cmd.extend(cmd_overrides)
    
    # Add wandb args if enabled
    if args.use_wandb:
        train_cmd.extend([
            "use_wandb=true",
            f"wandb_project={args.wandb_project}",
            f"wandb_entity={args.wandb_entity}",
            f"wandb_name={exp_name}"
        ])
        if args.offline:
            train_cmd.append("wandb_mode=offline")
    
    # Run the training process
    print(f"Running training for experiment: {exp_name}")
    print(f"Command: {' '.join(train_cmd)}")
    
    try:
        train_process = subprocess.run(
            train_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(train_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        print(e.stdout)
        return False
        
    # Construct the evaluation command
    best_checkpoint = os.path.join(exp_paths['checkpoints_dir'], 'best_model.pt')
    config_path = os.path.join(exp_paths['exp_dir'], 'config.yaml')
    
    # Only run evaluation if both checkpoint and config exist
    if os.path.exists(best_checkpoint) and os.path.exists(config_path):
        eval_cmd = [
            "python", "scripts/evaluate.py",
            "--checkpoint_path", best_checkpoint,
            "--config_path", config_path,
            "--output_dir", exp_paths['evaluation_dir']
        ]
        
        # Run the evaluation process
        print(f"Running evaluation for experiment: {exp_name}")
        print(f"Command: {' '.join(eval_cmd)}")
        
        try:
            eval_process = subprocess.run(
                eval_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            print(eval_process.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed with exit code {e.returncode}")
            print(e.stdout)
            return False
    else:
        print(f"Skipping evaluation: checkpoint or config not found")
        return False
    
    print(f"Experiment {exp_name} completed successfully!")
    return True

def run_all_experiments(args):
    """
    Run all specified experiments.
    
    Args:
        args: Command line arguments
    """
    # Initialize wandb for the experiment group if specified
    if args.use_wandb and args.group_experiments:
        try:
            if args.offline:
                os.environ["WANDB_MODE"] = "offline"
                print("Running wandb in offline mode")
                
            wandb.init(
                project=args.wandb_project,
                entity=None if args.offline else args.wandb_entity,
                name=f"experiment_group_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "results_dir": args.results_dir,
                    "experiments": args.experiments
                }
            )
        except Exception as e:
            print(f"Warning: Failed to initialize W&B for experiment group: {e}")
            args.use_wandb = False
    
    # Define experiment configurations as Hydra command line overrides
    experiment_configs = {}
    
    # Adaptive quantization experiments with different k values
    for k in args.k_values:
        exp_name = f"adaptive_k{k}"
        if exp_name in args.experiments:
            experiment_configs[exp_name] = [
                "quantization=adaptive_config",
                f"quantization.k_initial={k}",
                f"quantization.k_max={2*k}",
                f"training.epochs={args.epochs}",
                f"training.batch_size={args.batch_size}",
                "dataset=synthetic_config"
            ]
    
    # Fixed quantization experiments with different k values
    for k in args.k_values:
        exp_name = f"fixed_k{k}"
        if exp_name in args.experiments:
            experiment_configs[exp_name] = [
                "quantization=fixed_config",
                f"quantization.k_initial={k}",
                f"training.epochs={args.epochs}",
                f"training.batch_size={args.batch_size}",
                "dataset=synthetic_config"
            ]
    
    # Learned policy experiment (if requested)
    if "learned_policy" in args.experiments:
        experiment_configs["learned_policy"] = [
            "quantization=adaptive_config",
            "quantization.k_initial=10",
            "quantization.k_max=50",
            "policy=learned_config",
            f"training.epochs={args.epochs}",
            f"training.batch_size={args.batch_size}",
            "dataset=synthetic_config"
        ]
    
    # Run all experiments
    results = {}
    for exp_name, overrides in tqdm(experiment_configs.items(), desc="Running experiments"):
        success = run_experiment(exp_name, overrides, args)
        results[exp_name] = "Success" if success else "Failed"
    
    # Generate comparative plots using the plot_ablation_results.py script
    if args.plot_results:
        print("\nGenerating comparative plots...")
        plot_cmd = [
            "python", "scripts/plot_ablation_results.py",
            "--results_dir", args.results_dir,
            "--output_dir", os.path.join(args.results_dir, "plots"),
            "--experiments"
        ] + list(experiment_configs.keys())
        
        if args.use_wandb:
            plot_cmd.append("--use_wandb")
            plot_cmd.extend([
                "--wandb_project", args.wandb_project,
                "--wandb_entity", args.wandb_entity,
                "--wandb_name", f"ablation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ])
            if args.offline:
                plot_cmd.append("--offline")
        
        print(f"Running command: {' '.join(plot_cmd)}")
        try:
            subprocess.run(plot_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Plotting failed with exit code {e.returncode}")
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary:")
    print("="*80)
    for exp_name, status in results.items():
        print(f"{exp_name}: {status}")
    
    # Finish wandb run for experiment group
    if args.use_wandb and args.group_experiments:
        # Log summary information
        summary_table = wandb.Table(columns=["Experiment", "Status"])
        for exp_name, status in results.items():
            summary_table.add_data(exp_name, status)
        
        wandb.log({"experiment_summary": summary_table})
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments with Weights & Biases integration')
    
    # Experiment selection
    parser.add_argument('--experiments', type=str, nargs='+', 
                        default=['adaptive_k10', 'adaptive_k20', 'fixed_k10', 'fixed_k20', 'learned_policy'],
                        help='Experiments to run')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values to test for fixed and adaptive quantization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Output directories
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Base directory for saving results')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    parser.add_argument('--offline', action='store_true',
                        help='Run W&B in offline mode')
    parser.add_argument('--group_experiments', action='store_true',
                        help='Group all experiments under a single W&B run')
    
    # Additional options
    parser.add_argument('--plot_results', action='store_true', default=True,
                        help='Generate comparative plots after experiments complete')
    
    args = parser.parse_args()
    
    # Run all experiments
    run_all_experiments(args)

if __name__ == "__main__":
    main() 