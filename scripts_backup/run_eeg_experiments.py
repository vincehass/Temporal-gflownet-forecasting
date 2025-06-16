#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run all EEG dataset experiments with various configurations.
This script bypasses the shell scripts and directly creates and executes
the training commands with the proper Hydra configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EEG experiments')
    parser.add_argument('--results_dir', type=str, default='results/eeg_experiments',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use W&B for logging')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    parser.add_argument('--quantization_types', type=str, default='adaptive,fixed',
                        help='Comma-separated list of quantization types')
    parser.add_argument('--k_values', type=str, default='5,10,20',
                        help='Comma-separated list of k values')
    parser.add_argument('--policy_types', type=str, default='uniform,learned',
                        help='Comma-separated list of policy types')
    parser.add_argument('--entropy_values', type=str, default='0,0.001,0.01,0.1',
                        help='Comma-separated list of entropy values')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all combinations (warning: many experiments)')
    parser.add_argument('--ablation_type', type=str, default='quantization',
                        choices=['quantization', 'policy', 'entropy', 'all'],
                        help='Type of ablation study to run')
    
    return parser.parse_args()

def run_training(config):
    """Run the training with the specified configuration."""
    cmd = [
        "python", "scripts/train.py",
        f"dataset={config['dataset']}_config",
        f"quantization.k_initial={config['k_value']}",
        f"quantization.adaptive={'true' if config['quantization_type'] == 'adaptive' else 'false'}",
        f"policy.backward_policy_type={config['policy_type']}",
        f"training.epochs={config['epochs']}",
        f"training.batch_size={config['batch_size']}",
        f"gfn.lambda_entropy={config['entropy_value']}",
        f"results_dir={config['results_dir']}"
    ]
    
    print(f"\n{'='*70}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    # Create experiment directory and save config
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'logs'), exist_ok=True)
    os.makedirs(os.path.join(config['results_dir'], 'evaluation'), exist_ok=True)
    
    with open(os.path.join(config['results_dir'], 'config.yaml'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Run the command
    result = subprocess.run(cmd, check=False)
    
    return result.returncode == 0

def run_quantization_ablation(args):
    """Run the quantization ablation study."""
    quant_types = args.quantization_types.split(',')
    k_values = [int(k) for k in args.k_values.split(',')]
    
    results = []
    base_config = {
        'dataset': 'eeg',
        'policy_type': 'uniform',
        'entropy_value': 0.01,
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    for quant_type in quant_types:
        for k_value in k_values:
            config = base_config.copy()
            config['quantization_type'] = quant_type
            config['k_value'] = k_value
            config['name'] = f"ablation_quant_{quant_type}_k{k_value}"
            config['results_dir'] = os.path.join(args.results_dir, config['name'])
            
            success = run_training(config)
            results.append({
                'name': config['name'],
                'success': success
            })
    
    return results

def run_policy_ablation(args):
    """Run the policy ablation study."""
    policy_types = args.policy_types.split(',')
    
    results = []
    base_config = {
        'dataset': 'eeg',
        'quantization_type': 'adaptive',
        'k_value': 10,
        'entropy_value': 0.01,
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    for policy_type in policy_types:
        config = base_config.copy()
        config['policy_type'] = policy_type
        config['name'] = f"ablation_policy_{policy_type}"
        config['results_dir'] = os.path.join(args.results_dir, config['name'])
        
        success = run_training(config)
        results.append({
            'name': config['name'],
            'success': success
        })
    
    return results

def run_entropy_ablation(args):
    """Run the entropy ablation study."""
    entropy_values = args.entropy_values.split(',')
    
    results = []
    base_config = {
        'dataset': 'eeg',
        'quantization_type': 'adaptive',
        'k_value': 10,
        'policy_type': 'uniform',
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    for entropy_value in entropy_values:
        config = base_config.copy()
        config['entropy_value'] = float(entropy_value)
        config['name'] = f"ablation_entropy_{entropy_value}"
        config['results_dir'] = os.path.join(args.results_dir, config['name'])
        
        success = run_training(config)
        results.append({
            'name': config['name'],
            'success': success
        })
    
    return results

def run_analysis(args, experiments):
    """Run analysis on the experiment results."""
    # Construct the command for our analysis script
    cmd = [
        "python", "scripts/run_analysis.py",
        f"--results_dir={args.results_dir}",
        f"--output_dir={os.path.join(args.results_dir, 'analysis')}",
        "--dataset=eeg",
        "--plot_style=seaborn-colorblind",
        "--colormap=colorblind"
    ]
    
    print(f"\n{'='*70}")
    print(f"Running analysis")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the analysis
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def main():
    args = parse_arguments()
    
    # Create the results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Track start time
    start_time = time.time()
    
    # Run the requested ablation studies
    results = []
    
    if args.ablation_type == 'quantization' or args.ablation_type == 'all':
        print(f"\n{'='*70}")
        print("Running Quantization Ablation Study")
        print(f"{'='*70}")
        quant_results = run_quantization_ablation(args)
        results.extend(quant_results)
    
    if args.ablation_type == 'policy' or args.ablation_type == 'all':
        print(f"\n{'='*70}")
        print("Running Policy Ablation Study")
        print(f"{'='*70}")
        policy_results = run_policy_ablation(args)
        results.extend(policy_results)
    
    if args.ablation_type == 'entropy' or args.ablation_type == 'all':
        print(f"\n{'='*70}")
        print("Running Entropy Ablation Study")
        print(f"{'='*70}")
        entropy_results = run_entropy_ablation(args)
        results.extend(entropy_results)
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Run analysis
    analysis_success = run_analysis(args, results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Experiment Summary:")
    print(f"{'='*70}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}")
    print(f"Failed experiments: {sum(1 for r in results if not r['success'])}")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nExperiment Results:")
    for result in results:
        status = "Success" if result['success'] else "Failed"
        print(f"  {result['name']}: {status}")
    
    print(f"\nAll experiments {'and analysis ' if analysis_success else ''}completed!")
    print(f"Results saved to: {args.results_dir}")

if __name__ == "__main__":
    main() 