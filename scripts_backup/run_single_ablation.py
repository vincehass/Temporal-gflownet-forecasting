#!/usr/bin/env python3
"""
Run a single model training experiment with W&B logging properly enabled.
This script directly calls train.py with Hydra parameters to ensure W&B works correctly.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single ablation experiment with W&B logging")
    parser.add_argument("--name", type=str, default=f"eeg_ablation_{int(time.time())}",
                        help="Experiment name")
    parser.add_argument("--dataset", type=str, default="eeg",
                        help="Dataset to use")
    parser.add_argument("--quantization", type=str, default="adaptive",
                        choices=["adaptive", "fixed"],
                        help="Quantization type")
    parser.add_argument("--k", type=int, default=10,
                        help="Initial number of quantization bins")
    parser.add_argument("--policy", type=str, default="uniform",
                        choices=["uniform", "learned"],
                        help="Backward policy type")
    parser.add_argument("--entropy", type=float, default=0.01,
                        help="Entropy bonus lambda value")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory")
    parser.add_argument("--wandb-project", type=str, default="temporal-gfn-forecasting",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="nadhirvincenthassen",
                        help="W&B entity name")
    parser.add_argument("--offline", action="store_true",
                        help="Use W&B in offline mode")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up environment variable to avoid OpenMP issues
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Create results directory if not specified
    if args.results_dir is None:
        args.results_dir = os.path.join("results", args.name)
    
    # Make sure directories exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "evaluation"), exist_ok=True)
    
    # Generate the Hydra command
    cmd = [
        "python", "scripts/train.py",
        "--config-name", "base_config",
        f"dataset={args.dataset}_config",
        f"model=transformer_config",
        f"quantization={'adaptive' if args.quantization == 'adaptive' else 'fixed'}_config",
        f"quantization.k_initial={args.k}",
        f"quantization.adaptive={'true' if args.quantization == 'adaptive' else 'false'}",
        f"policy={'learned' if args.policy == 'learned' else 'uniform'}_config",
        f"policy.backward_policy_type={args.policy}",
        f"training.epochs={args.epochs}",
        f"training.batch_size={args.batch_size}",
        f"gfn.lambda_entropy={args.entropy}",
        f"use_wandb=true",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_project={args.wandb_project}",
        f"wandb_name={args.name}",
        f"wandb_mode={'offline' if args.offline else 'online'}",
        f"results_dir={args.results_dir}"
    ]
    
    # Print command
    print("Running command:")
    print(" ".join(cmd))
    
    # Save the command to a file in the results directory
    with open(os.path.join(args.results_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd))
    
    # Run the command
    try:
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Save output to a log file
        with open(os.path.join(args.results_dir, "train_log.txt"), "w") as f:
            f.write(process.stdout)
        
        print("Training completed successfully!")
        print(f"Results saved to: {args.results_dir}")
        print(f"Check W&B dashboard at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        
        # Save error output to a log file
        with open(os.path.join(args.results_dir, "error_log.txt"), "w") as f:
            f.write(e.stdout)
        
        print(f"Error log saved to: {os.path.join(args.results_dir, 'error_log.txt')}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main()) 