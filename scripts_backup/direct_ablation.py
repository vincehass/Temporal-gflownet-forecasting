#!/usr/bin/env python3
"""
Direct ablation experiment runner with proper W&B integration.
This script bypasses the shell scripts and directly calls the necessary Python functions.
"""

import os
import sys
import argparse
import time
import yaml
import wandb
import subprocess
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment with direct W&B integration")
    parser.add_argument('--name', type=str, default=f"eeg_experiment_{int(time.time())}")
    parser.add_argument('--dataset', type=str, default="eeg")
    parser.add_argument('--quantization', type=str, choices=['adaptive', 'fixed'], default='adaptive')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--policy', type=str, choices=['uniform', 'learned'], default='uniform')
    parser.add_argument('--entropy', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--wandb-project', type=str, default="temporal-gfn-forecasting")
    parser.add_argument('--wandb-entity', type=str, default="nadhirvincenthassen")
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--offline', action='store_true', help="Run W&B in offline mode")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Handle OpenMP issues
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Set up results directory
    if args.results_dir is None:
        results_dir = os.path.join("results", args.name)
    else:
        results_dir = args.results_dir
        
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "evaluation"), exist_ok=True)
    
    # Save configuration
    config = {
        "experiment_name": args.name,
        "dataset": args.dataset,
        "model_type": "transformer",
        "quantization": {
            "type": args.quantization,
            "k_initial": args.k,
            "adaptive": args.quantization == "adaptive"
        },
        "policy_type": args.policy,
        "entropy_bonus": args.entropy,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Initialize W&B directly
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.name,
        config=config,
        mode="offline" if args.offline else "online"
    )
    
    print(f"W&B run initialized with ID: {run.id}")
    print(f"View run at: {run.get_url()}")
    
    # Generate hydra command
    hydra_command = [
        "python", "scripts/train.py",
        "--config-name", "base_config",
        f"dataset={args.dataset}_config",
        f"model=transformer_config",
        f"quantization={args.quantization}_config",
        f"quantization.k_initial={args.k}",
        f"quantization.adaptive={'true' if args.quantization == 'adaptive' else 'false'}",
        f"policy={'learned' if args.policy == 'learned' else 'uniform'}_config",
        f"policy.backward_policy_type={args.policy}",
        f"training.epochs={args.epochs}",
        f"training.batch_size={args.batch_size}",
        f"gfn.lambda_entropy={args.entropy}",
        f"gpu={args.gpu}",
        f"results_dir={results_dir}"
    ]
    
    # Save the command
    with open(os.path.join(results_dir, "command.txt"), "w") as f:
        f.write(" ".join(hydra_command))
    
    # Print the command
    print("Running command:")
    print(" ".join(hydra_command))
    
    # Run the command
    process = subprocess.Popen(
        hydra_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Set up log file
    log_file = open(os.path.join(results_dir, "logs", "train.log"), "w")
    
    # Process output line by line
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        log_file.write(line)
        log_file.flush()
        
        # Extract metrics for W&B logging
        if "Epoch" in line and "Loss" in line:
            try:
                # Example: Epoch 0 - Loss: 5152.813585, Forward: -54.456765, Backward: -55.262046, Reward: -68.820735, Entropy: 54.492954, K: 10
                parts = line.strip().split(' - ')[1].split(', ')
                metrics = {}
                
                for part in parts:
                    # More robust parsing with error handling
                    if ":" in part:
                        key, value = part.split(':', 1)  # Split on first colon only
                        key = key.strip().lower()
                        try:
                            metrics[key] = float(value.strip())
                        except ValueError:
                            # If we can't convert to float, just use the string value
                            metrics[key] = value.strip()
                    else:
                        # Handle case where there's no colon
                        continue
                
                # Log to wandb
                if metrics:
                    wandb.log(metrics)
            except Exception as e:
                print(f"Error parsing metrics: {e}")
                # Print the line that caused the error for debugging
                print(f"Problematic line: {line.strip()}")
        
        # Extract adaptive quantization metrics if available
        elif "Adaptive K updated" in line:
            try:
                # Example: [2023-01-01 12:00:00,000][TemporalGFN][INFO] - Adaptive K updated: K: 10 -> 11, Delta_t: 1, Reward variance: 12.34, Reward mean: 5.67
                quant_metrics = {}
                
                # Extract K before and after
                if "K:" in line and "->" in line:
                    k_part = line.split("K:")[1].split(",")[0]
                    k_before, k_after = map(int, k_part.split("->"))
                    quant_metrics["k_before"] = k_before
                    quant_metrics["k_after"] = k_after
                    quant_metrics["k_delta"] = k_after - k_before
                
                # Extract Delta_t
                if "Delta_t:" in line:
                    delta_t = int(line.split("Delta_t:")[1].split(",")[0].strip())
                    quant_metrics["delta_t"] = delta_t
                
                # Extract reward stats
                if "Reward variance:" in line:
                    reward_var = float(line.split("Reward variance:")[1].split(",")[0].strip())
                    quant_metrics["reward_variance"] = reward_var
                
                if "Reward mean:" in line:
                    reward_mean = float(line.split("Reward mean:")[1].split(",")[0].strip())
                    quant_metrics["reward_mean"] = reward_mean
                
                # Log ratio of variance to mean (this is used in the adaptive update)
                if "reward_variance" in quant_metrics and "reward_mean" in quant_metrics:
                    var_mean_ratio = quant_metrics["reward_variance"] / abs(quant_metrics["reward_mean"])
                    quant_metrics["var_mean_ratio"] = var_mean_ratio
                
                # Log to wandb
                if quant_metrics:
                    wandb.log(quant_metrics)
            except Exception as e:
                print(f"Error parsing adaptive quantization metrics: {e}")
                print(f"Problematic line: {line.strip()}")
        
        # Extract trajectory statistics if available
        elif "Trajectory stats" in line:
            try:
                # Example: [2023-01-01 12:00:00,000][TemporalGFN][INFO] - Trajectory stats: Diversity: 0.85, Min reward: -12.3, Max reward: 5.6, Mean reward: -3.4
                traj_metrics = {}
                
                # Extract diversity (entropy-based measure)
                if "Diversity:" in line:
                    diversity = float(line.split("Diversity:")[1].split(",")[0].strip())
                    traj_metrics["trajectory_diversity"] = diversity
                
                # Extract reward stats
                if "Min reward:" in line:
                    min_reward = float(line.split("Min reward:")[1].split(",")[0].strip())
                    traj_metrics["min_trajectory_reward"] = min_reward
                
                if "Max reward:" in line:
                    max_reward = float(line.split("Max reward:")[1].split(",")[0].strip())
                    traj_metrics["max_trajectory_reward"] = max_reward
                
                if "Mean reward:" in line:
                    mean_reward = float(line.split("Mean reward:")[1].split(",")[0].strip())
                    traj_metrics["mean_trajectory_reward"] = mean_reward
                
                # Calculate reward range
                if "min_trajectory_reward" in traj_metrics and "max_trajectory_reward" in traj_metrics:
                    reward_range = traj_metrics["max_trajectory_reward"] - traj_metrics["min_trajectory_reward"]
                    traj_metrics["trajectory_reward_range"] = reward_range
                
                # Log to wandb
                if traj_metrics:
                    wandb.log(traj_metrics)
            except Exception as e:
                print(f"Error parsing trajectory metrics: {e}")
                print(f"Problematic line: {line.strip()}")
    
    # Close log file
    log_file.close()
    
    # Wait for process to complete
    process.wait()
    
    # Check if the process completed successfully
    if process.returncode == 0:
        print(f"Training completed successfully!")
        
        # Run evaluation
        eval_command = [
            "python", "scripts/evaluate.py",
            "--checkpoint_path", os.path.join(results_dir, "checkpoints", "best_model.pt"),
            "--config_path", os.path.join(results_dir, "config.yaml"),
            "--output_dir", os.path.join(results_dir, "evaluation")
        ]
        
        print("\nRunning evaluation:")
        print(" ".join(eval_command))
        
        eval_process = subprocess.run(
            eval_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Save evaluation log
        with open(os.path.join(results_dir, "logs", "eval.log"), "w") as f:
            f.write(eval_process.stdout)
        
        # Load metrics if available
        metrics_path = os.path.join(results_dir, "evaluation", "metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                
                # Log evaluation metrics to W&B
                if "overall" in metrics:
                    wandb.log({"eval_" + k: v for k, v in metrics["overall"].items()})
                    
        # Finish W&B run
        wandb.finish()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"W&B run: {run.get_url()}")
        
        return 0
    else:
        print(f"Training failed with exit code {process.returncode}")
        wandb.finish()
        return process.returncode

if __name__ == "__main__":
    sys.exit(main()) 