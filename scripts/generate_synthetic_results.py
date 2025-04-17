#!/usr/bin/env python
"""
Script to generate synthetic metrics data for testing plot_results.py.
"""
import os
import json
import numpy as np
import random
import torch
from datetime import datetime
import shutil
import argparse

# Define experiment configurations
experiments = [
    {
        "name": "adaptive_k10",
        "quantization": "adaptive",
        "k": 10,
        "policy_type": "uniform",
        "entropy_bonus": 0.01
    },
    {
        "name": "adaptive_k20",
        "quantization": "adaptive",
        "k": 20,
        "policy_type": "uniform",
        "entropy_bonus": 0.01
    },
    {
        "name": "fixed_k10",
        "quantization": "fixed",
        "k": 10,
        "policy_type": "uniform",
        "entropy_bonus": 0.01
    },
    {
        "name": "fixed_k20",
        "quantization": "fixed",
        "k": 20,
        "policy_type": "uniform",
        "entropy_bonus": 0.01
    },
    {
        "name": "learned_policy",
        "quantization": "adaptive",
        "k": 10,
        "policy_type": "learned",
        "entropy_bonus": 0.01
    }
]

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def generate_metrics(config):
    """Generate synthetic metrics data in the format expected by plot_results.py."""
    # Base performance values
    base_wql = 0.3
    base_crps = 0.2
    base_mase = 1.5
    
    # Modify based on configuration
    k_factor = 1.0 - 0.02 * config["k"]  # More bins = better performance
    quant_factor = 0.9 if config["quantization"] == "adaptive" else 1.0  # Adaptive is better
    policy_factor = 0.95 if config["policy_type"] == "learned" else 1.0  # Learned is better
    
    # Add some randomness
    noise = np.random.normal(0, 0.05)
    
    # Calculate metrics
    wql = base_wql * k_factor * quant_factor * policy_factor + noise
    crps = base_crps * k_factor * quant_factor * policy_factor + noise
    mase = base_mase * k_factor * quant_factor * policy_factor + noise
    
    # Ensure values are positive
    wql = max(0.05, wql)
    crps = max(0.05, crps)
    mase = max(0.5, mase)
    
    # Format metrics in the expected structure
    return {
        "overall": {
            "wql": wql,
            "crps": crps,
            "mase": mase,
        },
        "per_horizon": {
            "wql": [wql * (1 + 0.05 * i) for i in range(8)],
            "crps": [crps * (1 + 0.05 * i) for i in range(8)],
            "mase": [mase * (1 + 0.05 * i) for i in range(8)]
        },
        "per_series": {
            "wql": [wql * (1 + np.random.normal(0, 0.1)) for _ in range(10)],
            "crps": [crps * (1 + np.random.normal(0, 0.1)) for _ in range(10)],
            "mase": [mase * (1 + np.random.normal(0, 0.1)) for _ in range(10)]
        },
        "metadata": {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "quantization": config["quantization"],
                "k": config["k"],
                "policy_type": config["policy_type"],
                "entropy_bonus": config["entropy_bonus"]
            }
        }
    }

def generate_tensorboard_data(config, num_epochs=50):
    """Generate synthetic training curve data."""
    # Base starting and ending values
    start_loss = 2.0
    end_loss = 0.5
    
    # Modify based on configuration
    k_factor = 1.0 - 0.01 * config["k"]  # More bins = better convergence
    quant_factor = 0.9 if config["quantization"] == "adaptive" else 1.0  # Adaptive converges better
    policy_factor = 0.95 if config["policy_type"] == "learned" else 1.0  # Learned converges better
    
    end_loss = end_loss * k_factor * quant_factor * policy_factor
    
    # Generate exponentially decreasing loss curve with noise
    steps = np.linspace(0, 1, num_epochs)
    decay_rate = 3  # Controls how quickly the loss decreases
    
    loss_curve = start_loss * np.exp(-decay_rate * steps) + end_loss
    
    # Add noise to make it look realistic
    noise = np.random.normal(0, 0.1, num_epochs)
    loss_curve += noise
    
    # Ensure values are positive
    loss_curve = np.maximum(0.1, loss_curve)
    
    # Generate reward curve (inverse relationship with loss)
    reward_curve = 1.0 - 0.5 * loss_curve / start_loss + 0.1 * np.random.normal(0, 1, num_epochs)
    reward_curve = np.clip(reward_curve, 0.0, 1.0)
    
    # Generate entropy curve
    entropy_curve = 1.0 * np.exp(-1 * steps) + 0.1 + 0.05 * np.random.normal(0, 1, num_epochs)
    entropy_curve = np.clip(entropy_curve, 0.0, 1.0)
    
    # Generate quantization bins curve (for adaptive quantization)
    if config["quantization"] == "adaptive":
        k_curve = np.ones(num_epochs) * config["k"]
        
        # Increase k over time for adaptive quantization
        for i in range(10, num_epochs, 8):
            if k_curve[i-1] < 100:  # Max K value
                k_curve[i:] += 5
    else:
        k_curve = np.ones(num_epochs) * config["k"]
    
    return {
        "loss": loss_curve.tolist(),
        "reward": reward_curve.tolist(),
        "entropy": entropy_curve.tolist(),
        "k": k_curve.tolist(),
        "epochs": list(range(1, num_epochs + 1))
    }

def create_tensorboard_logs(config, tb_data, logs_dir):
    """Create synthetic TensorBoard log files."""
    # Since we can't easily create actual TensorBoard log files, 
    # we'll create a format that our plot_results.py can read
    
    metrics_by_epoch = []
    for i in range(len(tb_data["epochs"])):
        metrics_by_epoch.append({
            "step": tb_data["epochs"][i],
            "loss": tb_data["loss"][i],
            "reward": tb_data["reward"][i],
            "entropy": tb_data["entropy"][i],
            "k": tb_data["k"][i]
        })
    
    with open(os.path.join(logs_dir, "tensorboard_metrics.json"), "w") as f:
        json.dump(metrics_by_epoch, f, indent=2)

def main():
    """Main function to generate and save synthetic results."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate synthetic ablation study data")
    parser.add_argument("--output_dir", type=str, default="results/synthetic_data",
                       help="Directory to save synthetic data (default: results/synthetic_data)")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs for training curves (default: 50)")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing data")
    args = parser.parse_args()
    
    base_dir = args.output_dir
    
    # Remove existing directory to avoid confusion
    if os.path.exists(base_dir):
        if args.force:
            shutil.rmtree(base_dir)
        else:
            print(f"Directory {base_dir} already exists. Use --force to overwrite.")
            return
    
    # Remove the evaluation directory if it exists
    if os.path.exists(os.path.join(base_dir, "evaluation")):
        shutil.rmtree(os.path.join(base_dir, "evaluation"))
    
    # Generate data for each experiment
    for config in experiments:
        # Create experiment directory
        exp_dir = os.path.join(base_dir, config["name"])
        eval_dir = os.path.join(exp_dir, "evaluation")
        logs_dir = os.path.join(exp_dir, "logs")
        tensorboard_dir = os.path.join(logs_dir, "tensorboard")
        
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Generate and save metrics
        metrics = generate_metrics(config)
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate and save training curves
        tb_data = generate_tensorboard_data(config, num_epochs=args.num_epochs)
        with open(os.path.join(logs_dir, "training_curves.json"), "w") as f:
            json.dump(tb_data, f, indent=2)
        
        # Create synthetic TensorBoard logs
        create_tensorboard_logs(config, tb_data, tensorboard_dir)
        
        # Save configuration
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            f.write(f"experiment_name: {config['name']}\n")
            f.write(f"quantization:\n")
            f.write(f"  type: {config['quantization']}\n")
            f.write(f"  k_initial: {config['k']}\n")
            f.write(f"policy_type: {config['policy_type']}\n")
            f.write(f"entropy_bonus: {config['entropy_bonus']}\n")
            f.write(f"epochs: {args.num_epochs}\n")
            f.write(f"batch_size: 32\n")
            f.write(f"learning_rate: 0.001\n")
    
    print(f"Generated synthetic results for {len(experiments)} experiments in {base_dir}")

if __name__ == "__main__":
    main() 