#!/usr/bin/env python
"""
Python script to run multiple experiments with different configurations.

This script provides a programmatic way to run a series of experiments
with varying hyperparameters like quantization methods, entropy values, etc.,
and automatically logs results to Weights & Biases.
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import itertools

def create_experiment_dir(results_dir: str, experiment_name: str) -> str:
    """Create experiment directory and subdirectories."""
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def build_train_command(
    experiment_name: str,
    dataset: str,
    quantization_type: str,
    k_value: int,
    entropy: float,
    policy_type: str,
    epochs: int,
    batch_size: int,
    results_dir: str,
    gpu_id: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    offline: bool = False,
) -> List[str]:
    """Build command for training a model with the given configuration."""
    
    # Base command with python script
    cmd = ["python", "scripts/train.py"]
    
    # Add Hydra config overrides
    cmd.append(f"dataset={dataset}_config")
    cmd.append(f"model=transformer_config")
    
    # Quantization settings
    if quantization_type == "adaptive":
        cmd.append(f"quantization=adaptive_config")
        cmd.append(f"quantization.adaptive=true")
    else:
        cmd.append(f"quantization=fixed_config")
        cmd.append(f"quantization.adaptive=false")
    cmd.append(f"quantization.k_initial={k_value}")
    
    # Policy type
    cmd.append(f"policy={policy_type}_config")
    cmd.append(f"policy.backward_policy_type={policy_type}")
    
    # Training parameters
    cmd.append(f"training.epochs={epochs}")
    cmd.append(f"training.batch_size={batch_size}")
    cmd.append(f"gfn.lambda_entropy={entropy}")
    
    # Results directory
    exp_dir = os.path.join(results_dir, experiment_name)
    cmd.append(f"+results_dir={exp_dir}")
    
    # GPU settings
    if gpu_id >= 0:
        cmd.append(f"gpu={gpu_id}")
    
    # Wandb settings
    if use_wandb:
        cmd.append(f"use_wandb=true")
        cmd.append(f"wandb_project={wandb_project}")
        cmd.append(f"wandb_entity={wandb_entity}")
        cmd.append(f"wandb_name={experiment_name}")
        
        if offline:
            cmd.append("offline=true")
    
    return cmd

def build_eval_command(
    experiment_name: str,
    results_dir: str,
    gpu_id: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    offline: bool = False,
) -> List[str]:
    """Build command for evaluating a trained model."""
    
    exp_dir = os.path.join(results_dir, experiment_name)
    checkpoint_path = os.path.join(exp_dir, "checkpoints", "best_model.pt")
    config_path = os.path.join(exp_dir, "config.yaml")
    
    # Skip if checkpoint doesn't exist
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return []
    
    # Base command
    cmd = [
        "python", "scripts/evaluate.py",
        "--checkpoint_path", checkpoint_path,
        "--config_path", config_path,
        "--output_dir", os.path.join(exp_dir, "evaluation"),
    ]
    
    # GPU settings
    if gpu_id >= 0:
        cmd.extend(["--gpu", str(gpu_id)])
    
    # Wandb settings
    if use_wandb:
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", wandb_project])
        cmd.extend(["--wandb_entity", wandb_entity])
        cmd.extend(["--wandb_name", f"{experiment_name}_eval"])
        
        if offline:
            cmd.append("--offline")
    
    return cmd

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return whether it succeeded."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"Command failed with exception: {e}")
        return False

def run_experiment(
    dataset: str,
    quantization_type: str,
    k_value: int,
    entropy: float,
    policy_type: str,
    epochs: int,
    batch_size: int,
    results_dir: str,
    gpu_id: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    offline: bool = False,
) -> Tuple[str, bool]:
    """Run a single experiment with the given configuration."""
    
    # Skip invalid combinations
    if quantization_type == "fixed" and policy_type == "learned":
        print("Skipping invalid combination: fixed quantization with learned policy")
        return "", False
    
    # Create experiment name
    experiment_name = f"{dataset}_{quantization_type}_k{k_value}_entropy{entropy}_{policy_type}"
    
    # Create experiment directory
    create_experiment_dir(results_dir, experiment_name)
    
    print(f"\n{'=' * 70}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'=' * 70}")
    
    # Run training
    train_cmd = build_train_command(
        experiment_name=experiment_name,
        dataset=dataset,
        quantization_type=quantization_type,
        k_value=k_value,
        entropy=entropy,
        policy_type=policy_type,
        epochs=epochs,
        batch_size=batch_size,
        results_dir=results_dir,
        gpu_id=gpu_id,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        offline=offline,
    )
    
    train_success = run_command(
        train_cmd, 
        f"Training model for {experiment_name}"
    )
    
    if not train_success:
        return experiment_name, False
    
    # Run evaluation
    eval_cmd = build_eval_command(
        experiment_name=experiment_name,
        results_dir=results_dir,
        gpu_id=gpu_id,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        offline=offline,
    )
    
    if eval_cmd:
        eval_success = run_command(
            eval_cmd,
            f"Evaluating model for {experiment_name}"
        )
        
        if not eval_success:
            return experiment_name, False
    
    return experiment_name, True

def run_analysis(
    datasets: List[str],
    results_dir: str,
    output_dir: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    offline: bool = False,
) -> None:
    """Run analysis scripts on experiment results."""
    
    print(f"\n{'=' * 70}")
    print(f"Running analysis on experiment results")
    print(f"{'=' * 70}")
    
    for dataset in datasets:
        # Build command for analysis
        cmd = ["python", "scripts/enhanced_ablation_viz.py"]
        cmd.extend(["--results_dir", results_dir])
        cmd.extend(["--output_dir", os.path.join(output_dir, f"{dataset}_analysis", "plots")])
        cmd.extend(["--format", "png"])
        
        if use_wandb:
            cmd.append("--use_wandb")
            cmd.extend(["--wandb_project", wandb_project])
            cmd.extend(["--wandb_entity", wandb_entity])
            cmd.extend(["--wandb_name", f"{dataset}_visualization"])
            
            if offline:
                cmd.append("--offline")
        
        run_command(cmd, f"Running enhanced visualization for {dataset}")
        
        # Additional quantization analysis
        cmd = ["python", "scripts/quantization_analysis.py"]
        cmd.extend(["--results_dir", results_dir])
        cmd.extend(["--output_dir", os.path.join(output_dir, f"{dataset}_analysis", "quantization")])
        
        if use_wandb:
            cmd.append("--use_wandb")
            cmd.extend(["--wandb_project", wandb_project])
            cmd.extend(["--wandb_entity", wandb_entity])
            cmd.extend(["--wandb_name", f"{dataset}_quantization_analysis"])
            
            if offline:
                cmd.append("--offline")
        
        run_command(cmd, f"Running quantization analysis for {dataset}")

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Run multiple experiments with different configurations")
    
    # Basic parameters
    parser.add_argument("--datasets", type=str, default="synthetic,eeg", 
                        help="Comma-separated list of datasets to use")
    parser.add_argument("--quantizations", type=str, default="adaptive,fixed",
                        help="Comma-separated list of quantization types to use")
    parser.add_argument("--k-values", type=str, default="5,10,20",
                        help="Comma-separated list of k values to use")
    parser.add_argument("--entropy-values", type=str, default="0,0.001,0.01,0.1",
                        help="Comma-separated list of entropy values to use")
    parser.add_argument("--policy-types", type=str, default="uniform,learned",
                        help="Comma-separated list of policy types to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    
    # Output parameters
    parser.add_argument("--results-dir", type=str, default="results/full_experiment",
                        help="Directory to save results")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip running analysis scripts after experiments")
    
    # Hardware parameters
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (-1 for CPU)")
    
    # W&B parameters
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="temporal-gfn-forecasting",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="nadhirvincenthassen",
                        help="W&B entity name")
    parser.add_argument("--offline", action="store_true",
                        help="Run W&B in offline mode")
    
    args = parser.parse_args()
    
    # Parse lists
    datasets = args.datasets.split(",")
    quantizations = args.quantizations.split(",")
    k_values = [int(k) for k in args.k_values.split(",")]
    entropy_values = [float(e) for e in args.entropy_values.split(",")]
    policy_types = args.policy_types.split(",")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create analysis directory if needed
    analysis_dir = os.path.join(args.results_dir, "analysis")
    if not args.no_analysis:
        os.makedirs(analysis_dir, exist_ok=True)
    
    # Track timing
    start_time = time.time()
    experiment_results = []
    
    # Log configuration
    config = {
        "datasets": datasets,
        "quantizations": quantizations,
        "k_values": k_values,
        "entropy_values": entropy_values,
        "policy_types": policy_types,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "results_dir": args.results_dir,
        "gpu": args.gpu,
        "use_wandb": not args.no_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "offline": args.offline,
        "run_analysis": not args.no_analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(args.results_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run all experiments
    for dataset, quant_type, k_value, entropy, policy_type in itertools.product(
        datasets, quantizations, k_values, entropy_values, policy_types
    ):
        # Skip invalid combinations
        if quant_type == "fixed" and policy_type == "learned":
            continue
            
        experiment_name, success = run_experiment(
            dataset=dataset,
            quantization_type=quant_type,
            k_value=k_value,
            entropy=entropy,
            policy_type=policy_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            results_dir=args.results_dir,
            gpu_id=args.gpu,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            offline=args.offline,
        )
        
        if experiment_name:
            experiment_results.append({
                "name": experiment_name,
                "success": success,
                "dataset": dataset,
                "quantization_type": quant_type,
                "k_value": k_value,
                "entropy": entropy,
                "policy_type": policy_type,
            })
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Run analysis if requested
    if not args.no_analysis:
        run_analysis(
            datasets=datasets,
            results_dir=args.results_dir,
            output_dir=analysis_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            offline=args.offline,
        )
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Experiment Summary")
    print(f"{'=' * 70}")
    print(f"Total experiments: {len(experiment_results)}")
    print(f"Successful experiments: {sum(1 for r in experiment_results if r['success'])}")
    print(f"Failed experiments: {sum(1 for r in experiment_results if not r['success'])}")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print()
    
    # Print individual results
    print("Experiment Results:")
    for result in experiment_results:
        status = "Success" if result["success"] else "Failed"
        print(f"{result['name']}: {status}")
    
    # Save results
    with open(os.path.join(args.results_dir, "experiment_results.json"), "w") as f:
        json.dump({
            "results": experiment_results,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    
    print(f"\nAll experiments and analysis completed!")
    print(f"Results saved to: {args.results_dir}")
    if not args.no_wandb:
        print(f"All results logged to W&B project: {args.wandb_project}")
        if args.offline:
            print("Note: W&B was run in offline mode. Use 'wandb sync' to upload results when online.")

if __name__ == "__main__":
    main() 