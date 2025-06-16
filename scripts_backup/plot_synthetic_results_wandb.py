#!/usr/bin/env python
"""
Script to visualize synthetic results with Weights & Biases integration.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import wandb
import pandas as pd
import argparse

def load_metrics(results_dir):
    """
    Load metrics from all experiments in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary of metrics by experiment name
    """
    metrics_by_exp = {}
    
    # List all directories in the results directory
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(results_dir, exp_name)
        metrics_path = os.path.join(exp_dir, "evaluation", "metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                metrics_by_exp[exp_name] = metrics
                
                # Add experiment name for reference
                if "metadata" not in metrics:
                    metrics["metadata"] = {}
                metrics["metadata"]["experiment"] = exp_name
    
    return metrics_by_exp

def load_training_curves(results_dir):
    """
    Load training curves from all experiments in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary of training curves by experiment name
    """
    curves_by_exp = {}
    
    # List all directories in the results directory
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(results_dir, exp_name)
        curves_path = os.path.join(exp_dir, "logs", "training_curves.json")
        
        if os.path.exists(curves_path):
            with open(curves_path, "r") as f:
                curves = json.load(f)
                curves_by_exp[exp_name] = curves
    
    return curves_by_exp

def load_configs(results_dir):
    """
    Load configurations from all experiments in the results directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary of configuration dictionaries by experiment name
    """
    configs_by_exp = {}
    
    # List all directories in the results directory
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    for exp_name in exp_dirs:
        exp_dir = os.path.join(results_dir, exp_name)
        config_path = os.path.join(exp_dir, "config.yaml")
        
        if os.path.exists(config_path):
            # Parse YAML manually since it's simple
            config = {}
            with open(config_path, "r") as f:
                for line in f:
                    if ":" in line and not line.strip().startswith("#"):
                        key, value = line.split(":", 1)
                        config[key.strip()] = value.strip()
            
            configs_by_exp[exp_name] = config
    
    return configs_by_exp

def plot_metrics_comparison(metrics_by_exp, output_dir, title="Metrics Comparison", use_wandb=False):
    """
    Plot comparison of metrics across experiments.
    
    Args:
        metrics_by_exp: Dictionary of metrics by experiment name
        output_dir: Directory to save plots
        title: Title for the plot
        use_wandb: Whether to log to W&B
    """
    if not metrics_by_exp:
        print("No metrics available for plotting")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure with a nice style
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Extract metrics and experiment names
    exp_names = list(metrics_by_exp.keys())
    metric_names = ["wql", "crps", "mase"]
    
    # Create a DataFrame for easier plotting
    data = []
    
    for exp_name in exp_names:
        for metric in metric_names:
            if metric in metrics_by_exp[exp_name]["overall"]:
                value = metrics_by_exp[exp_name]["overall"][metric]
                
                # Extract configuration info
                quant_type = "Unknown"
                k_value = "Unknown"
                policy_type = "Unknown"
                
                if "metadata" in metrics_by_exp[exp_name] and "config" in metrics_by_exp[exp_name]["metadata"]:
                    config = metrics_by_exp[exp_name]["metadata"]["config"]
                    quant_type = config.get("quantization", "Unknown")
                    k_value = config.get("k", "Unknown")
                    policy_type = config.get("policy_type", "Unknown")
                
                data.append({
                    "Experiment": exp_name,
                    "Metric": metric.upper(),
                    "Value": value,
                    "Quantization": quant_type,
                    "K": k_value,
                    "Policy": policy_type
                })
    
    df = pd.DataFrame(data)
    
    # Plot the metrics grouped by experiment
    plt.subplot(2, 1, 1)
    g = sns.barplot(x="Experiment", y="Value", hue="Metric", data=df)
    plt.title(f"{title} by Experiment", fontsize=16)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    
    # Plot the metrics grouped by quantization type and K value
    plt.subplot(2, 1, 2)
    df["Config"] = df.apply(lambda row: f"{row['Quantization']} (K={row['K']})", axis=1)
    g = sns.barplot(x="Config", y="Value", hue="Metric", data=df)
    plt.title(f"{title} by Configuration", fontsize=16)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    
    plt.tight_layout()
    
    # Save figure locally
    metrics_by_exp_path = os.path.join(output_dir, "metrics_comparison_by_experiment.png")
    plt.savefig(metrics_by_exp_path, dpi=300, bbox_inches="tight")
    
    # Log to W&B if requested
    if use_wandb:
        wandb.log({"metrics_comparison_by_experiment": wandb.Image(metrics_by_exp_path)})
        
        # Also log the metrics as tables/charts in W&B
        for metric in metric_names:
            metric_data = df[df["Metric"] == metric.upper()]
            metric_table = wandb.Table(dataframe=metric_data)
            wandb.log({f"{metric.upper()}_by_experiment": metric_table})
    
    # Plot the metrics by policy type
    plt.figure(figsize=(12, 6))
    g = sns.barplot(x="Policy", y="Value", hue="Metric", data=df)
    plt.title(f"{title} by Policy Type", fontsize=16)
    plt.xlabel("Policy Type", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(title="Metric")
    
    plt.tight_layout()
    
    # Save figure locally
    metrics_by_policy_path = os.path.join(output_dir, "metrics_comparison_by_policy.png")
    plt.savefig(metrics_by_policy_path, dpi=300, bbox_inches="tight")
    
    # Log to W&B if requested
    if use_wandb:
        wandb.log({"metrics_comparison_by_policy": wandb.Image(metrics_by_policy_path)})

def plot_training_curves(curves_by_exp, output_dir, title="Training Curves", use_wandb=False):
    """
    Plot training curves across experiments.
    
    Args:
        curves_by_exp: Dictionary of training curves by experiment name
        output_dir: Directory to save plots
        title: Title for the plot
        use_wandb: Whether to log to W&B
    """
    if not curves_by_exp:
        print("No training curves available for plotting")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure with a nice style
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Extract experiment names
    exp_names = list(curves_by_exp.keys())
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    for exp_name in exp_names:
        if "loss" in curves_by_exp[exp_name] and "epochs" in curves_by_exp[exp_name]:
            epochs = curves_by_exp[exp_name]["epochs"]
            loss = curves_by_exp[exp_name]["loss"]
            plt.plot(epochs, loss, label=exp_name)
            
            # Log to W&B directly as time series
            if use_wandb:
                for i, epoch in enumerate(epochs):
                    wandb.log({f"training/loss_{exp_name}": loss[i]}, step=epoch)
    
    plt.title("Loss over Time", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot reward curves
    plt.subplot(2, 2, 2)
    for exp_name in exp_names:
        if "reward" in curves_by_exp[exp_name] and "epochs" in curves_by_exp[exp_name]:
            epochs = curves_by_exp[exp_name]["epochs"]
            reward = curves_by_exp[exp_name]["reward"]
            plt.plot(epochs, reward, label=exp_name)
            
            # Log to W&B directly as time series
            if use_wandb:
                for i, epoch in enumerate(epochs):
                    wandb.log({f"training/reward_{exp_name}": reward[i]}, step=epoch)
    
    plt.title("Reward over Time", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot entropy curves
    plt.subplot(2, 2, 3)
    for exp_name in exp_names:
        if "entropy" in curves_by_exp[exp_name] and "epochs" in curves_by_exp[exp_name]:
            epochs = curves_by_exp[exp_name]["epochs"]
            entropy = curves_by_exp[exp_name]["entropy"]
            plt.plot(epochs, entropy, label=exp_name)
            
            # Log to W&B directly as time series
            if use_wandb:
                for i, epoch in enumerate(epochs):
                    wandb.log({f"training/entropy_{exp_name}": entropy[i]}, step=epoch)
    
    plt.title("Entropy over Time", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot k curves
    plt.subplot(2, 2, 4)
    for exp_name in exp_names:
        if "k" in curves_by_exp[exp_name] and "epochs" in curves_by_exp[exp_name]:
            epochs = curves_by_exp[exp_name]["epochs"]
            k = curves_by_exp[exp_name]["k"]
            plt.plot(epochs, k, label=exp_name)
            
            # Log to W&B directly as time series
            if use_wandb:
                for i, epoch in enumerate(epochs):
                    wandb.log({f"training/k_{exp_name}": k[i]}, step=epoch)
    
    plt.title("Quantization Bins (K) over Time", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("K", fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure locally
    training_curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(training_curves_path, dpi=300, bbox_inches="tight")
    
    # Log to W&B if requested
    if use_wandb:
        wandb.log({"training_curves": wandb.Image(training_curves_path)})

def plot_per_horizon_metrics(metrics_by_exp, output_dir, title="Per Horizon Metrics", use_wandb=False):
    """
    Plot metrics per prediction horizon across experiments.
    
    Args:
        metrics_by_exp: Dictionary of metrics by experiment name
        output_dir: Directory to save plots
        title: Title for the plot
        use_wandb: Whether to log to W&B
    """
    if not metrics_by_exp:
        print("No per-horizon metrics available for plotting")
        return
    
    # Check if per_horizon metrics exist
    has_per_horizon = False
    for exp_name in metrics_by_exp:
        if "per_horizon" in metrics_by_exp[exp_name]:
            has_per_horizon = True
            break
    
    if not has_per_horizon:
        print("No per-horizon metrics found")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure with a nice style
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # Extract experiment names and metrics
    exp_names = list(metrics_by_exp.keys())
    metric_names = ["wql", "crps", "mase"]
    
    # Create plots for each metric
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 1, figure=fig)
    
    for i, metric in enumerate(metric_names):
        ax = fig.add_subplot(gs[i, 0])
        
        for exp_name in exp_names:
            if "per_horizon" in metrics_by_exp[exp_name] and metric in metrics_by_exp[exp_name]["per_horizon"]:
                values = metrics_by_exp[exp_name]["per_horizon"][metric]
                horizons = range(1, len(values) + 1)
                ax.plot(horizons, values, marker='o', label=exp_name)
                
                # Log to W&B directly as time series
                if use_wandb:
                    for j, horizon in enumerate(horizons):
                        wandb.log({f"per_horizon/{metric}_{exp_name}": values[j]}, step=horizon)
        
        ax.set_title(f"{metric.upper()} by Prediction Horizon", fontsize=14)
        ax.set_xlabel("Prediction Horizon", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure locally
    per_horizon_path = os.path.join(output_dir, "per_horizon_metrics.png")
    plt.savefig(per_horizon_path, dpi=300, bbox_inches="tight")
    
    # Log to W&B if requested
    if use_wandb:
        wandb.log({"per_horizon_metrics": wandb.Image(per_horizon_path)})

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot synthetic results with W&B integration")
    parser.add_argument("--results_dir", type=str, default="results/synthetic_data",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="results/synthetic_plots",
                        help="Directory to save plots")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="temporal-gfn-forecasting",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="nadhirvincenthassen", 
                        help="W&B entity name")
    parser.add_argument("--wandb_name", type=str, default="temporal_gfn_adaptive_quant_ste",
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "results_dir": args.results_dir,
                "output_dir": args.output_dir
            }
        )
    
    # Load data
    metrics_by_exp = load_metrics(args.results_dir)
    curves_by_exp = load_training_curves(args.results_dir)
    configs_by_exp = load_configs(args.results_dir)
    
    # Print experiment info
    print(f"Found {len(metrics_by_exp)} experiments with metrics")
    print(f"Found {len(curves_by_exp)} experiments with training curves")
    print(f"Found {len(configs_by_exp)} experiments with configs")
    
    # Log experiment configurations to W&B
    if args.use_wandb:
        for exp_name, config in configs_by_exp.items():
            wandb.config.update({f"experiment_{exp_name}": config})
    
    # Create plots
    plot_metrics_comparison(metrics_by_exp, args.output_dir, use_wandb=args.use_wandb)
    plot_training_curves(curves_by_exp, args.output_dir, use_wandb=args.use_wandb)
    plot_per_horizon_metrics(metrics_by_exp, args.output_dir, use_wandb=args.use_wandb)
    
    print(f"Plots generated and saved to {args.output_dir}")
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 