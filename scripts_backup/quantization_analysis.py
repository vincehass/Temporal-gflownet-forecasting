#!/usr/bin/env python
"""
Script for quantization-specific analysis with full Weights & Biases integration.

This script analyzes the effects of different quantization strategies in temporal GFlowNets
and creates detailed visualizations of quantization dynamics, with all results synced to wandb.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.ticker as mticker

# Define consistent colors and styles for visualizations
COLORS = {
    'adaptive_k5': '#1f77b4',    # blue
    'adaptive_k10': '#ff7f0e',   # orange
    'adaptive_k20': '#2ca02c',   # green
    'fixed_k5': '#d62728',       # red
    'fixed_k10': '#9467bd',      # purple
    'fixed_k20': '#8c564b',      # brown
    'learned_policy': '#e377c2', # pink
    'default': '#7f7f7f'         # gray
}

def get_color(exp_name: str) -> str:
    """Get consistent color for experiment types."""
    for key in COLORS:
        if key in exp_name:
            return COLORS[key]
    return COLORS['default']

def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics from {metrics_file}: {e}")
        return {}

def load_training_curves(training_curves_file: str) -> Dict:
    """Load training curves from a JSON file."""
    try:
        with open(training_curves_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training curves from {training_curves_file}: {e}")
        return {}

def analyze_k_dynamics(curves_data: Dict, output_dir: str, use_wandb: bool = False):
    """
    Analyze quantization bin (k) dynamics throughout training.
    
    Args:
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save output plots
        use_wandb: Whether to log results to wandb
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect k dynamics data
    k_data = []
    for exp_name, curves in curves_data.items():
        if 'k' in curves and 'epochs' in curves:
            for epoch, k_value in zip(curves['epochs'], curves['k']):
                k_data.append({
                    'Experiment': exp_name,
                    'Epoch': epoch,
                    'K': k_value
                })
    
    if not k_data:
        print("No quantization (k) data found in training curves.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(k_data)
    
    # Create time series plot
    plt.figure(figsize=(12, 6))
    for exp_name in df['Experiment'].unique():
        exp_data = df[df['Experiment'] == exp_name]
        plt.plot(
            exp_data['Epoch'], 
            exp_data['K'],
            marker='o',
            label=exp_name,
            color=get_color(exp_name),
            linewidth=2
        )
    
    plt.title('Quantization Bin Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Quantization Bins (K)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Experiment')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'k_dynamics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if enabled
    if use_wandb:
        wandb.log({"quantization_dynamics/k_evolution": wandb.Image(plot_path)})
        
        # Create a wandb Table for interactive exploration
        columns = ["Experiment", "Epoch", "K"]
        k_table = wandb.Table(dataframe=df[columns])
        wandb.log({"quantization_dynamics/k_values_table": k_table})
        
        # Log individual k values as metrics for each experiment
        for exp_name in df['Experiment'].unique():
            exp_data = df[df['Experiment'] == exp_name]
            for _, row in exp_data.iterrows():
                wandb.log({
                    f"k_values/{exp_name}": row['K'],
                    "epoch": row['Epoch']
                })
    
    plt.close()
    
    # Create additional analysis - k growth rate
    adaptive_exps = [exp for exp in df['Experiment'].unique() if 'adaptive' in exp.lower()]
    if adaptive_exps:
        plt.figure(figsize=(10, 5))
        
        for exp_name in adaptive_exps:
            exp_data = df[df['Experiment'] == exp_name]
            
            # Calculate growth rate (change in k per epoch)
            exp_data = exp_data.sort_values('Epoch')
            k_values = exp_data['K'].values
            growth_rates = []
            epochs = []
            
            for i in range(1, len(k_values)):
                if exp_data['Epoch'].iloc[i] > exp_data['Epoch'].iloc[i-1]:
                    rate = (k_values[i] - k_values[i-1]) / (exp_data['Epoch'].iloc[i] - exp_data['Epoch'].iloc[i-1])
                    growth_rates.append(rate)
                    epochs.append(exp_data['Epoch'].iloc[i])
            
            if growth_rates:
                plt.plot(epochs, growth_rates, marker='o', label=exp_name, color=get_color(exp_name))
        
        plt.title('K Growth Rate in Adaptive Quantization')
        plt.xlabel('Epoch')
        plt.ylabel('K Growth Rate (bins/epoch)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Experiment')
        plt.tight_layout()
        
        # Save plot
        growth_plot_path = os.path.join(output_dir, 'k_growth_rate.png')
        plt.savefig(growth_plot_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({"quantization_dynamics/k_growth_rate": wandb.Image(growth_plot_path)})
        
        plt.close()

def analyze_quantization_vs_metrics(metrics_data: Dict, curves_data: Dict, output_dir: str, use_wandb: bool = False):
    """
    Analyze the relationship between quantization strategies and performance metrics.
    
    Args:
        metrics_data: Dictionary mapping experiment names to their metrics
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save output plots
        use_wandb: Whether to log results to wandb
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data on final k values and metrics
    analysis_data = []
    
    for exp_name in set(metrics_data.keys()) & set(curves_data.keys()):
        if 'overall' in metrics_data[exp_name] and 'k' in curves_data[exp_name]:
            # Extract experiment type
            if 'adaptive' in exp_name.lower():
                quant_type = 'Adaptive'
            elif 'fixed' in exp_name.lower():
                quant_type = 'Fixed'
            else:
                quant_type = 'Other'
                
            # Extract initial and final k values
            k_values = curves_data[exp_name]['k']
            initial_k = k_values[0] if k_values else np.nan
            final_k = k_values[-1] if k_values else np.nan
            
            # Extract metrics
            for metric_name, value in metrics_data[exp_name]['overall'].items():
                analysis_data.append({
                    'Experiment': exp_name,
                    'Quantization': quant_type,
                    'Initial_K': initial_k,
                    'Final_K': final_k,
                    'Metric': metric_name,
                    'Value': value
                })
    
    if not analysis_data:
        print("No suitable data for quantization vs metrics analysis.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Create scatter plot of Final K vs Metric Value
    for metric_name in df['Metric'].unique():
        metric_df = df[df['Metric'] == metric_name]
        
        plt.figure(figsize=(10, 6))
        
        # Plot points with different colors/markers by quantization type
        for quant_type in metric_df['Quantization'].unique():
            type_data = metric_df[metric_df['Quantization'] == quant_type]
            
            plt.scatter(
                type_data['Final_K'],
                type_data['Value'],
                label=quant_type,
                s=100,
                alpha=0.7,
                marker='o' if quant_type == 'Adaptive' else 's'
            )
            
            # Add experiment names as annotations
            for _, row in type_data.iterrows():
                plt.annotate(
                    row['Experiment'],
                    (row['Final_K'], row['Value']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        # Add trend line if enough data points
        if len(metric_df) > 2:
            try:
                # Simple linear regression
                z = np.polyfit(metric_df['Final_K'], metric_df['Value'], 1)
                p = np.poly1d(z)
                
                # Add trend line
                x_range = np.linspace(metric_df['Final_K'].min(), metric_df['Final_K'].max(), 100)
                plt.plot(x_range, p(x_range), '--', color='gray', alpha=0.7)
                
                # Add R² value
                from scipy import stats
                r_squared = stats.pearsonr(metric_df['Final_K'], metric_df['Value'])[0]**2
                plt.text(
                    0.05, 0.95, 
                    f'R² = {r_squared:.3f}', 
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top'
                )
            except:
                pass  # Skip if fitting fails
        
        plt.title(f'Impact of Quantization Bins (K) on {metric_name.upper()}')
        plt.xlabel('Final Number of Quantization Bins (K)')
        plt.ylabel(f'{metric_name.upper()} Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Quantization Type')
        plt.tight_layout()
        
        # Save plot
        metric_plot_path = os.path.join(output_dir, f'k_vs_{metric_name}.png')
        plt.savefig(metric_plot_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({f"quantization_impact/{metric_name}": wandb.Image(metric_plot_path)})
        
        plt.close()
    
    # Create bar plot comparing fixed vs adaptive for each metric
    for metric_name in df['Metric'].unique():
        metric_df = df[df['Metric'] == metric_name]
        
        # Group by quantization type and calculate mean
        grouped = metric_df.groupby('Quantization')['Value'].mean().reset_index()
        
        plt.figure(figsize=(8, 5))
        
        bars = plt.bar(
            grouped['Quantization'],
            grouped['Value'],
            color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(grouped)]
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01 * metric_df['Value'].max(),
                f'{height:.3f}',
                ha='center', 
                va='bottom', 
                fontsize=10
            )
        
        plt.title(f'Average {metric_name.upper()} by Quantization Strategy')
        plt.ylabel(f'{metric_name.upper()}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        avg_plot_path = os.path.join(output_dir, f'avg_{metric_name}_by_quant.png')
        plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({f"quantization_impact/avg_{metric_name}": wandb.Image(avg_plot_path)})
            
            # Create a wandb bar chart
            data = [[row['Quantization'], row['Value']] for _, row in grouped.iterrows()]
            table = wandb.Table(data=data, columns=["Quantization", metric_name])
            wandb.log({f"quantization_impact/{metric_name}_bar": wandb.plot.bar(
                table, "Quantization", metric_name,
                title=f"Average {metric_name.upper()} by Quantization Strategy")})
        
        plt.close()

def analyze_k_vs_entropy(curves_data: Dict, output_dir: str, use_wandb: bool = False):
    """
    Analyze the relationship between quantization levels (k) and entropy.
    
    Args:
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save output plots
        use_wandb: Whether to log results to wandb
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have both k and entropy data
    has_both = {exp_name: 'k' in curves and 'entropy' in curves 
                for exp_name, curves in curves_data.items()}
    
    valid_exps = [exp_name for exp_name, valid in has_both.items() if valid]
    
    if not valid_exps:
        print("No experiments with both k and entropy data.")
        return
    
    # Create scatter plot of k vs entropy for each experiment
    plt.figure(figsize=(12, 8))
    
    for exp_name in valid_exps:
        k_values = curves_data[exp_name]['k']
        entropy_values = curves_data[exp_name]['entropy']
        
        plt.scatter(
            k_values, 
            entropy_values,
            label=exp_name,
            alpha=0.7,
            s=50,
            color=get_color(exp_name)
        )
        
        # Add trend line if enough data points
        if len(k_values) > 5:
            try:
                z = np.polyfit(k_values, entropy_values, 1)
                p = np.poly1d(z)
                
                # Create points for the line
                x_range = np.linspace(min(k_values), max(k_values), 100)
                plt.plot(x_range, p(x_range), '--', color=get_color(exp_name), alpha=0.5)
            except:
                pass  # Skip if fitting fails
    
    plt.title('Relationship Between Quantization Bins (K) and Entropy')
    plt.xlabel('Number of Quantization Bins (K)')
    plt.ylabel('Entropy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Experiment')
    plt.tight_layout()
    
    # Save plot
    k_entropy_path = os.path.join(output_dir, 'k_vs_entropy.png')
    plt.savefig(k_entropy_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if enabled
    if use_wandb:
        wandb.log({"quantization_relationships/k_vs_entropy": wandb.Image(k_entropy_path)})
        
        # Create an interactive scatter plot for wandb
        k_entropy_data = []
        for exp_name in valid_exps:
            k_values = curves_data[exp_name]['k']
            entropy_values = curves_data[exp_name]['entropy']
            epochs = curves_data[exp_name]['epochs']
            
            for k, entropy, epoch in zip(k_values, entropy_values, epochs):
                k_entropy_data.append([exp_name, epoch, k, entropy])
        
        table = wandb.Table(data=k_entropy_data, columns=["Experiment", "Epoch", "K", "Entropy"])
        wandb.log({"quantization_relationships/k_entropy_scatter": wandb.plot.scatter(
            table, "K", "Entropy", "Experiment")})
    
    plt.close()
    
    # Create time series of both k and entropy
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    for exp_name in valid_exps:
        epochs = curves_data[exp_name]['epochs']
        k_values = curves_data[exp_name]['k']
        
        color = get_color(exp_name)
        ax1.plot(epochs, k_values, color=color, alpha=0.7, label=f"{exp_name} (K)")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Quantization Bins (K)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    
    for exp_name in valid_exps:
        epochs = curves_data[exp_name]['epochs']
        entropy_values = curves_data[exp_name]['entropy']
        
        color = get_color(exp_name)
        ax2.plot(epochs, entropy_values, color=color, linestyle='--', 
                alpha=0.7, label=f"{exp_name} (Entropy)")
    
    ax2.set_ylabel('Entropy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Evolution of K and Entropy During Training')
    plt.tight_layout()
    
    # Save plot
    evolution_path = os.path.join(output_dir, 'k_entropy_evolution.png')
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if enabled
    if use_wandb:
        wandb.log({"quantization_relationships/k_entropy_evolution": wandb.Image(evolution_path)})
    
    plt.close()

def analyze_adaptive_vs_fixed(metrics_data: Dict, curves_data: Dict, output_dir: str, use_wandb: bool = False):
    """
    Analyze and compare adaptive vs fixed quantization strategies.
    
    Args:
        metrics_data: Dictionary mapping experiment names to their metrics
        curves_data: Dictionary mapping experiment names to their training curves
        output_dir: Directory to save output plots
        use_wandb: Whether to log results to wandb
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize experiments
    adaptive_exps = [exp for exp in metrics_data.keys() if 'adaptive' in exp.lower()]
    fixed_exps = [exp for exp in metrics_data.keys() if 'fixed' in exp.lower()]
    
    if not adaptive_exps or not fixed_exps:
        print("Not enough data to compare adaptive vs fixed quantization.")
        return
    
    # Create comparison data
    comparison_data = []
    
    for exp_name, metrics in metrics_data.items():
        if 'overall' not in metrics:
            continue
            
        # Determine experiment category
        if 'adaptive' in exp_name.lower():
            category = 'Adaptive'
        elif 'fixed' in exp_name.lower():
            category = 'Fixed'
        else:
            continue  # Skip other experiment types
            
        # Extract initial k value from experiment name or curves
        k_initial = None
        for k in [5, 10, 20]:
            if f'k{k}' in exp_name.lower():
                k_initial = k
                break
                
        if k_initial is None and exp_name in curves_data and 'k' in curves_data[exp_name]:
            k_initial = curves_data[exp_name]['k'][0]
        
        if k_initial is None:
            continue  # Skip if we can't determine k_initial
            
        # Extract final k value from curves if available
        k_final = None
        if exp_name in curves_data and 'k' in curves_data[exp_name]:
            k_final = curves_data[exp_name]['k'][-1]
        else:
            k_final = k_initial  # Assume fixed k if not in curves
        
        # Extract metrics
        for metric_name, value in metrics['overall'].items():
            comparison_data.append({
                'Experiment': exp_name,
                'Category': category,
                'K_Initial': k_initial,
                'K_Final': k_final,
                'Metric': metric_name,
                'Value': value
            })
    
    if not comparison_data:
        print("No suitable data for adaptive vs fixed comparison.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Create bar plots comparing adaptive vs fixed for each initial k and metric
    for metric_name in df['Metric'].unique():
        metric_df = df[df['Metric'] == metric_name]
        
        # Group by initial k value and category
        grouped = metric_df.groupby(['K_Initial', 'Category'])['Value'].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar chart
        width = 0.35
        k_values = sorted(grouped['K_Initial'].unique())
        
        for i, category in enumerate(['Adaptive', 'Fixed']):
            if category not in grouped['Category'].values:
                continue
                
            category_data = grouped[grouped['Category'] == category]
            x_pos = np.arange(len(k_values)) + (i - 0.5) * width
            
            # Map each K_Initial to its position in x_pos
            indices = [k_values.index(k) for k in category_data['K_Initial']]
            heights = category_data['Value'].values
            
            bars = plt.bar(
                x_pos[indices], 
                heights,
                width, 
                label=category,
                color='#1f77b4' if category == 'Adaptive' else '#d62728'
            )
            
            # Add value labels
            for bar, height in zip(bars, heights):
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01 * metric_df['Value'].max(),
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=9
                )
        
        plt.xlabel('Initial K Value')
        plt.ylabel(f'{metric_name.upper()}')
        plt.title(f'Adaptive vs Fixed Quantization: {metric_name.upper()}')
        plt.xticks(np.arange(len(k_values)), k_values)
        plt.legend(title='Quantization Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        compare_path = os.path.join(output_dir, f'adaptive_vs_fixed_{metric_name}.png')
        plt.savefig(compare_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({f"adaptive_vs_fixed/{metric_name}": wandb.Image(compare_path)})
            
            # Create an interactive bar chart for wandb
            data = [[row['K_Initial'], row['Category'], row['Value']] 
                   for _, row in grouped.iterrows()]
            
            table = wandb.Table(data=data, columns=["Initial_K", "Quantization", metric_name])
            wandb.log({f"adaptive_vs_fixed/{metric_name}_interactive": wandb.plot.bar(
                table, 
                "Initial_K", 
                metric_name, 
                "Quantization")})
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Quantization analysis with W&B integration')
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/quantization_analysis',
                        help='Directory to save analysis plots')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiment folders to analyze (default: all in results_dir)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    parser.add_argument('--wandb_name', type=str, default='quantization_analysis',
                        help='W&B run name')
    parser.add_argument('--offline', action='store_true',
                        help='Run W&B in offline mode')
    
    args = parser.parse_args()
    
    # Initialize wandb if specified
    if args.use_wandb:
        try:
            # Set offline mode if requested
            if args.offline:
                os.environ["WANDB_MODE"] = "offline"
                print("Running W&B in offline mode")
                
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config={
                    "results_dir": args.results_dir,
                    "output_dir": args.output_dir,
                    "experiments": args.experiments
                }
            )
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            args.use_wandb = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of experiments to analyze
    if args.experiments is None:
        try:
            args.experiments = [d for d in os.listdir(args.results_dir) 
                              if os.path.isdir(os.path.join(args.results_dir, d))]
        except Exception as e:
            print(f"Error listing directory {args.results_dir}: {e}")
            return
    
    if not args.experiments:
        print(f"No experiments found in {args.results_dir}")
        return
        
    print(f"Analyzing experiments: {args.experiments}")
    
    # Load metrics and training curves data
    metrics_data = {}
    curves_data = {}
    
    for exp_name in args.experiments:
        exp_dir = os.path.join(args.results_dir, exp_name)
        if not os.path.exists(exp_dir):
            print(f"Warning: Experiment directory {exp_dir} not found, skipping.")
            continue
            
        metrics_file = os.path.join(exp_dir, 'evaluation', 'metrics.json')
        if os.path.exists(metrics_file):
            metrics_data[exp_name] = load_metrics(metrics_file)
        
        curves_file = os.path.join(exp_dir, 'logs', 'training_curves.json')
        if os.path.exists(curves_file):
            curves_data[exp_name] = load_training_curves(curves_file)
    
    if not metrics_data or not curves_data:
        print("No metrics or training curves data found.")
        if args.use_wandb:
            wandb.finish()
        return
    
    # Run analysis
    print("Analyzing quantization bin dynamics...")
    analyze_k_dynamics(curves_data, args.output_dir, args.use_wandb)
    
    print("Analyzing quantization vs performance metrics...")
    analyze_quantization_vs_metrics(metrics_data, curves_data, args.output_dir, args.use_wandb)
    
    print("Analyzing relationship between k and entropy...")
    analyze_k_vs_entropy(curves_data, args.output_dir, args.use_wandb)
    
    print("Comparing adaptive vs fixed quantization...")
    analyze_adaptive_vs_fixed(metrics_data, curves_data, args.output_dir, args.use_wandb)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Finish wandb run if active
    if args.use_wandb:
        print("Syncing results to W&B...")
        wandb.finish()

if __name__ == "__main__":
    main() 