#!/usr/bin/env python3
"""
Script to convert TensorBoard logs to W&B and integrate the advanced visualizations
from plot_ablation_results_with_tensorbaord.py into our W&B dashboards.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import wandb
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
COLORS = sns.color_palette("Set2", 10)

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization functions directly without using the script module
# from plot_ablation_results_with_tensorbaord import (
#     plot_tb_loss_visualization,
#     plot_ste_visualization,
#     plot_quantization_range_visualization,
#     plot_quantization_metrics,
#     draw_weighted_edge,
# )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert TensorBoard logs to W&B and integrate advanced visualizations')
    parser.add_argument('--results_dir', type=str, default='./results/ablations',
                        help='Directory containing experiment results with TensorBoard logs')
    parser.add_argument('--output_dir', type=str, default='./results/wandb_integration',
                        help='Directory to save local plots (if any)')
    parser.add_argument('--study_type', type=str, default='all',
                        help='Type of ablation study to process (e.g., "quantization", "entropy", "all")')
    parser.add_argument('--datasets', type=str, nargs='+', default=['electricity', 'traffic', 'ETTm1', 'ETTh1'],
                        help='Datasets to include in plots')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    
    return parser.parse_args()

def find_experiment_dirs(results_dir: str, study_type: str = 'all') -> List[str]:
    """Find all experiment directories for the given study type."""
    if study_type == 'all':
        pattern = os.path.join(results_dir, '*')
    else:
        pattern = os.path.join(results_dir, f'*{study_type}*')
    
    # Filter to only include directories
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return sorted(dirs)

def load_config(experiment_dir: str) -> Dict[str, Any]:
    """Load configuration for an experiment."""
    config_file = os.path.join(experiment_dir, 'config.yaml')
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}

def load_tensorboard_logs(experiment_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load tensorboard logs for a specific experiment."""
    log_dir = os.path.join(experiment_dir, 'logs')
    
    if not os.path.exists(log_dir):
        return {}
        
    # Find all event files recursively
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        return {}
    
    # Process all event files
    all_scalars = {}
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Extract scalar values
        scalar_tags = ea.Tags().get('scalars', [])
        
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            
            # If tag already exists, merge with existing data
            if tag in all_scalars:
                all_scalars[tag].extend([(event.step, event.value) for event in events])
            else:
                all_scalars[tag] = [(event.step, event.value) for event in events]
    
    # Sort by step for each tag
    for tag in all_scalars:
        all_scalars[tag] = sorted(all_scalars[tag], key=lambda x: x[0])
    
    return all_scalars

def extract_experiment_params(experiment_dir: str, config: Dict[str, Any] = None) -> Dict[str, str]:
    """Extract parameters from experiment directory name and config."""
    base_name = os.path.basename(experiment_dir)
    params = {}
    
    # Extract from directory name
    quant_match = re.search(r'(?:fixed|adaptive)_k(\d+)', base_name)
    if quant_match:
        params['quantization'] = 'fixed' if 'fixed' in base_name else 'adaptive'
        params['k_value'] = quant_match.group(1)
    
    # Extract policy type
    if 'uniform' in base_name:
        params['policy'] = 'uniform'
    elif 'learned' in base_name:
        params['policy'] = 'learned'
    
    # Extract from config if available
    if config:
        # Get dataset info
        if 'dataset' in config:
            if isinstance(config['dataset'], dict):
                params['dataset'] = config['dataset'].get('name', 'unknown')
            else:
                params['dataset'] = config['dataset']
        
        # Get quantization info
        if 'quantization' in config:
            quant_config = config['quantization']
            if 'adaptive' in quant_config:
                params['quantization'] = 'adaptive' if quant_config['adaptive'] else 'fixed'
            if 'k_initial' in quant_config:
                params['k_value'] = str(quant_config['k_initial'])
        
        # Get policy info
        if 'policy' in config and 'backward_policy_type' in config['policy']:
            params['policy'] = config['policy']['backward_policy_type']
        
        # Get entropy value
        if 'gfn' in config and 'lambda_entropy' in config['gfn']:
            params['entropy'] = str(config['gfn']['lambda_entropy'])
    
    # If nothing found, use the directory name
    if not params:
        params['experiment'] = base_name
    
    return params

def get_experiment_label(params: Dict[str, str]) -> str:
    """Generate a readable label for the experiment based on its parameters."""
    parts = []
    
    if 'quantization' in params:
        parts.append(f"{params['quantization'].capitalize()}")
        if 'k_value' in params:
            parts.append(f"k={params['k_value']}")
    
    if 'policy' in params:
        parts.append(f"Policy={params['policy']}")
    
    if 'entropy' in params:
        parts.append(f"λ={params['entropy']}")
    
    if parts:
        return " ".join(parts)
    elif 'experiment' in params:
        # Clean up the experiment name
        return params['experiment'].replace('_', ' ').title()
    else:
        return "Unknown"

def create_tb_loss_visualization() -> plt.Figure:
    """Generate visualization related to Trajectory Balance loss concept."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create simple graph nodes for state space
    nodes = {
        's0': (0, 0.5),    # Initial state
        's1a': (0.33, 0.8),
        's1b': (0.33, 0.5),
        's1c': (0.33, 0.2),
        's2a': (0.66, 0.9),
        's2b': (0.66, 0.7),
        's2c': (0.66, 0.5),
        's2d': (0.66, 0.3),
        's2e': (0.66, 0.1),
        'sT1': (1, 0.9),   # Terminal state 1
        'sT2': (1, 0.7),   # Terminal state 2
        'sT3': (1, 0.5),   # Terminal state 3
        'sT4': (1, 0.3),   # Terminal state 4
        'sT5': (1, 0.1),   # Terminal state 5
    }
    
    # Draw nodes
    for name, pos in nodes.items():
        if name == 's0':
            color = 'green'
            size = 1000
            label = 'Initial state $s_0$'
        elif name.startswith('sT'):
            color = 'red'
            size = 800
            label = f'Terminal state $s_T$' if name == 'sT1' else None
        else:
            color = 'skyblue'
            size = 600
            label = None
        
        ax.scatter(pos[0], pos[1], s=size, c=color, alpha=0.7, edgecolors='black', zorder=10)
        if label:
            ax.text(pos[0], pos[1], label, fontsize=10, ha='center', va='center')
    
    # Draw edges with different weights
    # Forward edges
    draw_weighted_edge(ax, nodes['s0'], nodes['s1a'], weight=0.5, color='blue', label='Forward\nFlow\n$P_F(τ)$')
    draw_weighted_edge(ax, nodes['s0'], nodes['s1b'], weight=0.3, color='blue')
    draw_weighted_edge(ax, nodes['s0'], nodes['s1c'], weight=0.2, color='blue')
    
    draw_weighted_edge(ax, nodes['s1a'], nodes['s2a'], weight=0.3, color='blue')
    draw_weighted_edge(ax, nodes['s1a'], nodes['s2b'], weight=0.2, color='blue')
    draw_weighted_edge(ax, nodes['s1b'], nodes['s2b'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s1b'], nodes['s2c'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s1b'], nodes['s2d'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s1c'], nodes['s2d'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s1c'], nodes['s2e'], weight=0.1, color='blue')
    
    draw_weighted_edge(ax, nodes['s2a'], nodes['sT1'], weight=0.3, color='blue')
    draw_weighted_edge(ax, nodes['s2b'], nodes['sT2'], weight=0.3, color='blue')
    draw_weighted_edge(ax, nodes['s2c'], nodes['sT3'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s2d'], nodes['sT4'], weight=0.1, color='blue')
    draw_weighted_edge(ax, nodes['s2e'], nodes['sT5'], weight=0.2, color='blue')
    
    # Backward edges
    draw_weighted_edge(ax, nodes['sT1'], nodes['s2a'], weight=0.3, color='red', linestyle='--', label='Backward\nFlow\n$P_B(τ|x)$')
    draw_weighted_edge(ax, nodes['sT2'], nodes['s2b'], weight=0.3, color='red', linestyle='--')
    draw_weighted_edge(ax, nodes['sT3'], nodes['s2c'], weight=0.1, color='red', linestyle='--')
    draw_weighted_edge(ax, nodes['sT4'], nodes['s2d'], weight=0.1, color='red', linestyle='--')
    draw_weighted_edge(ax, nodes['sT5'], nodes['s2e'], weight=0.2, color='red', linestyle='--')
    
    # Add explanatory text
    title = "Trajectory Balance: Flow Consistency in GFlowNets"
    ax.text(0.5, 1.05, title, fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    ax.text(0.5, -0.05, 
           "Trajectory Balance Loss ensures consistency between forward and backward flows:\n"
           "$L_{TB}(τ) = (\\log Z + \\sum_{t=0}^{T'-1} \\log P_F(s_{t+1}|s_t) - \\sum_{t=1}^{T'} \\log P_B(s_{t-1}|s_t) - \\log R(τ))^2$",
           fontsize=11, ha='center', va='center', transform=ax.transAxes)
    
    # Add reward explanation
    ax.text(1.05, 0.9, "$R(τ_1) = 0.9$", fontsize=10, ha='left', va='center')
    ax.text(1.05, 0.7, "$R(τ_2) = 0.7$", fontsize=10, ha='left', va='center')
    ax.text(1.05, 0.5, "$R(τ_3) = 0.5$", fontsize=10, ha='left', va='center')
    ax.text(1.05, 0.3, "$R(τ_4) = 0.3$", fontsize=10, ha='left', va='center')
    ax.text(1.05, 0.1, "$R(τ_5) = 0.1$", fontsize=10, ha='left', va='center')
    
    ax.text(1.15, 0.5, "Reward $R(τ)$", fontsize=12, ha='center', va='center', rotation=-90)
    
    ax.axis('off')
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    return fig

def create_ste_visualization() -> plt.Figure:
    """Generate visualization related to the Straight-Through Estimator (STE) mechanism."""
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Set up plot parameters
    k = 10  # Number of bins
    vmin, vmax = -10, 10  # Value range
    
    # Create example logits and probabilities
    logits = np.array([-3, -2, -1, 0, 1, 2, 1, 0, -1, -2])  # Example logits
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    bin_centers = np.linspace(vmin + (vmax-vmin)/(2*k), vmax - (vmax-vmin)/(2*k), k)
    
    # Create a shared axis for visual clarity
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
    
    # Probability distribution
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(np.arange(k), probs, alpha=0.7)
    ax1.set_title('Forward Policy Probabilities P(a_t = q_k|s_t)')
    ax1.set_xlabel('Bin Index k')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(np.arange(k))
    ax1.set_xticklabels([f'{i}' for i in range(k)])
    
    # Hard sample visualization
    hard_sample_idx = np.argmax(probs)  # Get the most likely bin
    ax2 = plt.subplot(gs[0, 1])
    ax2.bar(0, bin_centers[hard_sample_idx], color='green', width=0.5)
    ax2.set_title('Hard Sample')
    ax2.set_ylabel('Value')
    ax2.set_xticks([])
    ax2.text(0, bin_centers[hard_sample_idx]/2, f'q_{hard_sample_idx}', 
             ha='center', va='center', fontweight='bold')
    ax2.set_ylim(vmin, vmax)
    
    # Annotate for forward pass
    plt.annotate('', xy=(0, 0.5), xytext=(1, 0.5), 
                 xycoords=ax1.get_position(), textcoords=ax2.get_position(),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.annotate('argmax', xy=(0.5, 0.55), 
                 xycoords='figure fraction', ha='center')
    
    # Soft sample visualization
    ax3 = plt.subplot(gs[1, 0])
    # Expected value calculation using soft sample
    soft_sample = np.sum(probs * bin_centers)
    ax3.bar(np.arange(k), probs * bin_centers, alpha=0.7)
    ax3.set_title('Weighted Bin Values (Probability * Bin Value)')
    ax3.set_xlabel('Bin Index k')
    ax3.set_ylabel('Weighted Value')
    ax3.set_xticks(np.arange(k))
    ax3.set_xticklabels([f'{i}' for i in range(k)])
    
    # Expected value visualization
    ax4 = plt.subplot(gs[1, 1])
    ax4.bar(0, soft_sample, color='blue', width=0.5)
    ax4.set_title('Soft Sample')
    ax4.set_ylabel('Value')
    ax4.set_xticks([])
    ax4.text(0, soft_sample/2, f'E[q]', 
             ha='center', va='center', fontweight='bold')
    ax4.set_ylim(vmin, vmax)
    
    # Annotate for backward pass
    plt.annotate('', xy=(0, 0.5), xytext=(1, 0.5), 
                 xycoords=ax3.get_position(), textcoords=ax4.get_position(),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.annotate('sum', xy=(0.5, 0.25), 
                 xycoords='figure fraction', ha='center')
    
    # Annotate STE explanation
    plt.figtext(0.5, 0.02, 
               "Straight-Through Estimator (STE): For forward pass, use hard sample (argmax).\n"
               "For backward pass, gradient flows through soft sample (expectation).",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.suptitle('Straight-Through Estimator Mechanism', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

def create_adaptive_quantization_visualization() -> plt.Figure:
    """Generate visualization of adaptive quantization decision process."""
    # Create a grid of values for delta_R and H
    delta_R = np.linspace(0, 0.05, 100)  # Reward improvement
    H = np.linspace(0, 1, 100)  # Normalized entropy
    
    # Create meshgrid
    Delta_R, H_mesh = np.meshgrid(delta_R, H)
    
    # Set adaptation parameters
    epsilon = 0.02  # Reward improvement threshold
    lambda_adapt = 0.5  # Adaptation sensitivity
    
    # Calculate the adaptive update factor: η_e = 1 + λ * (max(0, ε - ΔR_e)/ε + (1 - H_e))
    improvement_signal = np.maximum(0, epsilon - Delta_R) / epsilon
    confidence_signal = 1 - H_mesh
    adaptive_factor = 1 + lambda_adapt * (improvement_signal + confidence_signal)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(Delta_R, H_mesh, adaptive_factor, 20, cmap='viridis')
    plt.colorbar(contour, label='Adaptive Update Factor (η_e)')
    
    # Draw the decision boundary for η_e = 1 (no change in K)
    ax.contour(Delta_R, H_mesh, adaptive_factor, levels=[1], colors='red', linestyles='dashed', linewidths=2)
    
    # Add labels
    ax.set_xlabel('Reward Improvement (ΔR_e)')
    ax.set_ylabel('Normalized Entropy (H_e)')
    ax.set_title('Adaptive Quantization Decision Boundaries')
    
    # Add annotation for decision regions
    ax.text(0.01, 0.2, 'Decrease K\n(η_e < 1)', color='white', fontsize=12, 
           ha='left', va='center', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(0.04, 0.8, 'Increase K\n(η_e > 1)', color='black', fontsize=12, 
           ha='left', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return fig

def draw_weighted_edge(ax, start, end, weight=1.0, color='black', linestyle='-', label=None):
    """Helper function to draw a weighted edge between nodes"""
    # Determine line width based on weight
    lw = weight * 3
    
    # Draw the line
    line = ax.plot([start[0], end[0]], [start[1], end[1]], 
                    color=color, linewidth=lw, linestyle=linestyle, zorder=5,
                    label=label if label else "")
    
    # Add weight label if significant
    if weight >= 0.2:
        # Position the label at the middle of the line
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Slight offset to avoid overlapping with the line
        offset = 0.02
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.arctan2(dy, dx)
        offset_x = -offset * np.sin(angle)
        offset_y = offset * np.cos(angle)
        
        ax.text(mid_x + offset_x, mid_y + offset_y, f"{weight:.1f}", 
                fontsize=8, ha='center', va='center', color=color,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
    
    return line

def convert_to_wandb(experiment_dir: str, wandb_project: str, wandb_entity: str) -> None:
    """Convert TensorBoard logs to W&B and add advanced visualizations."""
    # Load config
    config = load_config(experiment_dir)
    
    # Extract experiment parameters
    params = extract_experiment_params(experiment_dir, config)
    experiment_label = get_experiment_label(params)
    
    # Load TensorBoard logs
    logs = load_tensorboard_logs(experiment_dir)
    
    if not logs:
        print(f"No TensorBoard logs found for {experiment_dir}")
        return
    
    # Initialize W&B run
    run_name = f"{os.path.basename(experiment_dir)}"
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={**config, **params},
        job_type="tensorboard_integration",
        tags=["tensorboard_integrated"],
        reinit=True
    )
    
    try:
        # Log TensorBoard metrics to W&B
        steps_logged = set()
        for tag, values in logs.items():
            for step, value in values:
                if step not in steps_logged:
                    wandb.log({"step": step}, step=step)
                    steps_logged.add(step)
                wandb.log({tag: value}, step=step)
        
        # Create specialized visualizations based on experiment type
        # 1. TB Loss visualization
        fig = create_tb_loss_visualization()
        wandb.log({"tb_loss/concept": wandb.Image(fig)})
        plt.close(fig)
        
        # 2. STE visualization
        fig = create_ste_visualization()
        wandb.log({"ste/mechanism": wandb.Image(fig)})
        plt.close(fig)
        
        # 3. Adaptive quantization visualization (if applicable)
        if 'quantization' in params and params['quantization'] == 'adaptive':
            fig = create_adaptive_quantization_visualization()
            wandb.log({"quantization/adaptive_mechanism": wandb.Image(fig)})
            plt.close(fig)
        
        # Create artifact with experiment data
        artifact = wandb.Artifact(name=f"tensorboard_integration_{run_name}", type="experiment")
        
        # Add config file
        config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(config_path):
            artifact.add_file(config_path, name="config.yaml")
        
        # Add metrics files from evaluation directory
        eval_dir = os.path.join(experiment_dir, "evaluation")
        if os.path.exists(eval_dir):
            metrics_files = glob.glob(os.path.join(eval_dir, "*.json"))
            for metrics_file in metrics_files:
                artifact.add_file(metrics_file, name=f"metrics/{os.path.basename(metrics_file)}")
        
        # Log the artifact
        wandb.log_artifact(artifact)
        
        print(f"Successfully integrated TensorBoard data with W&B for {experiment_dir}")
        
    finally:
        wandb.finish()

def process_all_experiments(args: argparse.Namespace) -> None:
    """Process all experiments in the results directory."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.results_dir, args.study_type)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {args.results_dir} for study type '{args.study_type}'")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Process each experiment
    for experiment_dir in experiment_dirs:
        print(f"\nProcessing {os.path.basename(experiment_dir)}...")
        convert_to_wandb(experiment_dir, args.wandb_project, args.wandb_entity)
    
    print("\nAll experiments processed.")
    print(f"Check W&B dashboard at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    process_all_experiments(args)

if __name__ == "__main__":
    main() 