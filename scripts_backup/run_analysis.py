#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run analysis on experimental results using the consolidated plotting script.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run analysis on experiment results')
    parser.add_argument('--results_dir', type=str, default='results/synthetic_data',
                        help='Directory containing the experiment results')
    parser.add_argument('--output_dir', type=str, default='results/analysis_output',
                        help='Directory to save the analysis plots')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter experiments by dataset')
    parser.add_argument('--experiment_type', type=str, default=None, 
                        choices=['fixed', 'adaptive', 'learned_policy'],
                        help='Filter experiments by type')
    parser.add_argument('--metrics', type=str, nargs='+', default=['wql', 'crps', 'mase'],
                        help='Metrics to analyze')
    parser.add_argument('--plot_style', type=str, default='seaborn-v0_8-darkgrid',
                        help='Matplotlib style to use')
    parser.add_argument('--colormap', type=str, default='viridis',
                        help='Colormap to use')
    
    return parser.parse_args()

def run_analysis(args):
    """Run the analysis using the plot_consolidated_results.py script."""
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        'python', 'scripts/plot_consolidated_results.py',
        '--results_dir', args.results_dir,
        '--output_dir', args.output_dir,
        '--style', args.plot_style,
        '--colormap', args.colormap
    ]
    
    # Add optional arguments if provided
    if args.dataset:
        cmd.extend(['--dataset_filter', args.dataset])
    
    if args.experiment_type:
        cmd.extend(['--experiment_type', args.experiment_type])
    
    if args.metrics:
        cmd.extend(['--metrics'] + args.metrics)
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(result.stdout)
    
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    # Print summary of what was done
    print(f"\nAnalysis completed.")
    print(f"- Results analyzed from: {args.results_dir}")
    print(f"- Plots saved to: {args.output_dir}")
    
    # List the generated plots
    plots = list(Path(args.output_dir).glob('*.png'))
    print(f"\nGenerated {len(plots)} plots:")
    for plot in plots:
        print(f"- {plot.name}")

def main():
    args = parse_arguments()
    run_analysis(args)

if __name__ == "__main__":
    main() 