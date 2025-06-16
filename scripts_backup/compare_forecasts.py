#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import yaml
from glob import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare forecasts across different experiments')
    parser.add_argument('--results_dirs', nargs='+', type=str, required=True,
                        help='Directories containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='results/forecast_comparison',
                        help='Directory to save comparison plots')
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 10],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--series_indices', nargs='+', type=int, default=[0, 1, 2],
                        help='Indices of series to compare (default: 0 1 2)')
    return parser.parse_args()

def load_data(results_dir):
    """Load samples, context, and actuals from a results directory."""
    samples_path = os.path.join(results_dir, 'samples', 'evaluation', 'samples.npy')
    context_path = os.path.join(results_dir, 'samples', 'evaluation', 'context.npy')
    actuals_path = os.path.join(results_dir, 'samples', 'evaluation', 'actuals.npy')
    
    # Check if files exist
    if not (os.path.exists(samples_path) and 
            os.path.exists(context_path) and 
            os.path.exists(actuals_path)):
        return None, None, None
    
    # Load data
    samples = np.load(samples_path)
    context = np.load(context_path)
    actuals = np.load(actuals_path)
    
    return samples, context, actuals

def load_metrics(results_dir):
    """Load metrics from a results directory."""
    metrics_path = os.path.join(results_dir, 'evaluation', 'metrics.json')
    
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def get_experiment_label(exp_dir):
    """Extract a readable label from the experiment directory."""
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    
    # Parse experiment name
    if 'learned_policy' in exp_name:
        return 'Learned Policy'
    elif 'adaptive' in exp_name and 'k' in exp_name:
        k_value = exp_name.split('k')[-1]
        return f'Adaptive K={k_value}'
    elif 'fixed' in exp_name and 'k' in exp_name:
        k_value = exp_name.split('k')[-1]
        return f'Fixed K={k_value}'
    
    return exp_name

def compare_series_forecasts(results_dirs, series_idx, figsize, output_dir, dpi=300):
    """Compare forecasts for a single series across different experiments."""
    plt.figure(figsize=tuple(figsize))
    
    # Track if we have any data
    has_data = False
    
    # Use same context and actuals for all experiments
    context = None
    actuals = None
    
    # Setup colors for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))
    
    for i, results_dir in enumerate(results_dirs):
        samples, ctx, act = load_data(results_dir)
        
        if samples is None:
            continue
        
        has_data = True
        
        # Use the first valid experiment's context and actuals
        if context is None:
            context = ctx
            actuals = act
        
        # Get experiment label
        exp_label = get_experiment_label(results_dir)
        
        # Get data for this series
        if series_idx >= samples.shape[1]:
            continue
        
        series_samples = samples[:, series_idx, :]
        
        # Calculate mean and quantiles
        mean_forecast = np.mean(series_samples, axis=0)
        q25 = np.quantile(series_samples, 0.25, axis=0)
        q75 = np.quantile(series_samples, 0.75, axis=0)
        
        # Setup the time axis
        context_length = context.shape[1]
        prediction_horizon = actuals.shape[1]
        t_forecast = np.arange(context_length + 1, context_length + prediction_horizon + 1)
        
        # Plot mean forecast
        plt.plot(t_forecast, mean_forecast, color=colors[i], linewidth=2, 
                 label=f'{exp_label} (Mean)')
        
        # Plot prediction interval
        plt.fill_between(t_forecast, q25, q75, color=colors[i], alpha=0.2,
                         label=f'{exp_label} (50% Interval)')
    
    if not has_data:
        print(f"No data found for series {series_idx}")
        plt.close()
        return
    
    # Plot context and actuals
    series_context = context[series_idx]
    series_actuals = actuals[series_idx]
    
    context_length = context.shape[1]
    t_context = np.arange(1, context_length + 1)
    t_forecast = np.arange(context_length + 1, context_length + prediction_horizon + 1)
    
    plt.plot(t_context, series_context, 'k-', linewidth=2, label='Historical Data')
    plt.plot(t_forecast, series_actuals, 'k--', linewidth=2, label='Actual Future')
    
    # Add vertical line to separate context from prediction
    plt.axvline(x=context_length, color='gray', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Series {series_idx+1}: Forecast Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'series_{series_idx+1}_comparison.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def compare_metrics(results_dirs, figsize, output_dir, dpi=300):
    """Compare metrics across different experiments."""
    metrics_data = []
    
    for results_dir in results_dirs:
        metrics = load_metrics(results_dir)
        
        if metrics is None:
            continue
        
        exp_label = get_experiment_label(results_dir)
        
        # Extract overall metrics
        overall_metrics = metrics.get('overall', {})
        
        # Add to metrics data
        metrics_data.append({
            'Experiment': exp_label,
            **overall_metrics
        })
    
    if not metrics_data:
        print("No metrics found")
        return
    
    # Create metrics comparison plot
    plt.figure(figsize=tuple(figsize))
    
    # Get metric names
    metric_names = set()
    for data in metrics_data:
        for key in data.keys():
            if key != 'Experiment':
                metric_names.add(key)
    
    metric_names = sorted(list(metric_names))
    
    # Number of metrics and experiments
    n_metrics = len(metric_names)
    n_experiments = len(metrics_data)
    
    # Setup bar positions
    bar_width = 0.8 / n_experiments
    positions = np.arange(n_metrics)
    
    # Plot bars for each experiment
    for i, data in enumerate(metrics_data):
        values = [data.get(metric, 0) for metric in metric_names]
        x_pos = positions + (i - n_experiments/2 + 0.5) * bar_width
        
        plt.bar(x_pos, values, width=bar_width, label=data['Experiment'])
    
    # Add labels and title
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Metrics Comparison Across Experiments', fontsize=14)
    plt.xticks(positions, [m.upper() for m in metric_names])
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def compare_calibration(results_dirs, figsize, output_dir, dpi=300):
    """Compare calibration diagrams across different experiments."""
    plt.figure(figsize=tuple(figsize))
    
    # Ideal calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Setup colors for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dirs)))
    
    # Track if we have any data
    has_data = False
    
    for i, results_dir in enumerate(results_dirs):
        samples, context, actuals = load_data(results_dir)
        
        if samples is None:
            continue
        
        has_data = True
        
        # Get experiment label
        exp_label = get_experiment_label(results_dir)
        
        # Define quantile levels to evaluate
        quantile_levels = np.linspace(0.05, 0.95, 19)  # From 5% to 95% in steps of 5%
        
        # Calculate empirical frequencies
        empirical_freq = []
        for q in quantile_levels:
            # Calculate quantile forecast for each series and time step
            forecast_quantiles = np.quantile(samples, q, axis=0)
            
            # Count how often actual values are below the forecast quantile
            below_quantile = (actuals < forecast_quantiles).flatten()
            
            # Calculate empirical frequency
            freq = np.mean(below_quantile)
            empirical_freq.append(freq)
        
        # Plot calibration curve
        plt.plot(quantile_levels, empirical_freq, 'o-', color=colors[i], 
                 label=exp_label)
    
    if not has_data:
        print("No data found for calibration comparison")
        plt.close()
        return
    
    # Add labels and title
    plt.xlabel('Nominal Probability', fontsize=12)
    plt.ylabel('Empirical Frequency', fontsize=12)
    plt.title('Calibration Comparison Across Experiments', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'calibration_comparison.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def compare_sharpness(results_dirs, figsize, output_dir, dpi=300):
    """Compare sharpness diagrams across different experiments."""
    plt.figure(figsize=tuple(figsize))
    
    # Track if we have any data
    has_data = False
    
    # Get prediction horizon from first valid experiment
    prediction_horizon = None
    
    # Setup bar positions
    width_per_exp = 0.8  # width allocated per experiment for each horizon
    
    for results_dir in results_dirs:
        samples, _, _ = load_data(results_dir)
        
        if samples is not None and prediction_horizon is None:
            prediction_horizon = samples.shape[2]
            break
    
    if prediction_horizon is None:
        print("No data found for sharpness comparison")
        plt.close()
        return
    
    # Calculate positions for bars
    n_experiments = len(results_dirs)
    bar_width = width_per_exp / n_experiments
    
    for i, results_dir in enumerate(results_dirs):
        samples, _, _ = load_data(results_dir)
        
        if samples is None:
            continue
        
        has_data = True
        
        # Get experiment label
        exp_label = get_experiment_label(results_dir)
        
        # Calculate interval widths for each horizon
        interval_widths = []
        
        for h in range(prediction_horizon):
            # Get samples for this horizon across all series
            horizon_samples = samples[:, :, h].flatten()
            
            # Calculate interval width (75th - 25th percentile)
            q75 = np.percentile(horizon_samples, 75)
            q25 = np.percentile(horizon_samples, 25)
            width = q75 - q25
            
            interval_widths.append(width)
        
        # Calculate position for this experiment's bars
        positions = np.arange(prediction_horizon) + 1  # horizons start at 1
        bar_positions = positions + (i - n_experiments/2 + 0.5) * bar_width
        
        # Plot bars
        plt.bar(bar_positions, interval_widths, width=bar_width, 
                label=exp_label)
    
    if not has_data:
        print("No data found for sharpness comparison")
        plt.close()
        return
    
    # Add labels and title
    plt.xlabel('Forecast Horizon', fontsize=12)
    plt.ylabel('50% Interval Width', fontsize=12)
    plt.title('Sharpness Comparison Across Experiments', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sharpness_comparison.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def create_summary_table(results_dirs, output_dir):
    """Create a summary table of metrics across experiments."""
    # Collect metrics data
    table_data = []
    
    for results_dir in results_dirs:
        metrics = load_metrics(results_dir)
        
        if metrics is None:
            continue
        
        exp_label = get_experiment_label(results_dir)
        
        # Extract overall metrics
        overall_metrics = metrics.get('overall', {})
        
        # Extract metadata
        metadata = metrics.get('metadata', {}).get('config', {})
        
        # Combine data
        row = {'Experiment': exp_label, **overall_metrics, **metadata}
        table_data.append(row)
    
    if not table_data:
        print("No metrics found for summary table")
        return
    
    # Convert to Markdown table
    headers = []
    for row in table_data:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    
    # Ensure 'Experiment' is the first column
    if 'Experiment' in headers:
        headers.remove('Experiment')
        headers = ['Experiment'] + headers
    
    markdown = "# Experiment Summary\n\n"
    markdown += "## Metrics Comparison\n\n"
    
    # Add table header
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add table rows
    for row in table_data:
        values = []
        for h in headers:
            if h in row:
                if isinstance(row[h], float):
                    values.append(f"{row[h]:.4f}")
                else:
                    values.append(str(row[h]))
            else:
                values.append("")
        markdown += "| " + " | ".join(values) + " |\n"
    
    # Save as markdown file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(markdown)
    
    print(f"Saved summary table to {os.path.join(output_dir, 'summary.md')}")

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare series forecasts
    for series_idx in args.series_indices:
        compare_series_forecasts(args.results_dirs, series_idx, 
                               args.figsize, args.output_dir, args.dpi)
    
    # Compare metrics
    compare_metrics(args.results_dirs, args.figsize, args.output_dir, args.dpi)
    
    # Compare calibration
    compare_calibration(args.results_dirs, args.figsize, args.output_dir, args.dpi)
    
    # Compare sharpness
    compare_sharpness(args.results_dirs, args.figsize, args.output_dir, args.dpi)
    
    # Create summary table
    create_summary_table(args.results_dirs, args.output_dir)
    
    print(f"All comparison plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 