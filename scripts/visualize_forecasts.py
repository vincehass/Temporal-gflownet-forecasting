#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize probabilistic forecasts from saved samples')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (defaults to results_dir/plots)')
    parser.add_argument('--num_series', type=int, default=5,
                        help='Number of time series to visualize')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--context_length', type=int, default=24,
                        help='Context length used in the model')
    parser.add_argument('--prediction_horizon', type=int, default=8,
                        help='Prediction horizon used in the model')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to draw for visualizing probability')
    return parser.parse_args()

def load_samples(path):
    """Load saved samples from numpy files."""
    try:
        # Load samples, actual values, and context data
        samples = np.load(os.path.join(path, 'evaluation', 'samples.npy'))
        actuals = np.load(os.path.join(path, 'evaluation', 'actuals.npy'))
        context = np.load(os.path.join(path, 'evaluation', 'context.npy'))
        return samples, actuals, context
    except Exception as e:
        print(f"Error loading samples: {e}")
        return None, None, None

def plot_forecast_with_samples(context, actuals, samples, series_idx, 
                              context_length, prediction_horizon, 
                              figsize, output_dir, title=None, dpi=300):
    """Plot a single time series forecast with sample paths."""
    plt.figure(figsize=tuple(figsize))
    
    # Setup the time axis
    t_context = np.arange(1, context_length + 1)
    t_forecast = np.arange(context_length + 1, context_length + prediction_horizon + 1)
    
    # Get data for this series
    series_context = context[series_idx]
    series_actuals = actuals[series_idx]
    series_samples = samples[:, series_idx, :]
    
    # Calculate mean and quantiles
    mean_forecast = np.mean(series_samples, axis=0)
    q10 = np.quantile(series_samples, 0.1, axis=0)
    q25 = np.quantile(series_samples, 0.25, axis=0)
    q75 = np.quantile(series_samples, 0.75, axis=0)
    q90 = np.quantile(series_samples, 0.9, axis=0)
    
    # Plot context (historical) data
    plt.plot(t_context, series_context, color='blue', label='Historical Data')
    
    # Plot actual future data
    plt.plot(t_forecast, series_actuals, color='black', linestyle='--', label='Actual Future')
    
    # Plot mean forecast
    plt.plot(t_forecast, mean_forecast, color='red', label='Mean Forecast')
    
    # Plot sample paths (with low alpha for transparency)
    num_paths = min(20, samples.shape[0])  # Limit the number of sample paths shown
    for i in range(num_paths):
        plt.plot(t_forecast, series_samples[i], color='gray', alpha=0.1)
    
    # Plot prediction intervals
    plt.fill_between(t_forecast, q10, q90, color='red', alpha=0.2, label='80% Interval')
    plt.fill_between(t_forecast, q25, q75, color='red', alpha=0.3, label='50% Interval')
    
    # Add vertical line to separate context from prediction
    plt.axvline(x=context_length, color='gray', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    else:
        plt.title(f'Series {series_idx+1}: Forecast with Probabilistic Samples')
    
    # Custom legend
    handles = [
        Line2D([0], [0], color='blue', label='Historical Data'),
        Line2D([0], [0], color='black', linestyle='--', label='Actual Future'),
        Line2D([0], [0], color='red', label='Mean Forecast'),
        Line2D([0], [0], color='gray', alpha=0.5, label='Sample Paths'),
        plt.Rectangle((0,0), 1, 1, color='red', alpha=0.2, label='80% Interval'),
        plt.Rectangle((0,0), 1, 1, color='red', alpha=0.3, label='50% Interval')
    ]
    labels = ['Historical Data', 'Actual Future', 'Mean Forecast', 'Sample Paths', '80% Interval', '50% Interval']
    plt.legend(handles=handles, labels=labels, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'forecast_series_{series_idx+1}.png'), dpi=dpi, bbox_inches='tight')
    
    plt.close()

def plot_all_series_overview(context, actuals, samples, num_series, 
                            context_length, prediction_horizon, 
                            figsize, output_dir, dpi=300):
    """Plot an overview of multiple time series forecasts."""
    # Determine the number of rows and columns for the subplot grid
    n_rows = int(np.ceil(num_series / 2))
    n_cols = min(2, num_series)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=tuple(figsize), sharex=True)
    if num_series == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Setup the time axis
    t_context = np.arange(1, context_length + 1)
    t_forecast = np.arange(context_length + 1, context_length + prediction_horizon + 1)
    
    for i in range(num_series):
        ax = axes[i]
        
        # Get data for this series
        series_context = context[i]
        series_actuals = actuals[i]
        series_samples = samples[:, i, :]
        
        # Calculate mean and quantiles
        mean_forecast = np.mean(series_samples, axis=0)
        q25 = np.quantile(series_samples, 0.25, axis=0)
        q75 = np.quantile(series_samples, 0.75, axis=0)
        
        # Plot context data
        ax.plot(t_context, series_context, color='blue')
        
        # Plot actual future data
        ax.plot(t_forecast, series_actuals, color='black', linestyle='--')
        
        # Plot mean forecast
        ax.plot(t_forecast, mean_forecast, color='red')
        
        # Plot prediction interval
        ax.fill_between(t_forecast, q25, q75, color='red', alpha=0.3)
        
        # Add vertical line to separate context from prediction
        ax.axvline(x=context_length, color='gray', linestyle='--')
        
        ax.set_title(f'Series {i+1}')
        ax.grid(True, alpha=0.3)
    
    # Add common legend for the whole figure
    handles = [
        Line2D([0], [0], color='blue', label='Historical Data'),
        Line2D([0], [0], color='black', linestyle='--', label='Actual Future'),
        Line2D([0], [0], color='red', label='Mean Forecast'),
        plt.Rectangle((0,0), 1, 1, color='red', alpha=0.3, label='50% Interval')
    ]
    labels = ['Historical Data', 'Actual Future', 'Mean Forecast', '50% Interval']
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))
    
    # Add common labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time Step', labelpad=10)
    plt.ylabel('Value', labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for the legend
    
    # Save the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'forecast_overview.png'), dpi=dpi, bbox_inches='tight')
    
    plt.close()

def plot_calibration_diagram(context, actuals, samples, 
                            figsize, output_dir, dpi=300):
    """Plot calibration diagram to assess probabilistic forecast quality."""
    plt.figure(figsize=tuple(figsize))
    
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
    
    # Plot the diagram
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(quantile_levels, empirical_freq, 'ro-', label='Model Calibration')
    
    plt.xlabel('Nominal Probability')
    plt.ylabel('Empirical Frequency')
    plt.title('Probabilistic Calibration Diagram')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'calibration_diagram.png'), dpi=dpi, bbox_inches='tight')
    
    plt.close()

def plot_sharpness_diagram(samples, prediction_horizon, 
                          figsize, output_dir, dpi=300):
    """Plot sharpness diagram showing the width of prediction intervals by horizon."""
    plt.figure(figsize=tuple(figsize))
    
    # Calculate prediction interval widths for each horizon
    interval_widths = []
    
    for h in range(prediction_horizon):
        # Get samples for this horizon across all series
        horizon_samples = samples[:, :, h].flatten()
        
        # Calculate interval width (75th - 25th percentile)
        q75 = np.percentile(horizon_samples, 75)
        q25 = np.percentile(horizon_samples, 25)
        width = q75 - q25
        
        interval_widths.append(width)
    
    # Plot the diagram
    horizons = np.arange(1, prediction_horizon + 1)
    plt.bar(horizons, interval_widths, color='skyblue')
    
    plt.xlabel('Forecast Horizon')
    plt.ylabel('50% Interval Width')
    plt.title('Forecast Sharpness by Horizon')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'sharpness_diagram.png'), dpi=dpi, bbox_inches='tight')
    
    plt.close()

def load_metrics(experiment_dir):
    """Load metrics from an experiment directory."""
    metrics_path = os.path.join(experiment_dir, 'evaluation', 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Warning: No metrics found in {experiment_dir}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_per_horizon_metrics_with_samples(metrics, samples, actuals, prediction_horizon,
                                         figsize, output_dir, dpi=300):
    """Plot per-horizon metrics alongside sample spread."""
    if 'per_horizon' not in metrics:
        print("No per-horizon metrics found")
        return
    
    plt.figure(figsize=tuple(figsize))
    
    # Extract per-horizon metrics
    horizons = np.arange(1, prediction_horizon + 1)
    
    # Calculate empirical standard deviation of samples for each horizon
    sample_std = np.std(samples, axis=0).mean(axis=0)
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=tuple(figsize))
    ax2 = ax1.twinx()
    
    # Plot metrics on the first y-axis
    for metric_name, values in metrics['per_horizon'].items():
        if len(values) != prediction_horizon:
            continue
        ax1.plot(horizons, values, 'o-', label=f'{metric_name.upper()}')
    
    # Plot sample standard deviation on the second y-axis
    ax2.plot(horizons, sample_std, 'rs--', label='Sample Std Dev')
    
    # Add labels and legend
    ax1.set_xlabel('Forecast Horizon')
    ax1.set_ylabel('Metric Value')
    ax2.set_ylabel('Sample Standard Deviation')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Forecast Performance and Uncertainty by Horizon')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'horizon_metrics_and_spread.png'), dpi=dpi, bbox_inches='tight')
    
    plt.close()

def create_experiment_summary(experiment_dir, output_dir, context, actuals, samples, metrics):
    """Create a visual summary of the experiment."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 6))
    
    # Extract experiment name
    exp_name = os.path.basename(os.path.normpath(experiment_dir))
    
    # Add title
    plt.text(0.5, 0.95, f"Experiment Summary: {exp_name}", 
             horizontalalignment='center', fontsize=16, fontweight='bold')
    
    # Add overall metrics
    y_pos = 0.85
    plt.text(0.5, y_pos, "Overall Metrics:", horizontalalignment='center', fontsize=14)
    y_pos -= 0.05
    
    for metric, value in metrics.get('overall', {}).items():
        plt.text(0.5, y_pos, f"{metric.upper()}: {value:.4f}", 
                 horizontalalignment='center', fontsize=12)
        y_pos -= 0.04
    
    # Add metadata if available
    y_pos -= 0.05
    plt.text(0.5, y_pos, "Configuration:", horizontalalignment='center', fontsize=14)
    y_pos -= 0.05
    
    for key, value in metrics.get('metadata', {}).get('config', {}).items():
        plt.text(0.5, y_pos, f"{key}: {value}", horizontalalignment='center', fontsize=12)
        y_pos -= 0.04
    
    # Add dataset stats
    y_pos -= 0.05
    plt.text(0.5, y_pos, "Dataset Statistics:", horizontalalignment='center', fontsize=14)
    y_pos -= 0.05
    
    plt.text(0.5, y_pos, f"Number of series: {context.shape[0]}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"Context length: {context.shape[1]}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"Prediction horizon: {actuals.shape[1]}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"Number of samples: {samples.shape[0]}", 
             horizontalalignment='center', fontsize=12)
    
    # Turn off axes
    plt.axis('off')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'experiment_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_arguments()
    
    # Default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load samples, actuals, and context
    samples, actuals, context = load_samples(args.results_dir)
    
    if samples is None or actuals is None or context is None:
        print(f"Failed to load data from {args.results_dir}")
        return
    
    # Load metrics
    metrics = load_metrics(args.results_dir)
    
    # Create experiment summary
    if metrics is not None:
        create_experiment_summary(args.results_dir, args.output_dir, context, actuals, samples, metrics)
    
    # Plot individual series
    num_series = min(args.num_series, context.shape[0])
    for i in range(num_series):
        plot_forecast_with_samples(
            context, actuals, samples, i, 
            args.context_length, args.prediction_horizon, 
            args.figsize, args.output_dir, dpi=args.dpi
        )
    
    # Plot overview of all series
    plot_all_series_overview(
        context, actuals, samples, num_series, 
        args.context_length, args.prediction_horizon, 
        args.figsize, args.output_dir, dpi=args.dpi
    )
    
    # Plot calibration diagram
    plot_calibration_diagram(
        context, actuals, samples, 
        args.figsize, args.output_dir, dpi=args.dpi
    )
    
    # Plot sharpness diagram
    plot_sharpness_diagram(
        samples, args.prediction_horizon, 
        args.figsize, args.output_dir, dpi=args.dpi
    )
    
    # Plot per-horizon metrics with sample spread
    if metrics is not None:
        plot_per_horizon_metrics_with_samples(
            metrics, samples, actuals, args.prediction_horizon,
            args.figsize, args.output_dir, dpi=args.dpi
        )
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 