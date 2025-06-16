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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate samples based on metrics for visualization')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (defaults to results_dir/samples)')
    parser.add_argument('--num_series', type=int, default=5,
                        help='Number of time series to visualize')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--context_length', type=int, default=24,
                        help='Context length for simulation')
    parser.add_argument('--prediction_horizon', type=int, default=8,
                        help='Prediction horizon for simulation')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_metrics(metrics_path):
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def simulate_context_data(num_series, context_length, seed=42):
    """Simulate context data for visualization purposes."""
    np.random.seed(seed)
    
    # Generate synthetic time series
    contexts = []
    for i in range(num_series):
        # Create time series with AR(1) process
        phi = 0.9
        noise_level = 0.1
        x = np.zeros(context_length)
        x[0] = np.random.normal(0, 1)
        for t in range(1, context_length):
            x[t] = phi * x[t-1] + np.random.normal(0, noise_level)
        contexts.append(x)
    
    return np.array(contexts)

def simulate_actual_values(contexts, metrics, prediction_horizon, seed=42):
    """Simulate actual values based on context and metrics."""
    np.random.seed(seed)
    
    num_series = contexts.shape[0]
    actuals = []
    
    # Get per-horizon metrics for scaling
    per_horizon_metrics = metrics.get('per_horizon', {})
    mase_by_horizon = per_horizon_metrics.get('mase', [1.0] * prediction_horizon)
    
    for i in range(num_series):
        # The actual values should depend on the last context value
        # but also have a structure informed by metrics
        last_context = contexts[i, -1]
        
        # Create a forecast that has increasing error with horizon
        forecast = np.zeros(prediction_horizon)
        forecast[0] = last_context  # Start from last context value
        
        # Add trend and errors that scale with horizon
        for h in range(prediction_horizon):
            # Add a slight trend
            if h > 0:
                forecast[h] = forecast[h-1] + 0.1
            
            # Add error that increases with horizon
            error_scale = mase_by_horizon[h] * 0.05 
            forecast[h] += np.random.normal(0, error_scale)
        
        actuals.append(forecast)
    
    return np.array(actuals)

def simulate_samples(contexts, actuals, metrics, num_samples, prediction_horizon, seed=42):
    """Simulate forecast samples based on actuals and metrics."""
    np.random.seed(seed)
    
    num_series = contexts.shape[0]
    
    # Get per-horizon metrics for scaling
    per_horizon_metrics = metrics.get('per_horizon', {})
    mase_by_horizon = per_horizon_metrics.get('mase', [1.0] * prediction_horizon)
    
    # Get per-series metrics for scaling
    per_series_metrics = metrics.get('per_series', {})
    mase_by_series = per_series_metrics.get('mase', [1.0] * num_series)
    
    # Generate samples for each series
    all_samples = []
    
    for s in range(num_samples):
        samples_for_series = []
        
        for i in range(num_series):
            # Get the actual values for this series
            actual = actuals[i]
            
            # Create samples that have errors dependent on:
            # 1. The series noise level (based on per-series MASE)
            # 2. The horizon (based on per-horizon MASE)
            sample = np.zeros(prediction_horizon)
            
            for h in range(prediction_horizon):
                # Error scale depends on both series and horizon
                error_scale = mase_by_series[i % len(mase_by_series)] * mase_by_horizon[h] * 0.1
                
                # Sample from actual with error
                sample[h] = actual[h] + np.random.normal(0, error_scale)
            
            samples_for_series.append(sample)
        
        all_samples.append(samples_for_series)
    
    # Shape: (num_samples, num_series, prediction_horizon)
    return np.array(all_samples)

def save_simulated_data(samples, contexts, actuals, output_dir):
    """Save simulated data to disk."""
    os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
    
    np.save(os.path.join(output_dir, 'evaluation', 'samples.npy'), samples)
    np.save(os.path.join(output_dir, 'evaluation', 'context.npy'), contexts)
    np.save(os.path.join(output_dir, 'evaluation', 'actuals.npy'), actuals)
    
    print(f"Saved simulated data to {os.path.join(output_dir, 'evaluation')}")

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

def plot_calibration_diagram(actuals, samples, figsize, output_dir, dpi=300):
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

def plot_sharpness_diagram(samples, prediction_horizon, figsize, output_dir, dpi=300):
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

def create_experiment_summary(experiment_dir, output_dir, config, metrics, samples, contexts, actuals):
    """Create a visual summary of the experiment."""
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
    
    # Add configuration details
    y_pos -= 0.05
    plt.text(0.5, y_pos, "Configuration:", horizontalalignment='center', fontsize=14)
    y_pos -= 0.05
    
    # Add metadata if available
    if 'metadata' in metrics and 'config' in metrics['metadata']:
        for key, value in metrics['metadata']['config'].items():
            plt.text(0.5, y_pos, f"{key}: {value}", 
                    horizontalalignment='center', fontsize=12)
            y_pos -= 0.04
    
    # Add dataset stats
    y_pos -= 0.05
    plt.text(0.5, y_pos, "Dataset Statistics:", horizontalalignment='center', fontsize=14)
    y_pos -= 0.05
    
    plt.text(0.5, y_pos, f"Number of series: {contexts.shape[0]}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"Context length: {contexts.shape[1]}", 
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
        args.output_dir = os.path.join(args.results_dir, 'samples')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Load config
    try:
        config_path = os.path.join(args.results_dir, 'config.yaml')
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        config = {}
    
    # Load metrics
    try:
        metrics_path = os.path.join(args.results_dir, 'evaluation', 'metrics.json')
        metrics = load_metrics(metrics_path)
        print(f"Loaded metrics from {metrics_path}")
    except Exception as e:
        print(f"Error: Could not load metrics: {e}")
        return
    
    # Get context length and prediction horizon from config or args
    context_length = config.get('dataset', {}).get('context_length', args.context_length)
    prediction_horizon = config.get('dataset', {}).get('prediction_horizon', args.prediction_horizon)
    
    print(f"Using context_length={context_length}, prediction_horizon={prediction_horizon}")
    
    # Simulate data
    contexts = simulate_context_data(args.num_series, context_length)
    print(f"Generated context data: {contexts.shape}")
    
    actuals = simulate_actual_values(contexts, metrics, prediction_horizon)
    print(f"Generated actual values: {actuals.shape}")
    
    samples = simulate_samples(contexts, actuals, metrics, args.num_samples, prediction_horizon)
    print(f"Generated samples: {samples.shape}")
    
    # Save data
    save_simulated_data(samples, contexts, actuals, args.output_dir)
    
    # Create plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    
    # Create experiment summary
    create_experiment_summary(args.results_dir, plots_dir, config, metrics, samples, contexts, actuals)
    
    # Plot individual series
    for i in range(args.num_series):
        plot_forecast_with_samples(
            contexts, actuals, samples, i, 
            context_length, prediction_horizon, 
            args.figsize, plots_dir, dpi=args.dpi
        )
    
    # Plot overview of all series
    plot_all_series_overview(
        contexts, actuals, samples, args.num_series, 
        context_length, prediction_horizon, 
        args.figsize, plots_dir, dpi=args.dpi
    )
    
    # Plot calibration diagram
    plot_calibration_diagram(
        actuals, samples, 
        args.figsize, plots_dir, dpi=args.dpi
    )
    
    # Plot sharpness diagram
    plot_sharpness_diagram(
        samples, prediction_horizon, 
        args.figsize, plots_dir, dpi=args.dpi
    )
    
    print(f"All plots saved to {plots_dir}")

if __name__ == "__main__":
    main() 