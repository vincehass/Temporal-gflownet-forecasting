#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from tqdm import tqdm
import yaml

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.temporal_gfn.data.dataset import TimeSeriesDataset, SyntheticTimeSeriesDataset
from src.temporal_gfn.model.model import TemporalGFlowNet
from src.temporal_gfn.data.dataset import create_dataloader
from src.temporal_gfn.utils.metrics import calculate_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate and visualize samples from a trained model')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (defaults to results_dir/samples)')
    parser.add_argument('--checkpoint', type=str, default='best',
                        help='Checkpoint to use (default: best)')
    parser.add_argument('--num_series', type=int, default=5,
                        help='Number of time series to visualize')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                        help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dataset(config, split='test'):
    """Create dataset based on configuration."""
    dataset_config = config['dataset']
    
    if dataset_config['type'] == 'synthetic':
        # For synthetic data
        dataset = SyntheticTimeSeriesDataset(
            num_series=dataset_config.get('num_series', 100),
            series_length=dataset_config.get('series_length', 200),
            context_length=dataset_config['context_length'],
            prediction_horizon=dataset_config['prediction_horizon'],
            model_type=dataset_config.get('model_type', 'ar'),
            model_params=dataset_config.get('model_params', None),
            noise_level=dataset_config.get('noise_level', 0.1),
            scaler_type=dataset_config.get('scaler_type', 'mean'),
            stride=dataset_config.get('stride', 1),
            sample_stride=dataset_config.get('sample_stride', 1),
            return_indices=True,  # Always return indices for evaluation
            seed=dataset_config.get('seed', None),
        )
    else:
        # For real data
        data_path = dataset_config['path']
        if split == 'train':
            data_path = dataset_config.get('train_path', data_path)
        elif split == 'val':
            data_path = dataset_config.get('val_path', data_path)
        elif split == 'test':
            data_path = dataset_config.get('test_path', data_path)
        
        # Load data based on format
        if data_path.endswith('.csv'):
            import pandas as pd
            time_series = pd.read_csv(data_path)
        elif data_path.endswith('.npy'):
            import numpy as np
            time_series = np.load(data_path)
        elif data_path.endswith('.pt') or data_path.endswith('.pth'):
            time_series = torch.load(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Create dataset
        dataset = TimeSeriesDataset(
            time_series=time_series,
            context_length=dataset_config['context_length'],
            prediction_horizon=dataset_config['prediction_horizon'],
            scaler_type=dataset_config.get('scaler_type', 'mean'),
            stride=dataset_config.get('stride', 1),
            sample_stride=dataset_config.get('sample_stride', 1),
            return_indices=True,  # Always return indices for evaluation
            forecast_mode=(split == 'test'),
        )
    
    return dataset

def create_dataloaders(dataset, config):
    """Create dataloader from dataset."""
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 0)
    
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

def load_model(results_dir, checkpoint='best'):
    """Load a trained model from the results directory."""
    # Load config
    config_path = os.path.join(results_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Create model
    model = TemporalGFlowNet(config)
    
    # Load checkpoint
    if checkpoint == 'best':
        checkpoint_path = os.path.join(results_dir, 'checkpoints', 'best_model.pt')
    else:
        checkpoint_path = os.path.join(results_dir, 'checkpoints', f'checkpoint_{checkpoint}.pt')
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def generate_samples(model, test_loader, num_samples, device='cpu'):
    """Generate samples from the model for each test series."""
    all_samples = []
    all_contexts = []
    all_actuals = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating samples")):
            # Unpack the batch
            if isinstance(batch, dict):
                x = batch['context']
                y = batch['target']
                indices = batch.get('index', torch.arange(len(x)) + batch_idx * test_loader.batch_size)
            else:
                x, y = batch
                indices = torch.arange(len(x)) + batch_idx * test_loader.batch_size
            
            x = x.to(device)
            y = y.to(device)
            
            # Store context, actuals, and indices
            all_contexts.append(x.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
            
            # Generate samples
            batch_samples = []
            for _ in range(num_samples):
                sample = model.sample(x)
                batch_samples.append(sample.cpu().numpy())
            
            # Stack samples for this batch
            batch_samples = np.stack(batch_samples, axis=0)  # (num_samples, batch_size, horizon)
            all_samples.append(batch_samples)
    
    # Combine batches
    samples = np.concatenate([s for s in all_samples], axis=1)  # (num_samples, total_series, horizon)
    contexts = np.concatenate(all_contexts, axis=0)  # (total_series, context_length)
    actuals = np.concatenate(all_actuals, axis=0)  # (total_series, horizon)
    indices = np.concatenate(all_indices, axis=0)  # (total_series,)
    
    return samples, contexts, actuals, indices

def save_samples(samples, contexts, actuals, output_dir):
    """Save generated samples to disk."""
    os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
    
    np.save(os.path.join(output_dir, 'evaluation', 'samples.npy'), samples)
    np.save(os.path.join(output_dir, 'evaluation', 'context.npy'), contexts)
    np.save(os.path.join(output_dir, 'evaluation', 'actuals.npy'), actuals)
    
    print(f"Saved samples to {os.path.join(output_dir, 'evaluation')}")

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

def save_metrics(metrics, output_dir):
    """Save metrics to disk."""
    os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
    
    with open(os.path.join(output_dir, 'evaluation', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to {os.path.join(output_dir, 'evaluation', 'metrics.json')}")

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
    
    # Add quantization config
    quant_config = config.get('quantization', {})
    plt.text(0.5, y_pos, f"Quantization: {quant_config.get('type', 'N/A')}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"K: {quant_config.get('k_initial', 'N/A')}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    # Add policy config
    policy_config = config.get('policy', {})
    plt.text(0.5, y_pos, f"Policy: {policy_config.get('backward_policy_type', 'N/A')}", 
             horizontalalignment='center', fontsize=12)
    y_pos -= 0.04
    
    plt.text(0.5, y_pos, f"Entropy bonus: {policy_config.get('lambda_entropy', 'N/A')}", 
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
    
    # Load model and config
    try:
        model, config = load_model(args.results_dir, args.checkpoint)
        print(f"Loaded model from {args.results_dir}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create dataset and dataloader
    try:
        test_dataset = create_dataset(config, 'test')
        test_loader = create_dataloaders(test_dataset, config)
        print(f"Loaded test data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate samples
    try:
        print(f"Generating {args.num_samples} samples for each test series...")
        samples, contexts, actuals, indices = generate_samples(model, test_loader, args.num_samples)
        print(f"Generated samples shape: {samples.shape}")
    except Exception as e:
        print(f"Error generating samples: {e}")
        return
    
    # Save samples
    save_samples(samples, contexts, actuals, args.output_dir)
    
    # Calculate metrics
    try:
        print("Calculating metrics...")
        metrics = calculate_metrics(samples, actuals, contexts)
        save_metrics(metrics, args.output_dir)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = None
    
    # Create experiment summary
    if metrics is not None:
        create_experiment_summary(args.results_dir, os.path.join(args.output_dir, 'plots'), 
                                 config, metrics, samples, contexts, actuals)
    
    # Plot individual series
    plots_dir = os.path.join(args.output_dir, 'plots')
    num_series = min(args.num_series, contexts.shape[0])
    print(f"Generating plots for {num_series} series...")
    
    context_length = contexts.shape[1]
    prediction_horizon = actuals.shape[1]
    
    for i in range(num_series):
        plot_forecast_with_samples(
            contexts, actuals, samples, i, 
            context_length, prediction_horizon, 
            args.figsize, plots_dir, dpi=args.dpi
        )
    
    # Plot overview of all series
    plot_all_series_overview(
        contexts, actuals, samples, num_series, 
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