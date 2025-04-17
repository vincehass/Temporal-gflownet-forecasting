#!/usr/bin/env python
"""
Evaluation script for Temporal GFN model.
"""
import os
import sys
import argparse
import yaml
import json
import numpy as np
from typing import Dict, Any

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy
from src.temporal_gfn.gfn.env import GFNEnvironment
from src.temporal_gfn.trainers import TemporalGFNTrainer
from src.temporal_gfn.gfn.sampling import sample_trajectories_batch
from src.temporal_gfn.data.dataset import (
    TimeSeriesDataset,
    SyntheticTimeSeriesDataset,
    create_dataloader,
)
from src.temporal_gfn.utils.metrics import calculate_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: Dict[str, Any], split: str = 'test'):
    """
    Create dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
    
    Returns:
        Dataset object
    """
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


def create_model(config: Dict[str, Any], k: int, is_forward: bool = True):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        k: Number of quantization bins
        is_forward: Whether to create forward or backward policy model
        
    Returns:
        Model object
    """
    model_config = config['model']
    
    model = TemporalTransformerModel(
        d_model=model_config.get('d_model', 256),
        nhead=model_config.get('nhead', 8),
        d_hid=model_config.get('d_hid', 512),
        nlayers=model_config.get('nlayers', 4),
        dropout=model_config.get('dropout', 0.1),
        k=k,
        context_length=config['dataset']['context_length'],
        prediction_horizon=config['dataset']['prediction_horizon'],
        is_forward=is_forward,
        uniform_init=model_config.get('uniform_init', True),
    )
    
    return model


def plot_forecasts(
    context: torch.Tensor,
    target: torch.Tensor,
    forecast_samples: torch.Tensor,
    idx: int,
    scaler,
    output_dir: str,
    sample_indices: list = None,
    quantiles: list = [0.1, 0.5, 0.9],
):
    """
    Plot forecasts for a single time series.
    
    Args:
        context: Context window of shape [context_length]
        target: Target window of shape [prediction_horizon]
        forecast_samples: Samples of shape [num_samples, prediction_horizon]
        idx: Index of the time series
        scaler: Scaler object for inverse transformation
        output_dir: Directory to save plots
        sample_indices: Indices of samples to plot individually
        quantiles: Quantiles to plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy
    context_np = context.cpu().numpy()
    target_np = target.cpu().numpy()
    forecast_np = forecast_samples.cpu().numpy()
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape for inverse transform
        context_reshaped = context.unsqueeze(0)
        target_reshaped = target.unsqueeze(0)
        forecast_reshaped = forecast_samples.unsqueeze(1).transpose(0, 1)
        
        # Inverse transform
        context_inv = scaler.inverse_transform(context_reshaped).squeeze(0).cpu().numpy()
        target_inv = scaler.inverse_transform(target_reshaped).squeeze(0).cpu().numpy()
        forecast_inv = scaler.inverse_transform(forecast_reshaped).squeeze(1).cpu().numpy()
        
        # Use inverse transformed data
        context_np = context_inv
        target_np = target_inv
        forecast_np = forecast_inv
    
    # Calculate time indices
    context_length = len(context_np)
    prediction_horizon = len(target_np)
    context_time = np.arange(context_length)
    forecast_time = np.arange(context_length, context_length + prediction_horizon)
    full_time = np.arange(context_length + prediction_horizon)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot context
    ax.plot(context_time, context_np, 'b-', label='Context', linewidth=2)
    
    # Plot target
    ax.plot(forecast_time, target_np, 'k-', label='Target', linewidth=2)
    
    # Plot individual samples
    if sample_indices is not None:
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx < len(forecast_np):
                ax.plot(
                    forecast_time,
                    forecast_np[sample_idx],
                    'g-',
                    alpha=0.3,
                    label='Sample' if i == 0 else None
                )
    
    # Plot quantiles
    if quantiles is not None:
        for q in quantiles:
            q_idx = int(len(forecast_np) * q)
            q_forecast = np.sort(forecast_np, axis=0)[q_idx]
            if q == 0.5:
                label = 'Median Forecast'
                color = 'r'
                alpha = 1.0
                linewidth = 2
            else:
                label = f'{int(q * 100)}th Percentile'
                color = 'r'
                alpha = 0.5
                linewidth = 1
            
            ax.plot(
                forecast_time,
                q_forecast,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=label
            )
    
    # Add labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'Forecast for Series {idx}')
    ax.legend()
    
    # Save figure
    fig.savefig(os.path.join(output_dir, f'forecast_{idx}.png'), bbox_inches='tight')
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (if using CUDA)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for probabilistic forecasting")
    parser.add_argument("--use_wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--wandb_project", type=str, default="temporal-gfn", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="nadhirvincenthassen", help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--offline", action="store_true", help="Run W&B in offline mode")
    
    # Get args from command line that Hydra doesn't process
    raw_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            raw_args.append(arg)
            if "=" not in arg:
                try:
                    idx = sys.argv.index(arg)
                    raw_args.append(sys.argv[idx + 1])
                except (ValueError, IndexError):
                    pass
    
    args = parser.parse_args(raw_args)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Load configuration
    if args.config_path is not None:
        config = load_config(args.config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("Configuration not found in checkpoint and not provided")
    
    # Update config with command line arguments
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['num_samples'] = args.num_samples
    
    # Create test dataset
    test_dataset = create_dataset(config, 'test')
    
    # Create dataloader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['evaluation'].get('batch_size', 32),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=(device == 'cuda'),
    )
    
    # Create models
    k = checkpoint.get('k', config['quantization']['k_initial'])
    forward_model = create_model(config, k, is_forward=True)
    
    backward_model = None
    if config['policy']['backward_policy_type'] == 'learned':
        backward_model = create_model(config, k, is_forward=False)
    
    # Load model weights
    forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
    if backward_model is not None and 'backward_model_state_dict' in checkpoint:
        backward_model.load_state_dict(checkpoint['backward_model_state_dict'])
    
    # Move models to device
    forward_model = forward_model.to(device)
    if backward_model is not None:
        backward_model = backward_model.to(device)
    
    # Create environment
    env = GFNEnvironment(
        vmin=config['quantization']['vmin'],
        vmax=config['quantization']['vmax'],
        k=k,
        context_length=config['dataset']['context_length'],
        prediction_horizon=config['dataset']['prediction_horizon'],
        device=device,
    )
    
    # Create policies
    forward_policy = ForwardPolicy(
        model=forward_model,
        use_ste=True,
    )
    
    backward_policy = BackwardPolicy(
        policy_type=config['policy']['backward_policy_type'],
        model=backward_model,
        prediction_horizon=config['dataset']['prediction_horizon']
    )
    
    # Set to evaluation mode
    forward_policy.eval()
    if hasattr(backward_policy, 'model') and backward_policy.model is not None:
        backward_policy.model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate forecasts and calculate metrics
    all_metrics = {}
    all_forecasts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get context and target windows
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            indices = batch['index'].cpu().numpy()
            
            # Sample multiple forecasts
            forecast_samples = sample_trajectories_batch(
                context_batch=context,
                forward_policy=forward_policy,
                backward_policy=backward_policy,
                env=env,
                num_samples=args.num_samples,
                deterministic=False,
                is_eval=True
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                forecast_samples=forecast_samples,
                targets=target,
                insample=context,
                quantiles=config['evaluation'].get('quantiles', None),
                seasonality=config['evaluation'].get('seasonality', 1),
            )
            
            # Store metrics for each sample
            for i, idx in enumerate(indices):
                idx_int = int(idx)
                all_metrics[idx_int] = {
                    'wql': metrics['wql'][i].item(),
                    'crps': metrics['crps'][i].item(),
                }
                if 'mase' in metrics:
                    all_metrics[idx_int]['mase'] = metrics['mase'][i].item()
            
            # Generate plots if requested
            if args.plot:
                for i, idx in enumerate(indices):
                    idx_int = int(idx)
                    plot_forecasts(
                        context=context[i],
                        target=target[i],
                        forecast_samples=forecast_samples[i],
                        idx=idx_int,
                        scaler=test_dataset.get_scaler(),
                        output_dir=os.path.join(args.output_dir, 'plots'),
                        sample_indices=list(range(min(args.plot_samples, args.num_samples))),
                        quantiles=[0.1, 0.5, 0.9],
                    )
            
            # Store the forecasts
            all_forecasts.append({
                'indices': indices.tolist(),
                'context': context.cpu().numpy().tolist(),
                'target': target.cpu().numpy().tolist(),
                'forecasts': forecast_samples.cpu().numpy().tolist(),
            })
    
    # Calculate overall metrics
    overall_metrics = {
        'wql_mean': np.mean([m['wql'] for m in all_metrics.values()]),
        'wql_std': np.std([m['wql'] for m in all_metrics.values()]),
        'crps_mean': np.mean([m['crps'] for m in all_metrics.values()]),
        'crps_std': np.std([m['crps'] for m in all_metrics.values()]),
    }
    
    if 'mase' in next(iter(all_metrics.values())):
        overall_metrics['mase_mean'] = np.mean([m['mase'] for m in all_metrics.values()])
        overall_metrics['mase_std'] = np.std([m['mase'] for m in all_metrics.values()])
    
    # Print overall metrics
    print("Overall Metrics:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'individual': all_metrics,
            'overall': overall_metrics,
        }, f, indent=2)
    
    # Save forecasts to file (optional, can be large)
    save_forecasts = config['evaluation'].get('save_forecasts', False)
    if save_forecasts:
        with open(os.path.join(args.output_dir, 'forecasts.json'), 'w') as f:
            json.dump(all_forecasts, f)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 