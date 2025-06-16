#!/usr/bin/env python
"""
Training script for Temporal GFN model.
"""
import os
import sys
import argparse
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy
from src.temporal_gfn.trainers import TemporalGFNTrainer
from src.temporal_gfn.data.dataset import (
    TimeSeriesDataset,
    SyntheticTimeSeriesDataset,
    create_dataloader,
)


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


def create_dataset(config: Dict[str, Any], split: str = 'train'):
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
            return_indices=dataset_config.get('return_indices', False),
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
            return_indices=dataset_config.get('return_indices', False),
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


@hydra.main(config_path="../configs", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    """Main function."""
    # Convert Hydra config to dict
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get command line args
    use_wandb = hasattr(cfg, "use_wandb") and cfg.use_wandb
    
    # Set device based on arguments
    device = "cpu"
    if torch.cuda.is_available():
        if hasattr(cfg, "gpu") and cfg.gpu >= 0:
            device = f"cuda:{cfg.gpu}"
        else:
            device = "cuda:0"  # Default to first GPU
    
    # Convert string 'cpu' or 'gpu' to appropriate device
    if hasattr(cfg, "device"):
        if cfg.device == "cpu":
            device = "cpu"
        elif cfg.device.startswith("cuda"):
            device = cfg.device
    
    print(f"Using device: {device}")
    
    if hasattr(cfg, "seed"):
        # Set random seed
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Create results directory
    if "results_dir" in config:
        results_dir = config["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
        
        # Save config
        with open(os.path.join(results_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    
    # Create datasets
    train_dataset = create_dataset(config, 'train')
    
    val_dataset = None
    if config.get('validation', {}).get('enabled', True):
        val_dataset = create_dataset(config, 'val')
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=(device.startswith('cuda')),
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['evaluation'].get('batch_size', config['training']['batch_size']),
            shuffle=False,
            num_workers=config['training'].get('num_workers', 0),
            pin_memory=(device.startswith('cuda')),
        )
    
    # Create models
    k_initial = config['quantization']['k_initial']
    forward_model = create_model(config, k_initial, is_forward=True)
    
    backward_model = None
    if config['policy']['backward_policy_type'] == 'learned':
        backward_model = create_model(config, k_initial, is_forward=False)
    
    # Move models to device
    forward_model = forward_model.to(device)
    if backward_model is not None:
        backward_model = backward_model.to(device)
    
    # Create trainer
    trainer = TemporalGFNTrainer(
        config=config,
        forward_model=forward_model,
        backward_model=backward_model,
        device=device,
    )
    
    # Resume training if specified
    start_epoch = 0
    if hasattr(cfg, "resume") and cfg.resume is not None:
        start_epoch = trainer.load_checkpoint(cfg.resume)
    
    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main() 