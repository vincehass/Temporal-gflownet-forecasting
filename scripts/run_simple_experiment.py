#!/usr/bin/env python
"""
Script to run a simple experiment without Hydra dependency.
"""
import os
import sys
import torch
import yaml
import argparse
from typing import Dict, Any

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules
from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy
from src.temporal_gfn.trainers import TemporalGFNTrainer
from src.temporal_gfn.data.dataset import SyntheticTimeSeriesDataset, create_dataloader


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run a simple experiment with the Temporal GFN model')
    parser.add_argument('--config', type=str, default='results/example_test/config.yaml', help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-forecasting', help='W&B project name')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add wandb parameters to config
    config['use_wandb'] = args.use_wandb
    config['wandb_project'] = args.wandb_project
    
    # Print configuration
    print("Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print(f"  Use W&B: {args.use_wandb}")
    
    # Create directories
    results_dir = os.path.dirname(args.config)
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset_config = config['dataset']
    
    train_dataset = SyntheticTimeSeriesDataset(
        num_series=100,
        series_length=200,
        context_length=dataset_config['context_length'],
        prediction_horizon=dataset_config['prediction_horizon'],
        model_type='combined',
        noise_level=0.1,
        scaler_type='mean',
        stride=1,
        sample_stride=1,
        return_indices=False,
        seed=args.seed,
    )
    
    val_dataset = SyntheticTimeSeriesDataset(
        num_series=20,
        series_length=200,
        context_length=dataset_config['context_length'],
        prediction_horizon=dataset_config['prediction_horizon'],
        model_type='combined',
        noise_level=0.1,
        scaler_type='mean',
        stride=1,
        sample_stride=1,
        return_indices=False,
        seed=args.seed + 1,
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == 'cuda'),
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == 'cuda'),
    )
    
    # Create models
    print("Creating models...")
    model_config = config['model']
    k_initial = config['quantization']['k_initial']
    
    forward_model = TemporalTransformerModel(
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        d_hid=model_config['d_hid'],
        nlayers=model_config['nlayers'],
        dropout=model_config['dropout'],
        k=k_initial,
        context_length=dataset_config['context_length'],
        prediction_horizon=dataset_config['prediction_horizon'],
        is_forward=True,
        uniform_init=model_config['uniform_init'],
    )
    
    backward_model = None
    if config['policy']['backward_policy_type'] == 'learned':
        backward_model = TemporalTransformerModel(
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            d_hid=model_config['d_hid'],
            nlayers=model_config['nlayers'],
            dropout=model_config['dropout'],
            k=k_initial,
            context_length=dataset_config['context_length'],
            prediction_horizon=dataset_config['prediction_horizon'],
            is_forward=False,
            uniform_init=model_config['uniform_init'],
        )
    
    # Move models to device
    device = torch.device(args.device)
    forward_model = forward_model.to(device)
    if backward_model is not None:
        backward_model = backward_model.to(device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = TemporalGFNTrainer(
        config=config,
        forward_model=forward_model,
        backward_model=backward_model,
        device=device,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 