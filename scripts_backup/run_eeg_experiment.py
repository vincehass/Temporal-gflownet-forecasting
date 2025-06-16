#!/usr/bin/env python
"""
Script to run an EEG dataset experiment with Temporal GFN and wandb integration.
"""
import os
import sys
import torch
import yaml
import argparse
import numpy as np
from typing import Dict, Any
import wandb

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules
from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy
from src.temporal_gfn.trainers import TemporalGFNTrainer
from src.temporal_gfn.data.dataset import TimeSeriesDataset, create_dataloader


def load_eeg_data(path):
    """Load the EEG dataset."""
    try:
        if os.path.exists(path):
            return np.load(path)
        else:
            # If path doesn't exist, create synthetic data for demo purposes
            print(f"⚠️ WARNING: EEG data not found at {path}, creating synthetic data instead")
            return np.random.randn(100, 1000)  # 100 series of length 1000
    except Exception as e:
        print(f"Error loading EEG data: {e}, creating synthetic data instead")
        return np.random.randn(100, 1000)


def main():
    """Main function to run the experiment."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run an EEG experiment with Temporal GFN model')
    parser.add_argument('--config', type=str, default='results/eeg_experiment/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to use (cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Whether to use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-eeg', 
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Use adaptive quantization')
    parser.add_argument('--k', type=int, default=10,
                        help='Initial number of quantization bins')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    config['quantization']['adaptive'] = args.adaptive
    config['quantization']['k_initial'] = args.k
    
    # Add wandb parameters to config
    config['use_wandb'] = args.use_wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"eeg_{'adaptive' if args.adaptive else 'fixed'}_k{args.k}"
        )
    
    # Print configuration
    print("\n======== Configuration ========")
    print(f"Dataset: EEG")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Use W&B: {args.use_wandb}")
    print(f"Quantization: {'Adaptive' if args.adaptive else 'Fixed'} (K={args.k})")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print("==============================\n")
    
    # Create directories
    results_dir = config['logging']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    
    # Save updated config
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Load EEG dataset
    dataset_config = config['dataset']
    train_data = load_eeg_data(dataset_config.get('train_path', 'datasets/eeg/eeg_train.npy'))
    val_data = load_eeg_data(dataset_config.get('val_path', 'datasets/eeg/eeg_val.npy'))
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TimeSeriesDataset(
        time_series=train_data,
        context_length=dataset_config['context_length'],
        prediction_horizon=dataset_config['prediction_horizon'],
        scaler_type=dataset_config.get('scaler_type', 'mean'),
        stride=dataset_config.get('stride', 1),
        sample_stride=dataset_config.get('sample_stride', 1),
        return_indices=dataset_config.get('return_indices', False),
        forecast_mode=False,
    )
    
    val_dataset = TimeSeriesDataset(
        time_series=val_data,
        context_length=dataset_config['context_length'],
        prediction_horizon=dataset_config['prediction_horizon'],
        scaler_type=dataset_config.get('scaler_type', 'mean'),
        stride=dataset_config.get('stride', 1),
        sample_stride=dataset_config.get('sample_stride', 1),
        return_indices=dataset_config.get('return_indices', False),
        forecast_mode=False,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=(args.device == 'cuda'),
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['evaluation'].get('batch_size', config['training']['batch_size']),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
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
    # Fix issue with lr vs learning_rate in config
    if 'learning_rate' in config['training'] and 'lr' not in config['training']:
        config['training']['lr'] = config['training']['learning_rate']
    
    trainer = TemporalGFNTrainer(
        config=config,
        forward_model=forward_model,
        backward_model=backward_model,
        device=device,
    )
    
    # Train the model
    print("\n======= Starting training =======")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
    )
    
    print("\n======= Training completed! =======")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 