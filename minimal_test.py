#!/usr/bin/env python
"""
Minimal test script to run a temporal GFN experiment.
"""

import os
import sys
import types
import yaml
import torch
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the required modules
from src.temporal_gfn.trainers import TemporalGFNTrainer
from scripts.train import create_model, create_dataset, create_dataloader

# Create a monkey-patched version of TemporalGFNTrainer.__init__ that handles both 'lr' and 'learning_rate'
original_init = TemporalGFNTrainer.__init__

def patched_init(self, config, forward_model, backward_model=None, device='cuda'):
    # Fix the lr/learning_rate issue
    if 'training' in config and 'learning_rate' in config['training'] and 'lr' not in config['training']:
        config['training']['lr'] = config['training']['learning_rate']
    
    # Call the original __init__
    original_init(self, config, forward_model, backward_model, device)

# Apply the monkey patch
TemporalGFNTrainer.__init__ = patched_init

def main():
    # Create a temporary directory for results
    results_dir = Path("results/minimal_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple configuration
    config = {
        'dataset': {
            'type': 'synthetic',
            'context_length': 24,
            'prediction_horizon': 8,
            'scaler_type': 'mean',
            'stride': 1,
            'sample_stride': 5,
            'return_indices': False,
        },
        'model': {
            'd_model': 32,
            'nhead': 2,
            'd_hid': 64,
            'nlayers': 2,
            'dropout': 0.1,
            'uniform_init': True,
        },
        'quantization': {
            'adaptive': True,
            'k_initial': 5,
            'k_max': 10,
            'vmin': -10.0,
            'vmax': 10.0,
            'lambda_adapt': 0.9,
            'epsilon_adapt': 0.02,
            'delta_adapt': 5,
            'update_interval': 100,
        },
        'policy': {
            'backward_policy_type': 'uniform',
            'rand_action_prob': 0.01,
        },
        'gfn': {
            'Z_init': 0.0,
            'Z_lr': 0.01,
            'lambda_entropy': 0.01,
        },
        'training': {
            'epochs': 2,
            'batch_size': 8,
            'learning_rate': 0.001,  # Using learning_rate instead of lr
            'weight_decay': 0.0001,
            'use_lr_scheduler': False,
            'grad_clip_norm': 1.0,
            'num_workers': 0,
        },
        'validation': {
            'enabled': True,
        },
        'evaluation': {
            'batch_size': 8,
            'num_samples': 10,
            'quantiles': [0.1, 0.5, 0.9],
            'seasonality': 1,
            'save_forecasts': False,
        },
        'logging': {
            'results_dir': str(results_dir),
            'log_interval': 10,
            'checkpoint_interval': 1,
        },
        'results_dir': str(results_dir),
    }
    
    # Save the configuration to a file
    config_path = results_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {config_path}")
    print("Starting training...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=(str(device).startswith('cuda')),
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=(str(device).startswith('cuda')),
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
    
    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 