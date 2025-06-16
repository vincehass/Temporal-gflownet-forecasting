#!/usr/bin/env python3
"""
Minimal wrapper script for testing Temporal GFN on cluster.
This script follows the test_cluster.sh pattern and runs a basic experiment.
"""
import os
import sys
import torch
import yaml
import logging
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from temporal_gfn.utils.device import create_device_manager, setup_slurm_environment
    from temporal_gfn.models.transformer import TemporalTransformerModel
    from temporal_gfn.data.dataset import SyntheticTimeSeriesDataset, create_dataloader
    from temporal_gfn.trainers import TemporalGFNTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import path...")
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    try:
        from src.temporal_gfn.utils.device import create_device_manager, setup_slurm_environment
        from src.temporal_gfn.models.transformer import TemporalTransformerModel
        from src.temporal_gfn.data.dataset import SyntheticTimeSeriesDataset, create_dataloader
        from src.temporal_gfn.trainers import TemporalGFNTrainer
    except ImportError as e2:
        print(f"Second import error: {e2}")
        print("Available modules:")
        for path in sys.path:
            if os.path.exists(path):
                print(f"  {path}: {os.listdir(path)[:5]}...")
        sys.exit(1)

def create_minimal_config():
    """Create a minimal configuration for testing."""
    return {
        'dataset': {
            'type': 'synthetic',
            'num_series': 10,  # Small for testing
            'series_length': 200,
            'context_length': 96,
            'prediction_horizon': 24,
            'model_type': 'ar',
            'noise_level': 0.1,
            'scaler_type': 'mean',
            'stride': 1,
            'sample_stride': 1,
            'seed': 42
        },
        'model': {
            'd_model': 128,  # Smaller for testing
            'nhead': 4,
            'd_hid': 256,
            'nlayers': 2,
            'dropout': 0.1,
            'uniform_init': True
        },
        'quantization': {
            'vmin': -5.0,
            'vmax': 5.0,
            'k_initial': 10,
            'k_max': 50,
            'adaptive': True,
            'lambda_adapt': 0.9,
            'epsilon_adapt': 0.02,
            'delta_adapt': 5,
            'update_interval': 100
        },
        'policy': {
            'backward_policy_type': 'uniform',
            'rand_action_prob': 0.1
        },
        'gfn': {
            'Z_init': 1.0,
            'Z_lr': 0.01,
            'lambda_entropy': 0.01
        },
        'training': {
            'batch_size': 8,  # Small for testing
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'epochs': 5,  # Few epochs for testing
            'num_workers': 2,
            'grad_clip_norm': 1.0,
            'use_lr_scheduler': False,
            'multi_gpu': False
        },
        'evaluation': {
            'batch_size': 8,
            'num_samples': 10,
            'enabled': True
        },
        'logging': {
            'log_interval': 10,
            'checkpoint_interval': 100,
            'results_dir': 'results/minimal_test'
        }
    }

def main():
    """Main function for minimal cluster test."""
    print("=" * 60)
    print("MINIMAL TEMPORAL GFN CLUSTER TEST")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check system information
    print(f"\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check SLURM environment
    print(f"\n=== SLURM Environment ===")
    slurm_vars = setup_slurm_environment()
    if slurm_vars:
        print("SLURM environment detected:")
        for key, value in slurm_vars.items():
            print(f"  {key}: {value}")
    else:
        print("No SLURM environment detected (running locally)")
    
    # Setup device
    print(f"\n=== Device Setup ===")
    device_manager = create_device_manager(log_info=True)
    device = device_manager.get_device()
    print(f"Selected device: {device}")
    
    if device_manager.is_gpu():
        memory_info = device_manager.get_memory_info()
        print(f"GPU Memory: {memory_info['free_memory'] / 1024**3:.1f} GB free")
    
    # Create minimal config
    config = create_minimal_config()
    
    # Create results directory
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    
    # Save config
    config_path = os.path.join(config['logging']['results_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")
    
    try:
        # Create dataset
        print(f"\n=== Creating Dataset ===")
        dataset = SyntheticTimeSeriesDataset(
            num_series=config['dataset']['num_series'],
            series_length=config['dataset']['series_length'],
            context_length=config['dataset']['context_length'],
            prediction_horizon=config['dataset']['prediction_horizon'],
            model_type=config['dataset']['model_type'],
            noise_level=config['dataset']['noise_level'],
            scaler_type=config['dataset']['scaler_type'],
            seed=config['dataset']['seed']
        )
        print(f"Dataset created: {len(dataset)} samples")
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=device_manager.is_gpu()
        )
        print(f"Dataloader created: {len(dataloader)} batches")
        
        # Create model
        print(f"\n=== Creating Model ===")
        model = TemporalTransformerModel(
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            d_hid=config['model']['d_hid'],
            nlayers=config['model']['nlayers'],
            dropout=config['model']['dropout'],
            k=config['quantization']['k_initial'],
            context_length=config['dataset']['context_length'],
            prediction_horizon=config['dataset']['prediction_horizon'],
            is_forward=True,
            uniform_init=config['model']['uniform_init']
        )
        
        # Move model to device
        model = device_manager.to_device(model)
        print(f"Model created and moved to device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print(f"\n=== Testing Forward Pass ===")
        sample_batch = next(iter(dataloader))
        context = sample_batch['context'].to(device)
        target = sample_batch['target'].to(device)
        
        print(f"Context shape: {context.shape}")
        print(f"Target shape: {target.shape}")
        
        with torch.no_grad():
            # Test model forward pass
            forecast = torch.zeros_like(target)
            forecast_mask = torch.ones(target.shape[:-1], dtype=torch.bool).to(device)
            
            output = model(context, forecast, forecast_mask, step=0)
            print(f"Model output shape: {output.shape}")
            print(f"Model output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Create trainer
        print(f"\n=== Creating Trainer ===")
        trainer = TemporalGFNTrainer(
            config=config,
            forward_model=model,
            backward_model=None,  # Use uniform backward policy
            device=device,
            force_cpu=False,
            gpu_id=None
        )
        print("Trainer created successfully")
        
        # Run minimal training
        print(f"\n=== Running Minimal Training ===")
        print(f"Training for {config['training']['epochs']} epochs...")
        
        trainer.train(
            train_loader=dataloader,
            val_loader=None,  # No validation for minimal test
            num_epochs=config['training']['epochs']
        )
        
        print(f"\n=== Training Completed Successfully ===")
        
        # Test device switching if GPU available
        if torch.cuda.is_available() and device.type == 'cuda':
            print(f"\n=== Testing Device Switching ===")
            print("Switching to CPU...")
            trainer.switch_device('cpu')
            print(f"Current device: {trainer.device}")
            
            print("Switching back to GPU...")
            trainer.switch_device('cuda:0')
            print(f"Current device: {trainer.device}")
        
        # Final memory check
        if device_manager.is_gpu():
            final_memory = trainer.get_memory_usage()
            print(f"\nFinal GPU memory: {final_memory['allocated_memory'] / 1024**3:.2f} GB allocated")
        
        print(f"\n" + "=" * 60)
        print("MINIMAL TEST COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now()}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"ERROR: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 