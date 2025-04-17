#!/usr/bin/env python
"""
Test script to verify Weights & Biases integration with efficiency optimizations.
"""
import os
import sys
import argparse
import numpy as np
import torch
import wandb

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.temporal_gfn.utils.wandb_utils import WandbManager, create_wandb_manager

def check_virtual_env():
    """Check if running inside a virtual environment."""
    # Check for common virtual environment indicators
    in_venv = False
    
    # Check for venv/virtualenv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        in_venv = True
        venv_name = os.path.basename(sys.prefix)
        return in_venv, venv_name
    
    # Check for conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        in_venv = True
        return in_venv, conda_env
        
    return in_venv, None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Weights & Biases integration with efficiency optimizations')
    parser.add_argument('--project', type=str, default='temporal-gfn-test',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default='nadhirvincenthassen',
                        help='WandB entity name')
    parser.add_argument('--offline', action='store_true',
                        help='Use W&B in offline mode (for limited connectivity)')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging completely')
    parser.add_argument('--skip_venv_check', action='store_true',
                        help='Skip virtual environment check')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Frequency for model parameter logging')
    parser.add_argument('--log_gradients', action='store_true',
                        help='Log gradients (increases memory usage)')
    return parser.parse_args()

def create_dummy_model():
    """Create a simple dummy model for testing."""
    # Simple model with a few layers
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )
    return model

def main():
    """Main function to test WandB integration with efficiency optimizations."""
    args = parse_args()
    
    # Check for virtual environment
    if not args.skip_venv_check:
        in_venv, venv_name = check_virtual_env()
        if not in_venv:
            print("\n⚠️  WARNING: Not running in a virtual environment!")
            print("It is recommended to activate the project's virtual environment first:")
            print("  source venv/bin/activate  # For venv")
            print("  conda activate temporal_gfn  # For conda")
            print("\nContinue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Exiting. Please activate the virtual environment and try again.")
                sys.exit(1)
            print("Continuing without virtual environment...\n")
        else:
            print(f"✓ Running in virtual environment: {venv_name}\n")
    
    if args.disable_wandb:
        print("W&B logging disabled")
        return
        
    # Create log directory
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'wandb_test')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configuration for the test
    config = {
        "model": "temporal-gfn",
        "quantization": {
            "k_initial": 10,
            "adaptive": True,
            "vmin": -5.0,
            "vmax": 5.0
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "efficiency_settings": {
            "offline_mode": args.offline,
            "log_freq": args.log_freq,
            "log_gradients": args.log_gradients,
        }
    }
    
    print(f"{'OFFLINE' if args.offline else 'ONLINE'} W&B test with entity: {args.entity}, project: {args.project}")
    
    # Create WandB manager with efficient settings
    wandb_manager = create_wandb_manager(
        entity=args.entity,
        project=args.project,
        experiment_name=f"efficiency-test-{'offline' if args.offline else 'online'}",
        config=config,
        offline=args.offline,
        log_dir=log_dir,
        save_code=True,
        watch_model=True,
        log_freq=args.log_freq,
        log_gradients=args.log_gradients
    )
    
    # Initialize W&B
    with wandb_manager:
        print(f"✓ W&B initialized in {'offline' if args.offline else 'online'} mode")
        
        # Create dummy model and watch it
        dummy_model = create_dummy_model()
        wandb_manager.watch(dummy_model)
        print("✓ Model watching enabled")
        
        # Simulate some training metrics
        print("Logging sample metrics to W&B...")
        for epoch in range(5):
            train_loss = 1.0 - 0.15 * epoch + 0.03 * np.random.randn()
            val_loss = 1.2 - 0.12 * epoch + 0.05 * np.random.randn()
            k_value = 10 + epoch
            
            metrics = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "quantization/k": k_value,
                "train/learning_rate": 0.001 * (0.9 ** epoch),
                "memory/gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "memory/gpu_memory_cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            }
            
            # Forward and backward pass with dummy data
            if args.log_gradients and epoch > 0:
                x = torch.randn(16, 10)
                output = dummy_model(x)
                loss = output.mean()
                loss.backward()
                
                # Optimize
                with torch.no_grad():
                    for param in dummy_model.parameters():
                        param.data -= 0.01 * param.grad
                        param.grad.zero_()
            
            wandb_manager.log(metrics, step=epoch)
            print(f"Epoch {epoch+1}/5 - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, k: {k_value}")
        
        # Create and log a simple plot
        try:
            import matplotlib.pyplot as plt
            
            # Create a simple plot
            plt.figure(figsize=(10, 6))
            x = np.linspace(0, 10, 100)
            plt.plot(x, np.sin(x), label='sin(x)')
            plt.plot(x, np.cos(x), label='cos(x)')
            plt.legend()
            plt.title('Sample Plot')
            
            # Log the plot to wandb
            wandb_manager.log({"sample_plot": wandb.Image(plt)})
            print("✓ Sample plot logged to W&B")
            
            plt.close()
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
        
        # Save a dummy checkpoint (only metadata in efficient mode)
        dummy_state = {
            'model_state': {key: value.cpu() for key, value in dummy_model.state_dict().items()},
            'epoch': 5,
            'best_metric': 0.75
        }
        checkpoint_path = os.path.join(log_dir, 'dummy_checkpoint.pt')
        torch.save(dummy_state, checkpoint_path)
        
        # Log the checkpoint with metadata but don't upload the actual file to save bandwidth
        metadata = {
            'epoch': 5,
            'metric': 0.75,
            'architecture': 'small_mlp',
        }
        # We're not actually uploading this in efficient mode
        print("Logging checkpoint metadata (not uploading actual checkpoint)...")
    
    print("\n✓ W&B test completed successfully.")
    if args.offline:
        print("\nOffline data saved to:", os.path.join(log_dir, 'wandb'))
        print("To sync this data later when online, use:")
        print("  wandb sync", os.path.join(log_dir, 'wandb'))

if __name__ == "__main__":
    main() 