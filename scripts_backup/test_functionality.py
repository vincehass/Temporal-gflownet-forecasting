#!/usr/bin/env python
"""
Test script to verify core functionality of Temporal GFN components.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Any

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy, StraightThroughEstimator
from src.temporal_gfn.gfn.env import GFNEnvironment
from src.temporal_gfn.gfn.tb_loss import TrajectoryBalanceLoss
from src.temporal_gfn.gfn.sampling import sample_forward_trajectory
from src.temporal_gfn.quantization.base import quantize, dequantize
from src.temporal_gfn.quantization.adaptive import AdaptiveQuantization
from src.temporal_gfn.data.scaling import MeanScaler, StandardScaler
from src.temporal_gfn.data.windowing import create_windows
from src.temporal_gfn.utils.logging import create_logger, Logger

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test Temporal GFN functionality')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use W&B for logging')
    parser.add_argument('--wandb_entity', type=str, default='nadhirvincenthassen',
                        help='W&B entity name')
    parser.add_argument('--wandb_project', type=str, default='temporal-gfn-test',
                        help='W&B project name')
    parser.add_argument('--skip_venv_check', action='store_true',
                        help='Skip virtual environment check')
    return parser.parse_args()

def main():
    """Main function to test Temporal GFN functionality."""
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
    
    print("Testing Temporal GFN functionality...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up logging
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'test_run')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    config = {
        "device": str(device),
        "context_length": 96,
        "prediction_horizon": 24,
        "transformer": {
            "d_model": 64,
            "nhead": 4,
            "d_hid": 128,
            "nlayers": 2,
            "dropout": 0.1
        },
        "quantization": {
            "k": 10,
            "k_max": 20,
            "vmin": -5.0,
            "vmax": 5.0,
            "adaptive": True
        }
    }
    
    logger = create_logger(
        log_dir=log_dir,
        experiment_name="functionality_test",
        config=config,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project
    )
    
    # Log test initialization
    logger.log_metrics({"test/initialized": 1.0}, step=0)

    # Generate synthetic time series data for testing
    def generate_synthetic_data(batch_size=4, seq_len=200, freq=0.1, noise=0.1):
        """Generate synthetic time series data."""
        t = np.arange(0, seq_len)
        base_signal = np.sin(2 * np.pi * freq * t) + 0.1 * t
        
        # Create batch of time series
        batch = []
        for i in range(batch_size):
            noise_scale = noise * (i + 1) / batch_size  # Vary noise level
            signal = base_signal + noise_scale * np.random.randn(seq_len)
            batch.append(signal)
        
        return torch.tensor(batch, dtype=torch.float32)

    # Test parameters
    context_length = 96
    prediction_horizon = 24
    batch_size = 4
    k = 10
    vmin = -5.0
    vmax = 5.0
    d_model = 64  # Smaller for testing

    # Generate synthetic data
    print("Generating synthetic data...")
    time_series = generate_synthetic_data(batch_size, seq_len=context_length + prediction_horizon)
    time_series = time_series.to(device)
    logger.log_metrics({"test/data_generated": 1.0}, step=1)

    # Test windowing
    print("Testing windowing functionality...")
    context, target = create_windows(time_series, context_length, prediction_horizon)
    print(f"Context shape: {context.shape}, Target shape: {target.shape}")
    logger.log_metrics({
        "test/context_shape": context.shape[1],
        "test/target_shape": target.shape[1]
    }, step=2)

    # Test scaling
    print("Testing scaling functionality...")
    scaler = MeanScaler()
    scaled_context = scaler.fit_transform(context)
    print(f"Scaled context mean: {scaled_context.mean().item():.4f}")
    logger.log_metrics({"test/scaled_context_mean": scaled_context.mean().item()}, step=3)

    # Test quantization
    print("Testing quantization functionality...")
    test_values = torch.linspace(vmin-1, vmax+1, 20, device=device)
    quantized = quantize(test_values, vmin, vmax, k)
    dequantized = dequantize(quantized, vmin, vmax, k)
    print(f"Original values: {test_values[:5]}")
    print(f"Quantized indices: {quantized[:5]}")
    print(f"Dequantized values: {dequantized[:5]}")
    
    # Log quantization results
    quant_error = ((test_values - dequantized)**2).mean().item()
    logger.log_metrics({"test/quantization_mse": quant_error}, step=4)
    logger.log_histogram("test/original_values", test_values.cpu().numpy(), step=4)
    logger.log_histogram("test/dequantized_values", dequantized.cpu().numpy(), step=4)

    # Test adaptive quantization
    print("Testing adaptive quantization...")
    adaptive_quant = AdaptiveQuantization(
        k_initial=k, 
        k_max=20,
        vmin=vmin,
        vmax=vmax,
        update_interval=5
    )
    print(f"Initial k: {adaptive_quant.get_current_k()}")
    logger.log_metrics({"test/initial_k": adaptive_quant.get_current_k()}, step=5)

    # Test GFN environment
    print("Testing GFN environment...")
    env = GFNEnvironment(vmin, vmax, k, context_length, prediction_horizon, device=device)
    initial_state = env.get_initial_state(scaled_context)
    print(f"Initial state shapes - context: {initial_state['context'].shape}, forecast: {initial_state['forecast'].shape}")
    logger.log_metrics({
        "test/context_shape": initial_state['context'].shape[1],
        "test/forecast_shape": initial_state['forecast'].shape[1]
    }, step=6)

    # Test transformer model
    print("Testing transformer model...")
    transformer = TemporalTransformerModel(
        d_model=d_model,
        nhead=4,
        d_hid=128,
        nlayers=2,
        dropout=0.1,
        k=k,
        context_length=context_length,
        prediction_horizon=prediction_horizon
    ).to(device)
    print(f"Transformer parameter count: {sum(p.numel() for p in transformer.parameters())}")
    logger.log_metrics({"test/transformer_params": sum(p.numel() for p in transformer.parameters())}, step=7)

    # Test forward policy
    print("Testing forward policy...")
    forward_policy = ForwardPolicy(transformer).to(device)

    # Test sampling from policy
    logits = forward_policy(
        context=initial_state['context'],
        forecast=initial_state['forecast'],
        forecast_mask=initial_state['mask'],
        step=0
    )[2]
    print(f"Forward policy logits shape: {logits.shape}")
    logger.log_metrics({"test/logits_shape": logits.shape[1]}, step=8)
    logger.log_histogram("test/logits", logits.detach().cpu().numpy(), step=8)

    # Test TB loss
    print("Testing TB loss...")
    tb_loss = TrajectoryBalanceLoss(Z_init=0.0, Z_lr=0.01, lambda_entropy=0.01)
    dummy_forward_logprobs = torch.randn(batch_size, device=device)
    dummy_backward_logprobs = torch.randn(batch_size, device=device)
    dummy_log_rewards = torch.randn(batch_size, device=device)
    loss_value = tb_loss(dummy_forward_logprobs, dummy_backward_logprobs, dummy_log_rewards)
    print(f"TB loss value: {loss_value.item():.4f}")
    logger.log_metrics({"test/tb_loss": loss_value.item()}, step=9)

    # Test trajectory sampling
    print("Testing trajectory sampling...")
    try:
        terminal_state, actions, _ = sample_forward_trajectory(
            context=scaled_context,
            forward_policy=forward_policy,
            env=env
        )
        print(f"Sampled trajectory - num steps: {len(actions)}, terminal state forecast shape: {terminal_state['forecast'].shape}")
        logger.log_metrics({
            "test/trajectory_steps": len(actions),
            "test/trajectory_success": 1.0
        }, step=10)
    except Exception as e:
        print(f"Error in trajectory sampling: {str(e)}")
        logger.log_metrics({"test/trajectory_success": 0.0}, step=10)

    print("\nAll tests completed!")

    def plot_test_results():
        """Plot some test results."""
        plt.figure(figsize=(12, 8))
        
        # Plot original vs dequantized values
        plt.subplot(2, 2, 1)
        plt.plot(test_values.cpu().numpy(), label='Original')
        plt.plot(dequantized.cpu().numpy(), 'o-', label='Dequantized')
        plt.legend()
        plt.title('Quantization Test')
        
        # Plot original time series
        plt.subplot(2, 2, 2)
        for i in range(min(3, batch_size)):
            plt.plot(time_series[i].cpu().numpy(), label=f'Series {i+1}')
        plt.legend()
        plt.title('Synthetic Time Series')
        
        # Save the figure
        os.makedirs('results', exist_ok=True)
        
        # Log to W&B
        logger.log_image("test/results_plot", plt, step=11)
        
        # Save locally as well
        plt.tight_layout()
        plt.savefig('results/test_functionality.png')
        print("Saved test plot to results/test_functionality.png")

    # Generate plots
    try:
        plot_test_results()
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
    
    # Close logger
    logger.close()
    print("Logging completed and closed")

if __name__ == "__main__":
    main() 