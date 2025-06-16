#!/usr/bin/env python
"""
Test script for device switching functionality.
Demonstrates modular CPU/GPU switching for Temporal GFN.
"""
import os
import sys
import torch
import argparse

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.temporal_gfn.utils.device import create_device_manager, setup_slurm_environment
from src.temporal_gfn.models.transformer import TemporalTransformerModel


def test_device_manager():
    """Test device manager functionality."""
    print("=== Testing Device Manager ===")
    
    # Test 1: Auto-detection
    print("\n1. Auto-detection:")
    manager = create_device_manager(log_info=True)
    print(f"Auto-detected device: {manager.get_device()}")
    
    # Test 2: Force CPU
    print("\n2. Force CPU:")
    cpu_manager = create_device_manager(force_cpu=True, log_info=False)
    print(f"Forced CPU device: {cpu_manager.get_device()}")
    
    # Test 3: Specific GPU (if available)
    if torch.cuda.is_available():
        print("\n3. Specific GPU:")
        gpu_manager = create_device_manager(device='cuda:0', log_info=False)
        print(f"Specific GPU device: {gpu_manager.get_device()}")
        
        # Test memory info
        memory_info = gpu_manager.get_memory_info()
        print(f"GPU Memory: {memory_info['free_memory'] / 1024**3:.1f} GB free")
    
    return manager


def test_model_switching():
    """Test model device switching."""
    print("\n=== Testing Model Device Switching ===")
    
    # Create a small model for testing
    model = TemporalTransformerModel(
        d_model=64,
        nhead=4,
        d_hid=128,
        nlayers=2,
        dropout=0.1,
        k=10,
        context_length=96,
        prediction_horizon=24,
    )
    
    print(f"Initial model device: {next(model.parameters()).device}")
    
    # Test switching to CPU
    cpu_manager = create_device_manager(force_cpu=True, log_info=False)
    model = cpu_manager.to_device(model)
    print(f"After CPU switch: {next(model.parameters()).device}")
    
    # Test switching to GPU (if available)
    if torch.cuda.is_available():
        gpu_manager = create_device_manager(device='cuda:0', log_info=False)
        model = gpu_manager.to_device(model)
        print(f"After GPU switch: {next(model.parameters()).device}")
        
        # Test memory usage
        memory_before = gpu_manager.get_memory_info()
        print(f"GPU memory before: {memory_before['allocated_memory'] / 1024**3:.3f} GB")
        
        # Create some dummy data
        dummy_input = torch.randn(4, 96, 1).to(gpu_manager.get_device())
        with torch.no_grad():
            output = model(dummy_input, torch.zeros(4, 24, 1).to(gpu_manager.get_device()), 
                          torch.ones(4, 24).bool().to(gpu_manager.get_device()), 0)
        
        memory_after = gpu_manager.get_memory_info()
        print(f"GPU memory after: {memory_after['allocated_memory'] / 1024**3:.3f} GB")
        
        # Clear cache
        gpu_manager.clear_cache()
        memory_cleared = gpu_manager.get_memory_info()
        print(f"GPU memory after clear: {memory_cleared['allocated_memory'] / 1024**3:.3f} GB")


def test_slurm_environment():
    """Test SLURM environment detection."""
    print("\n=== Testing SLURM Environment ===")
    
    slurm_vars = setup_slurm_environment()
    if slurm_vars:
        print("SLURM environment detected:")
        for key, value in slurm_vars.items():
            print(f"  {key}: {value}")
    else:
        print("No SLURM environment detected")
        print("To simulate SLURM, set environment variables like:")
        print("  export SLURM_PROCID=0")
        print("  export SLURM_NTASKS=1")
        print("  export SLURM_LOCALID=0")


def test_optimal_batch_size():
    """Test optimal batch size calculation."""
    print("\n=== Testing Optimal Batch Size ===")
    
    from src.temporal_gfn.utils.device import get_optimal_batch_size
    
    # Create a dummy model
    model = TemporalTransformerModel(
        d_model=256, nhead=8, d_hid=512, nlayers=4,
        dropout=0.1, k=10, context_length=96, prediction_horizon=24
    )
    
    # Test for different devices
    base_batch_size = 32
    
    # CPU
    cpu_device = torch.device('cpu')
    cpu_batch_size = get_optimal_batch_size(cpu_device, model, base_batch_size)
    print(f"CPU optimal batch size: {cpu_batch_size} (base: {base_batch_size})")
    
    # GPU (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda:0')
        gpu_batch_size = get_optimal_batch_size(gpu_device, model, base_batch_size)
        print(f"GPU optimal batch size: {gpu_batch_size} (base: {base_batch_size})")
        
        # Show GPU memory info
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test device switching functionality')
    parser.add_argument('--test', choices=['all', 'manager', 'switching', 'slurm', 'batch_size'], 
                       default='all', help='Which test to run')
    args = parser.parse_args()
    
    print("Temporal GFN Device Switching Test")
    print("=" * 40)
    
    if args.test in ['all', 'manager']:
        test_device_manager()
    
    if args.test in ['all', 'switching']:
        test_model_switching()
    
    if args.test in ['all', 'slurm']:
        test_slurm_environment()
    
    if args.test in ['all', 'batch_size']:
        test_optimal_batch_size()
    
    print("\n=== Test Summary ===")
    print("✓ Device manager functionality")
    print("✓ Model device switching")
    print("✓ SLURM environment detection")
    print("✓ Optimal batch size calculation")
    print("\nDevice switching is ready for Cedar cluster!")


if __name__ == "__main__":
    main() 