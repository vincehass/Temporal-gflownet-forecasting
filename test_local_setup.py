#!/usr/bin/env python3
"""
Local test script to verify Temporal GFN setup before cluster submission.
This script tests all imports and basic functionality.
"""
import os
import sys
import torch

print("=" * 60)
print("LOCAL TEMPORAL GFN SETUP TEST")
print("=" * 60)

# Test basic imports
print("\n=== Testing Basic Imports ===")
try:
    import yaml
    import logging
    from datetime import datetime
    import numpy as np
    import pandas as pd
    print("✓ Basic imports successful")
except ImportError as e:
    print(f"✗ Basic import failed: {e}")
    sys.exit(1)

# Test PyTorch
print("\n=== Testing PyTorch ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Test Temporal GFN imports
print("\n=== Testing Temporal GFN Imports ===")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from temporal_gfn.utils.device import create_device_manager, setup_slurm_environment
    print("✓ Device utils imported")
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from src.temporal_gfn.utils.device import create_device_manager, setup_slurm_environment
        print("✓ Device utils imported (alternative path)")
    except ImportError as e:
        print(f"✗ Device utils import failed: {e}")
        sys.exit(1)

try:
    from temporal_gfn.models.transformer import TemporalTransformerModel
    print("✓ Transformer model imported")
except ImportError:
    try:
        from src.temporal_gfn.models.transformer import TemporalTransformerModel
        print("✓ Transformer model imported (alternative path)")
    except ImportError as e:
        print(f"✗ Transformer model import failed: {e}")
        sys.exit(1)

try:
    from temporal_gfn.data.dataset import SyntheticTimeSeriesDataset, create_dataloader
    print("✓ Dataset imported")
except ImportError:
    try:
        from src.temporal_gfn.data.dataset import SyntheticTimeSeriesDataset, create_dataloader
        print("✓ Dataset imported (alternative path)")
    except ImportError as e:
        print(f"✗ Dataset import failed: {e}")
        sys.exit(1)

try:
    from temporal_gfn.trainers import TemporalGFNTrainer
    print("✓ Trainer imported")
except ImportError:
    try:
        from src.temporal_gfn.trainers import TemporalGFNTrainer
        print("✓ Trainer imported (alternative path)")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        sys.exit(1)

# Test device manager
print("\n=== Testing Device Manager ===")
try:
    device_manager = create_device_manager(log_info=False)
    device = device_manager.get_device()
    print(f"✓ Device manager created: {device}")
    
    if device_manager.is_gpu():
        memory_info = device_manager.get_memory_info()
        print(f"✓ GPU memory info: {memory_info['free_memory'] / 1024**3:.1f} GB free")
except Exception as e:
    print(f"✗ Device manager failed: {e}")
    sys.exit(1)

# Test model creation
print("\n=== Testing Model Creation ===")
try:
    model = TemporalTransformerModel(
        d_model=64,
        nhead=4,
        d_hid=128,
        nlayers=2,
        dropout=0.1,
        k=10,
        context_length=96,
        prediction_horizon=24,
        is_forward=True,
        uniform_init=True
    )
    model = device_manager.to_device(model)
    print(f"✓ Model created and moved to device: {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test dataset creation
print("\n=== Testing Dataset Creation ===")
try:
    dataset = SyntheticTimeSeriesDataset(
        num_series=5,
        series_length=200,
        context_length=96,
        prediction_horizon=24,
        model_type='ar',
        noise_level=0.1,
        scaler_type='mean',
        seed=42
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=device_manager.is_gpu()
    )
    print(f"✓ Dataloader created: {len(dataloader)} batches")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test forward pass
print("\n=== Testing Forward Pass ===")
try:
    sample_batch = next(iter(dataloader))
    context = sample_batch['context'].to(device)
    target = sample_batch['target'].to(device)
    
    print(f"✓ Batch loaded - Context: {context.shape}, Target: {target.shape}")
    
    with torch.no_grad():
        forecast = torch.zeros_like(target)
        forecast_mask = torch.ones(target.shape[:-1], dtype=torch.bool).to(device)
        
        output = model(context, forecast, forecast_mask, step=0)
        print(f"✓ Forward pass successful - Output: {output.shape}")
        print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test SLURM environment detection
print("\n=== Testing SLURM Environment ===")
slurm_vars = setup_slurm_environment()
if slurm_vars:
    print("✓ SLURM environment detected:")
    for key, value in slurm_vars.items():
        print(f"    {key}: {value}")
else:
    print("✓ No SLURM environment (expected for local testing)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("Setup is ready for cluster submission.")
print("=" * 60)

print("\nNext steps:")
print("1. Submit test_cluster.sh to verify cluster setup")
print("2. Submit scripts/submit_cedar.sh for full training")
print("3. Use minimal_wrapper.py for quick testing") 