#!/usr/bin/env python3
"""
Quick test to verify CPU device setup for local development.
Run this before the main CPU tests to ensure everything is configured correctly.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from temporal_gfn.utils.device import create_device_manager, get_optimal_batch_size
    from temporal_gfn.models.transformer import TemporalTransformerModel
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_device_setup():
    """Test device manager CPU setup."""
    print("\n=== Testing Device Manager ===")
    
    # Test automatic CPU detection
    print("1. Testing automatic device detection...")
    device_manager = create_device_manager(force_cpu=True, log_info=True)
    device = device_manager.get_device()
    
    if device.type == 'cpu':
        print("✓ Device correctly set to CPU")
    else:
        print(f"✗ Expected CPU, got {device}")
        return False
    
    # Test model creation and device placement
    print("\n2. Testing model creation and device placement...")
    model = TemporalTransformerModel(
        d_model=64,
        nhead=4, 
        d_hid=128,
        nlayers=2,
        dropout=0.1,
        k=10,
        context_length=48,
        prediction_horizon=12,
    )
    
    model = device_manager.to_device(model)
    model_device = next(model.parameters()).device
    
    if model_device.type == 'cpu':
        print("✓ Model correctly placed on CPU")
    else:
        print(f"✗ Model on wrong device: {model_device}")
        return False
    
    # Test optimal batch size calculation
    print("\n3. Testing optimal batch size calculation...")
    optimal_batch_size = get_optimal_batch_size(device, model, base_batch_size=32)
    print(f"✓ Optimal batch size for CPU: {optimal_batch_size}")
    
    # Test simple forward pass
    print("\n4. Testing model forward pass...")
    try:
        batch_size = 2
        context_length = 48
        pred_horizon = 12
        
        context = torch.randn(batch_size, context_length, 1)
        target = torch.zeros(batch_size, pred_horizon, 1)
        mask = torch.ones(batch_size, pred_horizon, dtype=torch.bool)
        
        with torch.no_grad():
            output = model(context, target, mask, step=0)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Configuration Loading ===")
    
    config_path = Path("configs/device/cpu_local_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ CPU configuration loaded successfully")
        print(f"  Force CPU: {config['device']['force_cpu']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Testing epochs: {config['testing']['epochs']}")
        return True
    else:
        print(f"✗ CPU config not found: {config_path}")
        return False

def test_environment():
    """Test environment setup."""
    print("\n=== Testing Environment ===")
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available: True")
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("CPU SETUP VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Configuration", test_config_loading), 
        ("Device Setup", test_device_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run CPU tests.")
        print("\nNext step: Run the main CPU tests with:")
        print("  ./run_cpu_tests.sh")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("Make sure you have:")
        print("  1. Activated the temporal_gfn environment") 
        print("  2. All dependencies installed")
        print("  3. Repository is in a clean state")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 