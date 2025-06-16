#!/usr/bin/env python
import os
import yaml

# Create output directory
output_dir = "results/example_test"
os.makedirs(output_dir, exist_ok=True)

# Create a simple configuration
config = {
    'dataset': {
        'type': 'synthetic',
        'context_length': 96,
        'prediction_horizon': 24,
        'path': './datasets/eeg/eeg_data.npy',
        'train_path': './datasets/eeg/eeg_train.npy',
        'val_path': './datasets/eeg/eeg_val.npy',
        'test_path': './datasets/eeg/eeg_test.npy',
    },
    'model': {
        'd_model': 32,
        'nhead': 4,
        'nlayers': 2,
        'd_hid': 64,
        'dropout': 0.1,
        'uniform_init': True,
    },
    'gfn': {
        'Z_init': 0.0,
        'Z_lr': 0.01,
        'lambda_entropy': 0.01,
    },
    'quantization': {
        'adaptive': True,
        'k_initial': 10,
        'k_max': 100,
        'vmin': -10.0,
        'vmax': 10.0,
    },
    'policy': {
        'backward_policy_type': 'uniform',
    },
    'training': {
        'epochs': 1,
        'batch_size': 16,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'use_lr_scheduler': False,
        'grad_clip_norm': 1.0,
    },
    'validation': {
        'enabled': True,
    },
    'evaluation': {
        'batch_size': 16,
        'num_samples': 10,
    },
}

# Save the configuration
config_path = os.path.join(output_dir, 'config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print(f"Simple config saved to {config_path}")

# Also create a checkpoint directory
os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
print(f"Created checkpoints directory in {output_dir}") 