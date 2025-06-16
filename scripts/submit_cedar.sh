#!/bin/bash
#SBATCH --account=def-irina         # irina pays for your job
#SBATCH --cpus-per-task=6           # Ask for 6 CPUs
#SBATCH --gres=gpu:1                # Ask for 1 GPU
#SBATCH --mem=32G                   # Ask for 32 GB of RAM
#SBATCH --time=23:00:00             # The job will run for 23 hours
#SBATCH --job-name=temporal_gfn     # Job name
#SBATCH -o /scratch/vincehas/slurm-%j.out  # Write the log in $SCRATCH

# Cedar Compute Canada submission script for Temporal GFN training
# Following the exact pattern from test_cluster.sh

echo "Starting Temporal GFN training on Cedar cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load necessary modules (same as test_cluster.sh)
echo "Loading modules..."
module load StdEnv/2023
module load python/3.10
module load scipy-stack  # Includes numpy, pandas, matplotlib, etc.
module load arrow
module load cuda

# Set up additional module paths (same as test_cluster.sh)
echo "Setting up environment..."
export PATH=$HOME/.local/bin:$PATH
export PYTHONUSERBASE=$HOME/.local

echo "Starting job at: $(date)"
echo "Hostname: $(hostname)"

# Install PyTorch with pip (preferred method on Compute Canada)
echo "Installing PyTorch with pip..."
pip install --no-index torch torchvision torchaudio

# Install required packages for Temporal GFN (same as test_cluster.sh)
echo "Installing additional packages..."
pip install --user PyYAML hydra-core omegaconf matplotlib pandas scikit-learn tensorboard
pip install --user wandb tqdm

# Set W&B environment variables
export WANDB_MODE=online  # or offline for no internet
export WANDB_PROJECT=temporal-gfn-cedar
export WANDB_ENTITY=nadhirvincenthassen  # Use your actual W&B username

# Export CUDA_VISIBLE_DEVICES for GPU usage (same as test_cluster.sh)
export CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create a GPU test script to verify GPU availability (same as test_cluster.sh)
cat > minimal_gpu_test.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import torch

print("=== System Information ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\n=== GPU Information ===")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n=== Running GPU Test ===")
if torch.cuda.is_available():
    # Create tensors on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    
    # Perform operations
    z = torch.matmul(x, y)
    print(f"Matrix multiplication result shape: {z.shape}")
    
    z = torch.relu(z)
    print(f"ReLU result shape: {z.shape}")
    
    z = torch.mean(z, dim=1)
    print(f"Mean result shape: {z.shape}")
    
    print("\nGPU TEST SUCCESSFUL!")
else:
    print("CUDA NOT AVAILABLE! GPU TEST FAILED.")
EOL

# Make the script executable
chmod +x minimal_gpu_test.py

# Run the GPU test script
echo "Running GPU test script:"
python minimal_gpu_test.py

# If GPU test was successful, run the actual experiment
if [ $? -eq 0 ]; then
    echo "GPU test successful. Running Temporal GFN training..."
    
    # Test device switching first
    echo "Testing device switching..."
    python scripts/test_device_switching.py --test all
    
    # Run training with Cedar configuration
    echo "Starting full training..."
    python scripts/train.py \
        --config-name base_config \
        device=cedar_config \
        dataset=synthetic_config \
        model=transformer_config \
        quantization=adaptive_config \
        policy=uniform_config \
        training.epochs=100 \
        training.batch_size=64 \
        use_wandb=true \
        wandb_entity=nadhirvincenthassen \
        wandb_project=temporal-gfn-cedar \
        wandb_name="cedar_run_${SLURM_JOB_ID}" \
        results_dir="results/cedar_${SLURM_JOB_ID}"
    
    echo "Training completed!"
else
    echo "GPU test failed. Running minimal wrapper as fallback..."
    python minimal_wrapper.py
fi

echo "Job completed at: $(date)" 