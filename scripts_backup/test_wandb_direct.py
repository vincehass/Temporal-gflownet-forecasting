#!/usr/bin/env python3
"""
Simple test script to verify W&B is working correctly.
This script initializes W&B directly without any complex model training.
"""

import os
import sys
import wandb
import time
import random
import numpy as np

# Set environment variable to avoid OpenMP issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize W&B
run = wandb.init(
    project="temporal-gfn-forecasting",
    entity="nadhirvincenthassen",
    name=f"test_direct_{int(time.time())}",
    config={
        "dataset": "eeg",
        "model": "test",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
)

print(f"W&B run initialized with ID: {run.id}")
print(f"View run at: {run.get_url()}")

# Log some metrics
for epoch in range(5):
    # Simulate metrics
    loss = 1.0 - 0.15 * epoch + 0.02 * np.random.randn()
    accuracy = 0.7 + 0.05 * epoch + 0.01 * np.random.randn()
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
    })
    
    print(f"Epoch {epoch+1}/5 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    time.sleep(1)  # Give W&B time to sync

# Close the run
wandb.finish()
print("W&B run completed successfully")
print(f"Check results at: {run.get_url()}") 