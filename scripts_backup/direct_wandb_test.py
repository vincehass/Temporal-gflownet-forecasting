#!/usr/bin/env python3
"""
Simple script to test W&B logging directly.
This script will create a simple run that should appear on the W&B dashboard.
"""

import os
import sys
import argparse
import wandb
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Test W&B integration directly")
    parser.add_argument("--project", type=str, default="temporal-gfn-test",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default="nadhirvincenthassen",
                        help="W&B entity name")
    parser.add_argument("--name", type=str, default=f"direct-test-{int(time.time())}",
                        help="Run name")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set environment variable to handle OpenMP issues
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    print(f"Starting W&B run with:")
    print(f"- Entity: {args.entity}")
    print(f"- Project: {args.project}")
    print(f"- Name: {args.name}")
    
    # Initialize wandb directly
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        config={
            "test_param": 42,
            "learning_rate": 0.001,
            "architecture": "test",
            "dataset": "eeg",
            "epochs": 5
        }
    )
    
    if not run:
        print("Failed to initialize W&B run")
        return 1
        
    print(f"W&B run initialized with ID: {run.id}")
    print(f"View run at: {run.get_url()}")
    
    # Simulate training loop
    print("Logging metrics to W&B...")
    for epoch in range(5):
        # Simulate metrics
        train_loss = 1.0 - 0.15 * epoch + 0.02 * np.random.randn()
        val_loss = 1.2 - 0.1 * epoch + 0.03 * np.random.randn()
        accuracy = 0.7 + 0.05 * epoch + 0.01 * np.random.randn()
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "learning_rate": 0.001 * (0.9 ** epoch)
        })
        
        print(f"Epoch {epoch+1}/5: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
        time.sleep(1)  # Simulate training time
    
    # Create a simple plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Sample sine wave")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        # Log the plot
        wandb.log({"sample_plot": wandb.Image(plt)})
        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    # Finish the run
    wandb.finish()
    print("W&B run completed successfully")
    print(f"Visit {run.get_url()} to view your results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 