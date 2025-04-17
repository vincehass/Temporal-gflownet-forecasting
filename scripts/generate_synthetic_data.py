#!/usr/bin/env python
"""
Script to generate synthetic EEG-like data for testing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = "datasets/eeg"
os.makedirs(output_dir, exist_ok=True)

# Parameters
num_series = 100
series_length = 200
noise_level = 0.2

# Generate synthetic EEG-like data
def generate_eeg_like_data(num_series, series_length, noise_level=0.2):
    """
    Generate synthetic EEG-like data with multiple frequency components.
    
    Args:
        num_series: Number of time series to generate
        series_length: Length of each time series
        noise_level: Standard deviation of the noise
        
    Returns:
        Synthetic EEG-like data of shape [num_series, series_length]
    """
    time = np.linspace(0, 4 * np.pi, series_length)
    
    # Initialize data array
    data = np.zeros((num_series, series_length))
    
    # Generate different types of signals for each series
    for i in range(num_series):
        # Alpha waves (8-13 Hz)
        alpha_freq = np.random.uniform(8, 13)
        alpha_amp = np.random.uniform(0.5, 1.5)
        
        # Beta waves (13-30 Hz)
        beta_freq = np.random.uniform(13, 30)
        beta_amp = np.random.uniform(0.2, 0.8)
        
        # Theta waves (4-8 Hz)
        theta_freq = np.random.uniform(4, 8)
        theta_amp = np.random.uniform(0.3, 1.0)
        
        # Delta waves (0.5-4 Hz)
        delta_freq = np.random.uniform(0.5, 4)
        delta_amp = np.random.uniform(0.7, 2.0)
        
        # Combine waves with random phase shifts
        signal = (
            alpha_amp * np.sin(alpha_freq * time + np.random.uniform(0, 2 * np.pi)) +
            beta_amp * np.sin(beta_freq * time + np.random.uniform(0, 2 * np.pi)) +
            theta_amp * np.sin(theta_freq * time + np.random.uniform(0, 2 * np.pi)) +
            delta_amp * np.sin(delta_freq * time + np.random.uniform(0, 2 * np.pi))
        )
        
        # Add trending component for some series
        if i % 3 == 0:
            trend = np.linspace(0, np.random.uniform(1, 3), series_length)
            signal += trend
        
        # Add noise
        noise = np.random.normal(0, noise_level, series_length)
        
        # Combine signal and noise
        data[i] = signal + noise
    
    return data

# Generate full dataset
print("Generating synthetic EEG-like data...")
full_data = generate_eeg_like_data(num_series, series_length, noise_level)

# Split into train/val/test sets (70%, 15%, 15%)
train_idx = int(0.7 * num_series)
val_idx = int(0.85 * num_series)

train_data = full_data[:train_idx]
val_data = full_data[train_idx:val_idx]
test_data = full_data[val_idx:]

# Save the data
np.save(os.path.join(output_dir, "eeg_data.npy"), full_data)
np.save(os.path.join(output_dir, "eeg_train.npy"), train_data)
np.save(os.path.join(output_dir, "eeg_val.npy"), val_data)
np.save(os.path.join(output_dir, "eeg_test.npy"), test_data)

print(f"Full data shape: {full_data.shape}")
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Data saved to {output_dir}")

# Visualize some examples
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(full_data[i])
    plt.title(f"Example series {i+1}")
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eeg_examples.png"))
print(f"Example visualization saved to {output_dir}/eeg_examples.png")

if __name__ == "__main__":
    print("Data generation complete!") 