"""
Dataset classes for time series data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any

class TimeSeriesDataset(Dataset):
    """Basic time series dataset."""
    
    def __init__(self, data: np.ndarray = None, time_series: np.ndarray = None,
                 context_length: int = 96, prediction_horizon: int = 24, 
                 scaler_type: str = 'mean', stride: int = 1, sample_stride: int = 1, 
                 return_indices: bool = False, forecast_mode: bool = False):
        
        # Accept either data or time_series parameter
        if data is not None:
            self.data = data
        elif time_series is not None:
            self.data = time_series
        else:
            raise ValueError("Either 'data' or 'time_series' parameter must be provided")
            
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        self.stride = stride
        self.sample_stride = sample_stride
        self.return_indices = return_indices
        self.forecast_mode = forecast_mode
        
        # Apply scaling if specified
        if scaler_type == 'mean':
            self.data_mean = np.mean(self.data)
            self.data = self.data - self.data_mean
        elif scaler_type == 'standard':
            self.data_mean = np.mean(self.data)
            self.data_std = np.std(self.data)
            self.data = (self.data - self.data_mean) / self.data_std
        
    def __len__(self):
        total_length = self.context_length + self.prediction_horizon
        return max(0, (len(self.data) - total_length) // self.sample_stride + 1)
        
    def __getitem__(self, idx):
        start_idx = idx * self.sample_stride
        context = self.data[start_idx:start_idx + self.context_length]
        target = self.data[start_idx + self.context_length:start_idx + self.context_length + self.prediction_horizon]
        
        batch = {
            'context': torch.FloatTensor(context),
            'target': torch.FloatTensor(target)
        }
        
        if self.return_indices:
            batch['idx'] = idx
            
        return batch

class SyntheticTimeSeriesDataset(Dataset):
    """Synthetic time series dataset for testing."""
    
    def __init__(self, num_series: int = 100, series_length: int = 200,
                 context_length: int = 96, prediction_horizon: int = 24,
                 model_type: str = 'combined', model_params: Optional[Dict] = None,
                 noise_level: float = 0.1, scaler_type: str = 'mean',
                 stride: int = 1, sample_stride: int = 1, 
                 return_indices: bool = False, seed: Optional[int] = None):
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(42)  # Default seed for reproducibility
            
        self.num_series = num_series
        self.series_length = series_length
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model_params = model_params or {}
        self.noise_level = noise_level
        self.scaler_type = scaler_type
        self.stride = stride
        self.sample_stride = sample_stride
        self.return_indices = return_indices
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
        # Apply scaling if specified
        if scaler_type == 'mean':
            self.data_mean = np.mean(self.data)
            self.data = self.data - self.data_mean
        elif scaler_type == 'standard':
            self.data_mean = np.mean(self.data)
            self.data_std = np.std(self.data)
            self.data = (self.data - self.data_mean) / self.data_std
    
    def _generate_synthetic_data(self) -> np.ndarray:
        """Generate synthetic time series data based on model type."""
        all_series = []
        
        for _ in range(self.num_series):
            if self.model_type == 'ar':
                # Autoregressive model
                series = self._generate_ar_series()
            elif self.model_type == 'sine':
                # Sine wave model
                series = self._generate_sine_series()
            elif self.model_type == 'combined':
                # Combined AR + sine model
                series = self._generate_combined_series()
            else:
                # Default to simple sine wave
                series = self._generate_sine_series()
            
            all_series.append(series)
        
        # Concatenate all series
        return np.concatenate(all_series)
    
    def _generate_ar_series(self) -> np.ndarray:
        """Generate AR(1) time series."""
        phi = self.model_params.get('phi', 0.9)
        series = np.zeros(self.series_length)
        series[0] = np.random.normal(0, 1)
        
        for t in range(1, self.series_length):
            series[t] = phi * series[t-1] + np.random.normal(0, self.noise_level)
        
        return series
    
    def _generate_sine_series(self) -> np.ndarray:
        """Generate sine wave time series."""
        period = self.model_params.get('period', 20)
        amplitude = self.model_params.get('amplitude', 1.0)
        slope = self.model_params.get('slope', 0.01)
        
        t = np.arange(self.series_length)
        series = amplitude * np.sin(2 * np.pi * t / period) + slope * t
        series += np.random.normal(0, self.noise_level, self.series_length)
        
        return series
    
    def _generate_combined_series(self) -> np.ndarray:
        """Generate combined AR + sine wave time series."""
        # Start with AR component
        phi = self.model_params.get('phi', 0.9)
        ar_series = np.zeros(self.series_length)
        ar_series[0] = np.random.normal(0, 1)
        
        for t in range(1, self.series_length):
            ar_series[t] = phi * ar_series[t-1] + np.random.normal(0, self.noise_level * 0.5)
        
        # Add sine component
        period = self.model_params.get('period', 20)
        amplitude = self.model_params.get('amplitude', 1.0)
        slope = self.model_params.get('slope', 0.01)
        
        t = np.arange(self.series_length)
        sine_component = amplitude * np.sin(2 * np.pi * t / period) + slope * t
        
        # Combine
        combined_series = ar_series + sine_component
        combined_series += np.random.normal(0, self.noise_level * 0.5, self.series_length)
        
        return combined_series
        
    def __len__(self):
        total_length = self.context_length + self.prediction_horizon
        return max(0, (len(self.data) - total_length) // self.sample_stride + 1)
        
    def __getitem__(self, idx):
        start_idx = idx * self.sample_stride
        context = self.data[start_idx:start_idx + self.context_length]
        target = self.data[start_idx + self.context_length:start_idx + self.context_length + self.prediction_horizon]
        
        batch = {
            'context': torch.FloatTensor(context),
            'target': torch.FloatTensor(target)
        }
        
        if self.return_indices:
            batch['idx'] = idx
            
        return batch

def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 0,
                     pin_memory: bool = False, **kwargs) -> DataLoader:
    """Create a DataLoader for the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers, pin_memory=pin_memory, **kwargs)
