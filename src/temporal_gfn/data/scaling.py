
"""
Scaling utilities for time series data.
"""

import numpy as np
from typing import Optional

class MeanScaler:
    """Scale data by subtracting mean."""
    
    def __init__(self):
        self.mean = None
        
    def fit(self, data: np.ndarray) -> 'MeanScaler':
        self.mean = np.mean(data)
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Scaler not fitted")
        return data - self.mean
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Scaler not fitted")
        return data + self.mean

class StandardScaler:
    """Standard scaling (z-score normalization)."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray) -> 'StandardScaler':
        self.mean = np.mean(data)
        self.std = np.std(data)
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
        return data * self.std + self.mean
