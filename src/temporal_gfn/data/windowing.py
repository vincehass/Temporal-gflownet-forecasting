
"""
Windowing utilities for time series data.
"""

import numpy as np
from typing import Tuple, List

def create_windows(data: np.ndarray, window_size: int, 
                  stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data."""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def create_context_target_windows(data: np.ndarray, context_length: int,
                                prediction_horizon: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create context and target windows for forecasting."""
    contexts = []
    targets = []
    
    total_length = context_length + prediction_horizon
    for i in range(0, len(data) - total_length + 1, stride):
        context = data[i:i + context_length]
        target = data[i + context_length:i + total_length]
        contexts.append(context)
        targets.append(target)
    
    return np.array(contexts), np.array(targets)
