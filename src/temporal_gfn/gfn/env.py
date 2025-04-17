"""
GFN Environment for time series forecasting.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Any


class GFNEnvironment:
    """
    GFN Environment for time series forecasting.
    
    This environment defines the state as the time window and actions as
    the quantized bin indices for future time steps.
    
    Attributes:
        vmin (float): Minimum value for quantization
        vmax (float): Maximum value for quantization
        k (int): Number of quantization bins
        context_length (int): Length of the context window (T)
        prediction_horizon (int): Number of future steps to predict (T_prime)
        device (str): Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        vmin: float,
        vmax: float,
        k: int,
        context_length: int,
        prediction_horizon: int,
        device: str = 'cuda',
    ):
        """
        Initialize the environment.
        
        Args:
            vmin (float): Minimum value for quantization
            vmax (float): Maximum value for quantization
            k (int): Number of quantization bins
            context_length (int): Length of the context window (T)
            prediction_horizon (int): Number of future steps to predict (T_prime)
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.vmin = vmin
        self.vmax = vmax
        self._k = k
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        self._bin_width = (vmax - vmin) / k

    @property
    def k(self) -> int:
        """Get current number of quantization bins."""
        return self._k
    
    @k.setter
    def k(self, value: int):
        """Set number of quantization bins and update bin_width."""
        self._k = value
        self._bin_width = (self.vmax - self.vmin) / value
    
    @property
    def bin_width(self) -> float:
        """Get current bin width."""
        return self._bin_width
    
    @bin_width.setter
    def bin_width(self, value: float):
        """Set bin width directly (use with caution)."""
        self._bin_width = value

    def get_initial_state(self, context: torch.Tensor) -> Dict[str, Any]:
        """
        Returns the initial state (s0) given a context window.
        
        Args:
            context: Tensor of shape [batch_size, context_length] containing the context window
        
        Returns:
            state: Dictionary containing:
                - 'context': The input context window
                - 'forecast': Tensor of shape [batch_size, prediction_horizon] initialized with NaNs
                - 'mask': Tensor of shape [batch_size, prediction_horizon] with all False values
                - 'steps': Integer 0, representing the current step
        """
        batch_size = context.shape[0]
        forecast = torch.full((batch_size, self.prediction_horizon), float('nan'), device=self.device)
        mask = torch.zeros((batch_size, self.prediction_horizon), dtype=torch.bool, device=self.device)
        
        return {
            'context': context,
            'forecast': forecast,
            'mask': mask,
            'steps': 0
        }
    
    def step(self, state: Dict[str, Any], actions: torch.Tensor) -> Dict[str, Any]:
        """
        Take a step in the environment by adding a new forecasted value.
        
        Args:
            state: Current state dictionary
            actions: Tensor of shape [batch_size] containing quantized bin indices
        
        Returns:
            new_state: Updated state dictionary
        """
        batch_size = actions.shape[0]
        current_step = state['steps']
        
        # Dequantize the actions (bin indices) to continuous values
        values = self.dequantize(actions)
        
        # Create a new state with the updated forecast
        new_forecast = state['forecast'].clone()
        new_mask = state['mask'].clone()
        
        # Update the forecast at the current step for all batch elements
        batch_indices = torch.arange(batch_size, device=self.device)
        new_forecast[batch_indices, current_step] = values
        new_mask[batch_indices, current_step] = True
        
        return {
            'context': state['context'],
            'forecast': new_forecast,
            'mask': new_mask,
            'steps': current_step + 1
        }
    
    def get_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        Returns the set of possible actions (all K bins).
        
        Args:
            state: Current state dictionary
        
        Returns:
            List of integers from 0 to K-1 representing possible bin indices
        """
        return list(range(self.k))
    
    def is_terminal(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Check if the state is terminal (completed forecast).
        
        Args:
            state: Current state dictionary
        
        Returns:
            Boolean tensor of shape [batch_size] indicating if each sample is terminal
        """
        batch_size = state['context'].shape[0]
        return torch.full((batch_size,), state['steps'] >= self.prediction_horizon, dtype=torch.bool, device=self.device)
    
    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous values to bin indices.
        
        Args:
            values: Tensor of continuous values
        
        Returns:
            Tensor of bin indices (integers from 0 to K-1)
        """
        # Clip values to be within vmin and vmax
        clipped_values = torch.clamp(values, self.vmin, self.vmax)
        
        # Calculate bin indices
        bin_indices = torch.floor((clipped_values - self.vmin) / self.bin_width).long()
        
        # Ensure indices are within the valid range [0, K-1]
        return torch.clamp(bin_indices, 0, self.k - 1)
    
    def dequantize(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize bin indices to continuous values.
        
        Args:
            bin_indices: Tensor of bin indices (integers from 0 to K-1)
        
        Returns:
            Tensor of continuous values
        """
        # Map from bin indices to bin centers
        return self.vmin + (bin_indices + 0.5) * self.bin_width
    
    def reward(self, state: Dict[str, Any], targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate reward based on the forecasted trajectory and ground truth.
        
        Args:
            state: Terminal state dictionary
            targets: Ground truth values of shape [batch_size, prediction_horizon]
        
        Returns:
            Reward tensor of shape [batch_size]
        """
        forecast = state['forecast']
        
        # Handle NaN values by replacing them with zeros
        if torch.isnan(forecast).any():
            forecast = torch.where(torch.isnan(forecast), torch.zeros_like(forecast), forecast)
        
        # Calculate negative mean squared error as reward
        mse = torch.mean((forecast - targets) ** 2, dim=1)
        # Convert to a reward (higher is better), with numeric stability
        reward = torch.exp(-torch.clamp(mse, min=-88.0, max=88.0))  # Prevent overflow/underflow
        return reward
    
    def log_reward(self, state: Dict[str, Any], targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate log reward based on the forecasted trajectory and ground truth.
        
        Args:
            state: Terminal state dictionary
            targets: Ground truth values of shape [batch_size, prediction_horizon]
        
        Returns:
            Log reward tensor of shape [batch_size]
        """
        forecast = state['forecast']
        
        # Handle NaN values by replacing them with zeros
        if torch.isnan(forecast).any():
            forecast = torch.where(torch.isnan(forecast), torch.zeros_like(forecast), forecast)
        
        # Calculate negative mean squared error as log reward
        mse = torch.mean((forecast - targets) ** 2, dim=1)
        # Negative MSE is the log reward (numerically stable)
        log_reward = -torch.clamp(mse, min=-88.0, max=88.0)  # Prevent extreme values
        return log_reward 