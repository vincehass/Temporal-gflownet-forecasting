"""
Adaptive quantization mechanism for Temporal GFN.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from src.temporal_gfn.quantization.base import compute_entropy, compute_histogram


class AdaptiveQuantization:
    """
    Adaptive quantization mechanism for Temporal GFN.
    
    This class implements the adaptive curriculum-based quantization mechanism
    that dynamically adjusts the number of quantization bins (K) based on
    the learning progress of the model.
    
    The adaptation process follows these steps:
    1. Periodically compute the reward and entropy statistics
    2. Calculate the learning progress indicator (eta_e)
    3. If eta_e < epsilon_adapt, increase K by delta_adapt
    4. Update the model's action space accordingly
    
    Attributes:
        k_initial (int): Initial number of quantization bins
        k_max (int): Maximum number of quantization bins
        vmin (float): Minimum value for quantization range
        vmax (float): Maximum value for quantization range
        lambda_adapt (float): Weight for the exponential moving average
        epsilon_adapt (float): Threshold for triggering adaptation
        delta_adapt (int): Number of bins to add when adapting
        update_interval (int): Number of steps between adaptation checks
    """
    
    def __init__(
        self,
        k_initial: int = 10,
        k_max: int = 100,
        vmin: float = -10.0,
        vmax: float = 10.0,
        lambda_adapt: float = 0.9,
        epsilon_adapt: float = 0.02,
        delta_adapt: int = 5,
        update_interval: int = 1000,
    ):
        """
        Initialize the adaptive quantization.
        
        Args:
            k_initial: Initial number of quantization bins
            k_max: Maximum number of quantization bins
            vmin: Minimum value for quantization range
            vmax: Maximum value for quantization range
            lambda_adapt: Weight for the exponential moving average
            epsilon_adapt: Threshold for triggering adaptation
            delta_adapt: Number of bins to add when adapting
            update_interval: Number of steps between adaptation checks
        """
        self.k = k_initial
        self.k_max = k_max
        self.vmin = vmin
        self.vmax = vmax
        self.lambda_adapt = lambda_adapt
        self.epsilon_adapt = epsilon_adapt
        self.delta_adapt = delta_adapt
        self.update_interval = update_interval
        
        # Initialize statistics
        self.rewards_ema = None
        self.entropy_ema = None
        self.steps = 0
        self.last_update = 0
        self.adaptation_log = []
    
    def update_statistics(
        self,
        rewards: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
    ):
        """
        Update reward and entropy statistics.
        
        Args:
            rewards: Tensor of rewards
            entropy: Optional tensor of entropy values
            values: Optional tensor of continuous values for computing histogram entropy
        """
        # Update rewards EMA
        reward_mean = rewards.mean().item()
        if self.rewards_ema is None:
            self.rewards_ema = reward_mean
        else:
            self.rewards_ema = self.lambda_adapt * self.rewards_ema + (1 - self.lambda_adapt) * reward_mean
        
        # Update entropy EMA
        if entropy is not None:
            entropy_mean = entropy.mean().item()
        elif values is not None:
            # Compute histogram entropy from continuous values
            entropy_mean = compute_entropy(values, self.vmin, self.vmax, self.k)
        else:
            # No entropy information provided
            entropy_mean = 0.0
            
        if self.entropy_ema is None:
            self.entropy_ema = entropy_mean
        else:
            self.entropy_ema = self.lambda_adapt * self.entropy_ema + (1 - self.lambda_adapt) * entropy_mean
        
        self.steps += 1
    
    def check_adaptation(self) -> bool:
        """
        Check if K should be adapted based on learning progress.
        
        Returns:
            bool: Whether K should be adapted
        """
        # Only check periodically
        if self.steps - self.last_update < self.update_interval:
            return False
        
        # Don't adapt if we've reached the maximum K
        if self.k >= self.k_max:
            return False
        
        # Don't adapt if we don't have enough statistics
        if self.rewards_ema is None or self.entropy_ema is None:
            return False
        
        # Compute learning progress indicator eta_e
        # In our case, we use the normalized entropy change as an indicator
        # Low eta_e means the model is converging in the current quantization level
        
        # Entropy should decrease over time, so negative differences are good
        # Calculate stable entropy change
        prev_entropy = self.entropy_ema * self.lambda_adapt
        current_contribution = self.entropy_ema - prev_entropy
        # We cap it at 0 to avoid artificially increasing eta_e when entropy increases
        entropy_change = max(0.0, current_contribution)
        
        # Normalize by current entropy with small epsilon for numerical stability
        if self.entropy_ema > 1e-8:
            eta_e = entropy_change / (self.entropy_ema + 1e-8)
        else:
            eta_e = 0.0
        
        # Log adaptation check
        self.adaptation_log.append({
            'steps': self.steps,
            'k': self.k,
            'rewards_ema': self.rewards_ema,
            'entropy_ema': self.entropy_ema,
            'eta_e': eta_e,
            'adapted': eta_e < self.epsilon_adapt
        })
        
        # Adapt if eta_e is below the threshold
        return eta_e < self.epsilon_adapt
    
    def adapt(self, model) -> int:
        """
        Adapt the quantization level by increasing K.
        
        Args:
            model: The model to update
            
        Returns:
            int: The new value of K
        """
        old_k = self.k
        new_k = min(self.k + self.delta_adapt, self.k_max)
        self.k = new_k
        
        # Update the model's action space
        model.update_action_space(new_k)
        
        self.last_update = self.steps
        
        return new_k
    
    def get_current_k(self) -> int:
        """
        Get the current number of quantization bins.
        
        Returns:
            int: Current K
        """
        return self.k
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dict containing current statistics
        """
        return {
            'k': self.k,
            'rewards_ema': self.rewards_ema,
            'entropy_ema': self.entropy_ema,
            'steps': self.steps,
            'adaptation_log': self.adaptation_log,
        } 