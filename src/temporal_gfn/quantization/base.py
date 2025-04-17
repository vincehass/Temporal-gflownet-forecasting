"""
Base quantization utilities for Temporal GFN.
"""
import torch
import numpy as np
from typing import Tuple


def quantize(
    values: torch.Tensor,
    vmin: float,
    vmax: float,
    k: int
) -> torch.Tensor:
    """
    Quantize continuous values to discrete bin indices.
    
    Args:
        values: Tensor of continuous values
        vmin: Minimum value for quantization range
        vmax: Maximum value for quantization range
        k: Number of quantization bins
        
    Returns:
        Tensor of bin indices (integers from 0 to k-1)
    """
    # Clip values to be within [vmin, vmax]
    clipped_values = torch.clamp(values, vmin, vmax)
    
    # Calculate bin width
    bin_width = (vmax - vmin) / k
    
    # Calculate bin indices
    bin_indices = torch.floor((clipped_values - vmin) / bin_width).long()
    
    # Ensure indices are within the valid range [0, k-1]
    # This handles the edge case where values == vmax
    return torch.clamp(bin_indices, 0, k - 1)


def dequantize(
    bin_indices: torch.Tensor,
    vmin: float,
    vmax: float,
    k: int
) -> torch.Tensor:
    """
    Dequantize bin indices to continuous values.
    
    Args:
        bin_indices: Tensor of bin indices (integers from 0 to k-1)
        vmin: Minimum value for quantization range
        vmax: Maximum value for quantization range
        k: Number of quantization bins
        
    Returns:
        Tensor of continuous values (bin centers)
    """
    # Calculate bin width
    bin_width = (vmax - vmin) / k
    
    # Map from bin indices to bin centers
    return vmin + (bin_indices + 0.5) * bin_width


def get_bin_edges(
    vmin: float,
    vmax: float,
    k: int
) -> torch.Tensor:
    """
    Get the edges of each quantization bin.
    
    Args:
        vmin: Minimum value for quantization range
        vmax: Maximum value for quantization range
        k: Number of quantization bins
        
    Returns:
        Tensor of bin edges of shape [k+1]
    """
    # Calculate bin width
    bin_width = (vmax - vmin) / k
    
    # Create bin edges
    edges = torch.arange(k + 1, dtype=torch.float32) * bin_width + vmin
    
    return edges


def compute_histogram(
    values: torch.Tensor,
    vmin: float,
    vmax: float,
    k: int
) -> torch.Tensor:
    """
    Compute histogram of values across bins.
    
    Args:
        values: Tensor of continuous values
        vmin: Minimum value for quantization range
        vmax: Maximum value for quantization range
        k: Number of quantization bins
        
    Returns:
        Tensor of bin counts of shape [k]
    """
    # Quantize values to bin indices
    bin_indices = quantize(values, vmin, vmax, k)
    
    # Count occurrences of each bin index
    counts = torch.zeros(k, device=values.device)
    for i in range(k):
        counts[i] = (bin_indices == i).sum().float()
    
    return counts


def compute_entropy(
    values: torch.Tensor,
    vmin: float,
    vmax: float,
    k: int
) -> float:
    """
    Compute entropy of the histogram.
    
    Args:
        values: Tensor of continuous values
        vmin: Minimum value for quantization range
        vmax: Maximum value for quantization range
        k: Number of quantization bins
        
    Returns:
        Entropy value (scalar)
    """
    # Compute histogram
    hist = compute_histogram(values, vmin, vmax, k)
    
    # Normalize to get probabilities
    total = hist.sum()
    if total > 0:
        probs = hist / total
        
        # Compute entropy: -sum(p_i * log(p_i))
        # Avoid log(0) by filtering out zero probabilities
        non_zero_probs = probs[probs > 0]
        entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs))
        return entropy.item()
    else:
        return 0.0 