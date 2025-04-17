"""
Evaluation metrics for probabilistic time series forecasting.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union


def calculate_quantile_loss(
    forecasts: torch.Tensor,
    targets: torch.Tensor,
    quantile: float,
    aggregate: bool = True,
) -> torch.Tensor:
    """
    Calculate quantile loss for a specific quantile.
    
    Args:
        forecasts: Forecasted values of shape [batch_size, ...]
        targets: Ground truth values of shape [batch_size, ...]
        quantile: Quantile to evaluate (between 0 and 1)
        aggregate: Whether to aggregate the loss across all dimensions
        
    Returns:
        Quantile loss
    """
    # Calculate errors
    errors = targets - forecasts
    
    # Quantile loss function
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    
    # Aggregate if specified
    if aggregate:
        return loss.mean()
    else:
        return loss


def calculate_wql(
    forecast_samples: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Calculate Weighted Quantile Loss (WQL) across multiple quantiles.
    
    Args:
        forecast_samples: Samples of shape [batch_size, num_samples, prediction_horizon]
        targets: Ground truth values of shape [batch_size, prediction_horizon]
        quantiles: List of quantiles to evaluate
        weights: List of weights for each quantile
        
    Returns:
        WQL value
    """
    if quantiles is None:
        # Default quantiles: [0.1, 0.2, ..., 0.9]
        quantiles = [0.1 * q for q in range(1, 10)]
    
    if weights is None:
        # Default weights: equal weighting
        weights = [1.0 / len(quantiles)] * len(quantiles)
    
    # Ensure same length
    assert len(quantiles) == len(weights), "quantiles and weights must have the same length"
    
    # Get batch and prediction dimensions
    batch_size, num_samples, prediction_horizon = forecast_samples.shape
    
    # Compute empirical quantiles from samples
    wql_values = []
    for i, q in enumerate(quantiles):
        # Get the q-th quantile across samples for each batch and time step
        q_idx = int(num_samples * q)
        q_idx = max(0, min(q_idx, num_samples - 1))  # Ensure valid index
        
        # Sort samples along the sample dimension
        sorted_samples, _ = torch.sort(forecast_samples, dim=1)
        
        # Extract the q-th quantile forecasts
        q_forecasts = sorted_samples[:, q_idx, :]
        
        # Calculate quantile loss
        q_loss = calculate_quantile_loss(q_forecasts, targets, q, aggregate=False)
        
        # Apply weight
        wql_values.append(weights[i] * q_loss)
    
    # Sum weighted losses
    wql = torch.stack(wql_values).sum(dim=0)
    
    # Average across time steps
    return wql.mean(dim=1)


def calculate_mase(
    forecasts: torch.Tensor,
    targets: torch.Tensor,
    insample: torch.Tensor,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE = MAE / MAE_naive
    
    Args:
        forecasts: Forecasted values of shape [batch_size, prediction_horizon]
        targets: Ground truth values of shape [batch_size, prediction_horizon]
        insample: In-sample values of shape [batch_size, context_length]
        seasonality: Seasonality period for naive forecast
        eps: Small constant for numerical stability
        
    Returns:
        MASE value
    """
    # Calculate MAE
    mae = torch.abs(forecasts - targets).mean(dim=1)
    
    # Calculate in-sample naive forecast error
    naive_errors = []
    for i in range(seasonality, insample.shape[1]):
        naive_errors.append(torch.abs(insample[:, i] - insample[:, i - seasonality]))
    naive_errors = torch.stack(naive_errors, dim=1)
    
    # Calculate naive MAE
    naive_mae = naive_errors.mean(dim=1)
    
    # Calculate MASE
    mase = mae / (naive_mae + eps)
    
    return mase


def calculate_crps(
    forecast_samples: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate Continuous Ranked Probability Score (CRPS).
    
    This uses the empirical CDF approach for CRPS calculation.
    
    Args:
        forecast_samples: Samples of shape [batch_size, num_samples, prediction_horizon]
        targets: Ground truth values of shape [batch_size, prediction_horizon]
        
    Returns:
        CRPS value
    """
    batch_size, num_samples, prediction_horizon = forecast_samples.shape
    
    # Sort samples for each batch and prediction time
    sorted_samples, _ = torch.sort(forecast_samples, dim=1)
    
    # Calculate CRPS for each batch and time step
    crps_values = torch.zeros((batch_size, prediction_horizon), device=forecast_samples.device)
    
    for b in range(batch_size):
        for t in range(prediction_horizon):
            samples = sorted_samples[b, :, t]
            target = targets[b, t]
            
            # Empirical CDF approach
            n = num_samples
            
            # Compute distances
            right_terms = torch.sum(torch.abs(samples - target)) / n
            left_terms = 0.0
            for i in range(n):
                for j in range(n):
                    left_terms += torch.abs(samples[i] - samples[j])
            left_terms /= (2 * n * n)
            
            crps_values[b, t] = right_terms - left_terms
    
    # Average over time dimension
    return crps_values.mean(dim=1)


def calculate_metrics(
    forecast_samples: torch.Tensor,
    targets: torch.Tensor,
    insample: Optional[torch.Tensor] = None,
    quantiles: Optional[List[float]] = None,
    seasonality: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Calculate multiple evaluation metrics for probabilistic forecasts.
    
    Args:
        forecast_samples: Samples of shape [batch_size, num_samples, prediction_horizon]
        targets: Ground truth values of shape [batch_size, prediction_horizon]
        insample: In-sample values of shape [batch_size, context_length]
        quantiles: List of quantiles for WQL
        seasonality: Seasonality period for MASE
        
    Returns:
        Dictionary of metrics
    """
    batch_size, num_samples, prediction_horizon = forecast_samples.shape
    
    # Calculate point forecast (median)
    point_forecasts = torch.median(forecast_samples, dim=1)[0]
    
    # Calculate WQL
    wql = calculate_wql(forecast_samples, targets, quantiles)
    
    # Calculate CRPS
    crps = calculate_crps(forecast_samples, targets)
    
    # Calculate MASE if insample data is provided
    mase = None
    if insample is not None:
        mase = calculate_mase(point_forecasts, targets, insample, seasonality)
    
    # Return all metrics
    metrics = {
        'wql': wql,
        'crps': crps,
    }
    
    if mase is not None:
        metrics['mase'] = mase
    
    return metrics 