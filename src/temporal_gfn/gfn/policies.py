"""
Policy wrappers for GFlowNet with Straight-Through Estimator (STE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any, Optional


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for discrete actions.
    
    Forward: Use the discrete actions (hard samples).
    Backward: Use the differentiable distribution (soft samples).
    """
    
    @staticmethod
    def forward(ctx, logits, hard_samples):
        """
        Args:
            logits: Logits from the policy network
            hard_samples: Discrete actions (e.g., argmax or sampled from categorical)
        """
        # Save logits for backward pass
        ctx.save_for_backward(logits)
        # Pass through the hard samples unchanged
        return hard_samples
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Use the soft probabilities for gradient computation.
        """
        logits, = ctx.saved_tensors
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pass the gradient through the probabilities
        return grad_output.unsqueeze(-1) * probs, None


class ForwardPolicy(nn.Module):
    """
    Forward policy wrapper with Straight-Through Estimator.
    
    This policy predicts the next quantized value given the context and partial forecast.
    It uses STE during training to handle discrete actions.
    
    Attributes:
        model: The underlying policy network
        use_ste (bool): Whether to use Straight-Through Estimator
        training_mode (str): Either 'sample' for stochastic sampling or 'greedy' for argmax
        rand_action_prob (float): Probability of taking a random action for exploration
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ste: bool = True,
        training_mode: str = 'sample',
        rand_action_prob: float = 0.01,
    ):
        """
        Initialize the forward policy.
        
        Args:
            model: The underlying policy network
            use_ste: Whether to use Straight-Through Estimator
            training_mode: Either 'sample' or 'greedy'
            rand_action_prob: Probability of taking a random action
        """
        super().__init__()
        self.model = model
        self.use_ste = use_ste
        self.training_mode = training_mode
        self.rand_action_prob = rand_action_prob
    
    def forward(
        self,
        context: torch.Tensor,
        forecast: torch.Tensor,
        forecast_mask: torch.Tensor,
        step: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            context: Context window of shape [batch_size, context_length]
            forecast: Partial forecast of shape [batch_size, prediction_horizon]
            forecast_mask: Boolean mask of shape [batch_size, prediction_horizon]
            step: Current step in the prediction sequence
            deterministic: Whether to use deterministic action selection
            
        Returns:
            actions: Sampled discrete actions (bin indices)
            probs: Action probabilities
            logits: Raw logits from the policy network
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Get logits from the underlying model
        logits = self.model(context, forecast, forecast_mask, step)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        if not self.training or deterministic:
            # During evaluation or when explicitly requested, use deterministic (greedy) actions
            actions = torch.argmax(logits, dim=-1)
        else:
            if self.training_mode == 'sample':
                # Sample actions from the categorical distribution
                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample()
                
                # Apply exploration with random actions
                if self.rand_action_prob > 0:
                    random_actions = torch.randint(
                        0, self.model.k, (batch_size,), device=device
                    )
                    random_mask = torch.rand(batch_size, device=device) < self.rand_action_prob
                    actions = torch.where(random_mask, random_actions, actions)
            else:
                # Greedy actions
                actions = torch.argmax(logits, dim=-1)
        
        if self.training and self.use_ste:
            # Apply STE during training
            actions = StraightThroughEstimator.apply(logits, actions)
        
        return actions, probs, logits
    
    def sample_actions(
        self,
        context: torch.Tensor,
        forecast: torch.Tensor,
        forecast_mask: torch.Tensor,
        step: int,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample multiple actions for the same input.
        
        Args:
            context: Context window of shape [batch_size, context_length]
            forecast: Partial forecast of shape [batch_size, prediction_horizon]
            forecast_mask: Boolean mask of shape [batch_size, prediction_horizon]
            step: Current step in the prediction sequence
            num_samples: Number of samples to generate
            
        Returns:
            samples: Sampled actions of shape [batch_size, num_samples]
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Get logits from the underlying model
        logits = self.model(context, forecast, forecast_mask, step)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the categorical distribution multiple times
        samples = []
        for _ in range(num_samples):
            dist = torch.distributions.Categorical(probs=probs)
            sample = dist.sample()
            samples.append(sample)
        
        # Stack the samples
        return torch.stack(samples, dim=1)
    
    def update_action_space(self, new_k: int):
        """
        Update the action space size.
        
        Args:
            new_k: New number of quantization bins
        """
        self.model.update_action_space(new_k)


class BackwardPolicy(nn.Module):
    """
    Backward policy wrapper.
    
    This policy is used to predict which positions to modify given a complete forecast.
    For a uniform backward policy, this is a uniform distribution over all positions.
    For a learned backward policy, this is parameterized by a neural network.
    
    Attributes:
        policy_type (str): Either 'uniform' or 'learned'
        model: The underlying policy network (for learned policy)
        prediction_horizon (int): Number of future steps to predict
    """
    
    def __init__(
        self,
        policy_type: str = 'uniform',
        model: Optional[nn.Module] = None,
        prediction_horizon: int = 24,
    ):
        """
        Initialize the backward policy.
        
        Args:
            policy_type: Either 'uniform' or 'learned'
            model: The underlying policy network (for learned policy)
            prediction_horizon: Number of future steps to predict
        """
        super().__init__()
        self.policy_type = policy_type
        self.model = model
        self.prediction_horizon = prediction_horizon
        
        assert policy_type in ['uniform', 'learned']
        if policy_type == 'learned':
            assert model is not None, "Model must be provided for learned backward policy"
    
    def forward(
        self,
        context: torch.Tensor,
        forecast: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            context: Context window of shape [batch_size, context_length]
            forecast: Complete forecast of shape [batch_size, prediction_horizon]
            forecast_mask: Boolean mask of shape [batch_size, prediction_horizon]
            
        Returns:
            positions: Sampled positions to modify
            probs: Position probabilities
            logits: Raw logits (for learned policy) or None (for uniform policy)
        """
        batch_size = context.shape[0]
        device = context.device
        
        if self.policy_type == 'uniform':
            # Uniform distribution over all valid positions (where forecast_mask is True)
            valid_pos_count = forecast_mask.sum(dim=1).float()  # [batch_size]
            
            # Create uniform probabilities over valid positions
            probs = torch.zeros(batch_size, self.prediction_horizon, device=device)
            probs.masked_fill_(~forecast_mask, 0.0)
            
            # Normalize to sum to 1 for each batch element
            for i in range(batch_size):
                if valid_pos_count[i] > 0:
                    probs[i] = probs[i] / valid_pos_count[i]
            
            # Sample positions from the uniform distribution
            positions = []
            for i in range(batch_size):
                if valid_pos_count[i] > 0:
                    valid_indices = torch.where(forecast_mask[i])[0]
                    pos_idx = torch.randint(0, len(valid_indices), (1,), device=device)
                    positions.append(valid_indices[pos_idx])
                else:
                    # No valid positions (shouldn't happen in normal operation)
                    positions.append(torch.tensor(0, device=device))
            
            positions = torch.cat(positions)
            logits = None
        else:
            # Learned backward policy
            logits = self.model(context, forecast, forecast_mask)
            
            # Apply softmax to get probabilities (only over valid positions)
            masked_logits = logits.clone()
            masked_logits.masked_fill_(~forecast_mask, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)
            
            # Sample positions from the categorical distribution
            dist = torch.distributions.Categorical(probs=probs)
            positions = dist.sample()
        
        return positions, probs, logits
    
    def sample_positions(
        self,
        context: torch.Tensor,
        forecast: torch.Tensor,
        forecast_mask: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample multiple positions for the same input.
        
        Args:
            context: Context window of shape [batch_size, context_length]
            forecast: Complete forecast of shape [batch_size, prediction_horizon]
            forecast_mask: Boolean mask of shape [batch_size, prediction_horizon]
            num_samples: Number of samples to generate
            
        Returns:
            samples: Sampled positions of shape [batch_size, num_samples]
        """
        batch_size = context.shape[0]
        device = context.device
        
        if self.policy_type == 'uniform':
            # Uniform distribution over all valid positions
            samples = []
            for _ in range(num_samples):
                positions, _, _ = self.forward(context, forecast, forecast_mask)
                samples.append(positions)
        else:
            # Learned backward policy
            logits = self.model(context, forecast, forecast_mask)
            
            # Apply softmax to get probabilities (only over valid positions)
            masked_logits = logits.clone()
            masked_logits.masked_fill_(~forecast_mask, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)
            
            # Sample from the categorical distribution multiple times
            samples = []
            for _ in range(num_samples):
                dist = torch.distributions.Categorical(probs=probs)
                sample = dist.sample()
                samples.append(sample)
        
        # Stack the samples
        return torch.stack(samples, dim=1) 