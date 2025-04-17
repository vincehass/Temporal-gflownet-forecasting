"""
Trajectory Balance (TB) Loss implementation for time series GFN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List


class TrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance (TB) Loss for GFlowNets.
    
    This loss follows the balance equation: Z * P_F(τ) = R(x) * P_B(τ|x),
    where:
    - Z is a learned partition function
    - P_F is the forward policy (probability of trajectory τ)
    - R(x) is the reward for the final state x
    - P_B is the backward policy (probability of trajectory τ given final state x)
    
    Attributes:
        log_Z (nn.Parameter): Learnable log partition function
        Z_lr (float): Learning rate for Z updates
        lambda_entropy (float): Entropy bonus coefficient
    """
    
    def __init__(self, Z_init: float = 0.0, Z_lr: float = 0.01, lambda_entropy: float = 0.01):
        """
        Initialize the TB Loss.
        
        Args:
            Z_init (float): Initial value for log(Z)
            Z_lr (float): Learning rate for Z updates
            lambda_entropy (float): Entropy bonus coefficient
        """
        super().__init__()
        self.log_Z = nn.Parameter(torch.tensor(Z_init, dtype=torch.float32))
        self.Z_lr = Z_lr
        self.lambda_entropy = lambda_entropy
    
    def forward(
        self,
        forward_logprobs: torch.Tensor,
        backward_logprobs: torch.Tensor,
        log_rewards: torch.Tensor,
        entropy: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the TB loss.
        
        Args:
            forward_logprobs: Log probabilities of the forward policy P_F(τ), shape [batch_size]
            backward_logprobs: Log probabilities of the backward policy P_B(τ|x), shape [batch_size]
            log_rewards: Log rewards log(R(x)) for the final states, shape [batch_size]
            entropy: Optional entropy of the forward policy, shape [batch_size]
            
        Returns:
            loss: TB loss value (scalar)
        """
        # TB loss: (log_Z + log P_F(τ) - log P_B(τ|x) - log R(x))^2
        tb_terms = self.log_Z + forward_logprobs - backward_logprobs - log_rewards
        tb_loss = tb_terms.pow(2).mean()
        
        # Add entropy bonus if provided
        if entropy is not None and self.lambda_entropy > 0:
            tb_loss = tb_loss - self.lambda_entropy * entropy.mean()
        
        return tb_loss
    
    def update_Z(self, forward_logprobs: torch.Tensor, backward_logprobs: torch.Tensor, log_rewards: torch.Tensor):
        """
        Update the partition function Z.
        
        Args:
            forward_logprobs: Log probabilities of the forward policy P_F(τ), shape [batch_size]
            backward_logprobs: Log probabilities of the backward policy P_B(τ|x), shape [batch_size]
            log_rewards: Log rewards log(R(x)) for the final states, shape [batch_size]
        """
        with torch.no_grad():
            # Update Z according to: log(Z) -= Z_lr * (log(Z) + log(P_F) - log(P_B) - log(R))
            delta = forward_logprobs - backward_logprobs - log_rewards
            self.log_Z.data -= self.Z_lr * delta.mean()


def calculate_forward_probs(
    action_logits: List[torch.Tensor],
    actions_taken: List[torch.Tensor],
) -> torch.Tensor:
    """
    Calculate the forward probability of a trajectory.
    
    Args:
        action_logits: List of action logits from the forward policy at each step, 
                      each of shape [batch_size, num_actions]
        actions_taken: List of actions taken at each step, each of shape [batch_size]
        
    Returns:
        forward_logprobs: Log probabilities of the trajectories, shape [batch_size]
    """
    batch_size = actions_taken[0].shape[0]
    forward_logprobs = torch.zeros(batch_size, device=actions_taken[0].device)
    
    for step_logits, step_actions in zip(action_logits, actions_taken):
        # Convert to log probabilities
        step_logprobs = F.log_softmax(step_logits, dim=1)
        
        # Get logprobs of the actions actually taken
        batch_indices = torch.arange(batch_size, device=step_actions.device)
        forward_logprobs += step_logprobs[batch_indices, step_actions]
    
    return forward_logprobs


def calculate_backward_uniform_probs(
    actions_taken: List[torch.Tensor],
    num_actions: int,
) -> torch.Tensor:
    """
    Calculate uniform backward probabilities for a trajectory.
    
    Args:
        actions_taken: List of actions taken at each step, each of shape [batch_size]
        num_actions: Number of possible actions at each step
        
    Returns:
        backward_logprobs: Log probabilities of the trajectories, shape [batch_size]
    """
    batch_size = actions_taken[0].shape[0]
    num_steps = len(actions_taken)
    
    # Uniform backward probability: product of 1/num_actions for each step
    log_step_prob = -torch.log(torch.tensor(num_actions, dtype=torch.float32))
    backward_logprobs = torch.full((batch_size,), num_steps * log_step_prob, 
                                  device=actions_taken[0].device)
    
    return backward_logprobs


def calculate_backward_learned_probs(
    backward_logits: List[torch.Tensor],
    actions_taken: List[torch.Tensor],
) -> torch.Tensor:
    """
    Calculate learned backward probabilities for a trajectory.
    
    Args:
        backward_logits: List of backward policy logits at each step,
                        each of shape [batch_size, num_actions]
        actions_taken: List of actions taken at each step, each of shape [batch_size]
        
    Returns:
        backward_logprobs: Log probabilities of the trajectories, shape [batch_size]
    """
    batch_size = actions_taken[0].shape[0]
    backward_logprobs = torch.zeros(batch_size, device=actions_taken[0].device)
    
    for step_logits, step_actions in zip(backward_logits, actions_taken):
        # Convert to log probabilities
        step_logprobs = F.log_softmax(step_logits, dim=1)
        
        # Get logprobs of the actions
        batch_indices = torch.arange(batch_size, device=step_actions.device)
        backward_logprobs += step_logprobs[batch_indices, step_actions]
    
    return backward_logprobs


def calculate_forward_flow(
    forward_logprobs: List[torch.Tensor],
    log_rewards: List[float],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Calculate the forward flow for a batch of trajectories.
    
    Args:
        forward_logprobs: List of forward log probabilities, one per batch item
        log_rewards: List of log rewards, one per batch item
        device: Device to create the tensor on
        
    Returns:
        forward_flow: Forward flow values, shape [batch_size]
    """
    batch_size = len(forward_logprobs)
    forward_flow = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        # Sum log probabilities for each batch item (already summed across steps)
        if isinstance(forward_logprobs[i], torch.Tensor):
            forward_flow[i] = forward_logprobs[i].sum() + log_rewards[i]
        else:
            # Handle case where it's already a scalar
            forward_flow[i] = forward_logprobs[i] + log_rewards[i]
    
    return forward_flow


def calculate_backward_flow(
    backward_policy: Any,
    last_states: List[Dict[str, torch.Tensor]],
    traj_length: int,
    num_actions: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Calculate the backward flow for a batch of trajectories with uniform policy.
    
    Args:
        backward_policy: The backward policy object
        last_states: List of terminal states, one per batch item
        traj_length: Length of each trajectory
        num_actions: Number of possible actions
        device: Device to create the tensor on
        
    Returns:
        backward_flow: Backward flow values, shape [batch_size]
    """
    batch_size = len(last_states)
    
    # For uniform backward policy: product of 1/num_actions for each step
    log_step_prob = -torch.log(torch.tensor(num_actions, dtype=torch.float32, device=device))
    backward_flow = torch.full((batch_size,), traj_length * log_step_prob, device=device)
    
    return backward_flow 