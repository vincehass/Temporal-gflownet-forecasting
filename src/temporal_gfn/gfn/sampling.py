"""
Trajectory sampling for Temporal GFN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any, Optional

from src.temporal_gfn.gfn.env import GFNEnvironment
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy


def sample_forward_trajectory(
    context: torch.Tensor,
    forward_policy: ForwardPolicy,
    env: GFNEnvironment,
    deterministic: bool = False,
) -> Tuple[Dict[str, Any], List[torch.Tensor], List[torch.Tensor]]:
    """
    Sample a complete trajectory using the forward policy.
    
    Args:
        context: Input context window of shape [batch_size, context_length]
        forward_policy: Forward policy model
        env: GFN environment
        deterministic: Whether to use deterministic sampling
        
    Returns:
        terminal_state: Final state after sampling the trajectory
        actions: List of actions taken at each step
        action_logits: List of action logits from the policy at each step
    """
    batch_size = context.shape[0]
    device = context.device
    
    # Initialize state with context
    state = env.get_initial_state(context)
    
    # Lists to store trajectory information
    actions_list = []
    action_logits_list = []
    
    # Sample trajectory step-by-step
    step = 0
    while not env.is_terminal(state).all():
        # Use the forward policy to sample actions
        actions, _, logits = forward_policy(
            context=state['context'],
            forecast=state['forecast'],
            forecast_mask=state['mask'],
            step=step,
            deterministic=deterministic
        )
        
        # Store actions and logits
        actions_list.append(actions)
        action_logits_list.append(logits)
        
        # Update the state based on the actions
        state = env.step(state, actions)
        step += 1
    
    return state, actions_list, action_logits_list


def sample_backward_trajectory(
    context: torch.Tensor,
    forecast: torch.Tensor,
    backward_policy: BackwardPolicy,
    env: GFNEnvironment,
) -> Tuple[Dict[str, Any], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Sample a backward trajectory from a complete forecast.
    
    Args:
        context: Input context window of shape [batch_size, context_length]
        forecast: Complete forecast of shape [batch_size, prediction_horizon]
        backward_policy: Backward policy model
        env: GFN environment
        
    Returns:
        initial_state: Initial state after removing all forecast values (s0)
        positions: List of positions modified at each step
        position_logits: List of position logits from the policy (if learned)
    """
    batch_size = context.shape[0]
    device = context.device
    prediction_horizon = forecast.shape[1]
    
    # Create fully observed terminal state
    terminal_state = {
        'context': context,
        'forecast': forecast,
        'mask': torch.ones((batch_size, prediction_horizon), dtype=torch.bool, device=device),
        'steps': prediction_horizon
    }
    
    # Lists to store trajectory information
    positions_list = []
    position_logits_list = []
    
    # Current state starts as terminal state
    state = terminal_state.copy()
    state['forecast'] = terminal_state['forecast'].clone()
    state['mask'] = terminal_state['mask'].clone()
    
    # Sample backward trajectory by removing one forecast value at a time
    for step in range(prediction_horizon):
        # Use backward policy to sample which position to modify next
        positions, _, logits = backward_policy(
            context=state['context'],
            forecast=state['forecast'],
            forecast_mask=state['mask']
        )
        
        # Store positions and logits
        positions_list.append(positions)
        if logits is not None:
            position_logits_list.append(logits)
        
        # Update the state by "erasing" the selected positions
        batch_indices = torch.arange(batch_size, device=device)
        state['mask'][batch_indices, positions] = False
        # Set to NaN or a designated "empty" value
        state['forecast'][batch_indices, positions] = float('nan')
        state['steps'] -= 1
    
    # Return initial state (s0) and the trajectory
    initial_state = state
    
    # For uniform policy, position_logits_list will be empty
    position_logits = position_logits_list if position_logits_list else None
    
    return initial_state, positions_list, position_logits


def sample_trajectories(
    context: torch.Tensor,
    forward_policy: ForwardPolicy,
    backward_policy: BackwardPolicy,
    env: GFNEnvironment,
    num_trajectories: int = 1,
    deterministic: bool = False,
    is_eval: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """
    Sample multiple trajectories for probabilistic forecasting.
    
    Args:
        context: Input context window of shape [batch_size, context_length]
        forward_policy: Forward policy model
        backward_policy: Backward policy model
        env: GFN environment
        num_trajectories: Number of trajectories to sample per data point
        deterministic: Whether to use deterministic sampling
        is_eval: Whether this is evaluation mode (affects policy behavior)
        
    Returns:
        List of num_trajectories dictionaries, each containing:
            - 'forecast': Forecasted values
            - 'actions': Actions taken
    """
    batch_size = context.shape[0]
    device = context.device
    prediction_horizon = env.prediction_horizon
    
    # Set evaluation mode
    forward_policy.train(not is_eval)
    backward_policy.train(not is_eval)
    
    # List to store multiple trajectory results
    all_trajectories = []
    
    for _ in range(num_trajectories):
        # Sample a complete trajectory
        terminal_state, actions_list, _ = sample_forward_trajectory(
            context=context,
            forward_policy=forward_policy,
            env=env,
            deterministic=deterministic
        )
        
        # Convert actions to a tensor [batch_size, prediction_horizon]
        actions = torch.stack(actions_list, dim=1)
        
        # Store the trajectory
        all_trajectories.append({
            'forecast': terminal_state['forecast'],
            'actions': actions
        })
    
    return all_trajectories


def sample_trajectories_batch(
    context_batch: torch.Tensor,
    forward_policy: ForwardPolicy,
    backward_policy: BackwardPolicy,
    env: GFNEnvironment,
    num_samples: int = 100,
    deterministic: bool = False,
    is_eval: bool = False,
) -> torch.Tensor:
    """
    Sample multiple trajectories for a batch of data points.
    
    Args:
        context_batch: Batch of context windows, shape [batch_size, context_length]
        forward_policy: Forward policy model
        backward_policy: Backward policy model
        env: GFN environment
        num_samples: Number of trajectories to sample per data point
        deterministic: Whether to use deterministic sampling
        is_eval: Whether this is evaluation mode
        
    Returns:
        Sample trajectories of shape [batch_size, num_samples, prediction_horizon]
    """
    batch_size = context_batch.shape[0]
    device = context_batch.device
    
    # Set the model to evaluation mode if needed
    train_mode = not is_eval
    forward_policy.train(train_mode)
    backward_policy.train(train_mode)
    
    # Initialize storage for all samples
    all_samples = []
    
    # Generate samples for each batch element
    for i in range(batch_size):
        # Extract single context (add batch dimension back)
        context = context_batch[i:i+1]
        
        # Sample multiple trajectories
        trajectories = sample_trajectories(
            context=context,
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            env=env,
            num_trajectories=num_samples,
            deterministic=deterministic,
            is_eval=is_eval
        )
        
        # Extract forecasts from trajectories and stack
        forecasts = torch.stack([traj['forecast'][0] for traj in trajectories])
        all_samples.append(forecasts)
    
    # Stack along batch dimension
    return torch.stack(all_samples)


def sample_trajectories_batch_training(
    env: GFNEnvironment,
    forward_policy: ForwardPolicy,
    batch_size: int,
    context: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[List[Dict[str, torch.Tensor]], List[List[int]], List[float], List[torch.Tensor], List[Dict[str, torch.Tensor]], List[float]]:
    """
    Sample trajectories for training with the GFN.
    
    Args:
        env: GFN environment
        forward_policy: Forward policy
        batch_size: Batch size
        context: Context windows, shape [batch_size, context_length]
        target: Target values, shape [batch_size, prediction_horizon]
        
    Returns:
        Tuple of:
        - states: List of state dictionaries for each step
        - actions: List of action lists for each step
        - rewards: List of rewards
        - forward_logprobs: List of forward log probabilities
        - last_states: List of terminal states
        - log_rewards: List of log rewards
    """
    # Ensure batch size is consistent
    assert context.shape[0] == batch_size, f"Context batch size {context.shape[0]} doesn't match the provided batch size {batch_size}"
    assert target.shape[0] == batch_size, f"Target batch size {target.shape[0]} doesn't match the provided batch size {batch_size}"
    
    # Lists to store batch trajectories
    all_states = []
    all_actions = []
    all_rewards = []
    all_forward_logprobs = []
    all_last_states = []
    all_log_rewards = []
    
    # Sample trajectories for each item in the batch
    for i in range(batch_size):
        # Extract single context and target (add batch dimension back)
        single_context = context[i:i+1]
        single_target = target[i:i+1]
        
        # Sample a complete trajectory
        terminal_state, actions_list, action_logits_list = sample_forward_trajectory(
            context=single_context,
            forward_policy=forward_policy,
            env=env,
            deterministic=False
        )
        
        # Calculate forward log probabilities
        forward_logprobs = []
        for step in range(len(actions_list)):
            logits = action_logits_list[step]
            action = actions_list[step]
            log_prob = torch.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)
            forward_logprobs.append(log_prob)
        
        # Calculate reward
        log_reward = env.log_reward(terminal_state, single_target)
        reward = torch.exp(log_reward).item()
        
        # Reconstruct states for each step
        states = []
        state = env.get_initial_state(single_context)
        states.append(state)
        
        for step, action in enumerate(actions_list):
            state = env.step(state, action)
            if step < len(actions_list) - 1:  # Don't add the terminal state here
                states.append(state)
        
        # Store results
        all_states.append(states)
        all_actions.append([a.item() for a in actions_list])
        all_rewards.append(reward)
        all_forward_logprobs.append(torch.cat(forward_logprobs))
        all_last_states.append(terminal_state)
        all_log_rewards.append(log_reward.item())
    
    return all_states, all_actions, all_rewards, all_forward_logprobs, all_last_states, all_log_rewards 