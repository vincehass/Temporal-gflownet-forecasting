"""
Trainers for Temporal GFN models.
"""
import os
import sys
import time
import logging
from typing import Dict, Tuple, List, Any, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Restore tensorboard import for W&B integration
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import math
import copy
import numpy as np
from tqdm import tqdm

from src.temporal_gfn.models.transformer import TemporalTransformerModel
from src.temporal_gfn.gfn.env import GFNEnvironment
from src.temporal_gfn.gfn.policies import ForwardPolicy, BackwardPolicy
from src.temporal_gfn.gfn.tb_loss import (
    TrajectoryBalanceLoss,
    calculate_forward_probs,
    calculate_backward_uniform_probs,
    calculate_backward_learned_probs,
    calculate_forward_flow,
    calculate_backward_flow,
)
from src.temporal_gfn.gfn.sampling import (
    sample_forward_trajectory,
    sample_backward_trajectory,
    sample_trajectories_batch,
    sample_trajectories_batch_training,
)
from src.temporal_gfn.quantization.adaptive import AdaptiveQuantization
from src.temporal_gfn.utils.metrics import calculate_metrics
from src.temporal_gfn.utils.device import DeviceManager, create_device_manager, get_optimal_batch_size


class TemporalGFNTrainer:
    """
    Trainer for Temporal GFN models with modular CPU/GPU support.
    
    This trainer orchestrates the training process, including:
    - Forward and backward passes
    - Loss calculation
    - Optimization updates
    - Adaptive quantization
    - Logging and checkpointing
    - Modular device management (CPU/GPU switching)
    
    Attributes:
        env: GFN environment
        forward_policy: Forward policy wrapper
        backward_policy: Backward policy wrapper
        tb_loss: Trajectory Balance loss
        optimizer: Optimizer for model parameters
        device_manager: Device management for CPU/GPU switching
        adaptive_quant: Adaptive quantization mechanism
        logger: Logger for training progress
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        forward_model: TemporalTransformerModel,
        backward_model: Optional[TemporalTransformerModel] = None,
        device: Optional[Union[str, torch.device]] = None,
        force_cpu: bool = False,
        gpu_id: Optional[int] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            forward_model: Forward policy model
            backward_model: Backward policy model (optional)
            device: Device specification ('cpu', 'cuda', 'cuda:0', etc.)
            force_cpu: Force CPU usage even if GPU is available
            gpu_id: Specific GPU ID to use
        """
        self.config = config
        
        # Initialize device manager for modular CPU/GPU switching
        self.device_manager = create_device_manager(
            device=device,
            force_cpu=force_cpu,
            gpu_id=gpu_id,
            multi_gpu=config.get('training', {}).get('multi_gpu', False),
            log_info=True
        )
        self.device = self.device_manager.get_device()
        
        # Setup device optimizations
        device_config = self.device_manager.setup_for_training()
        
        # Optimize batch size based on device
        original_batch_size = config.get('training', {}).get('batch_size', 32)
        optimal_batch_size = get_optimal_batch_size(
            self.device, forward_model, original_batch_size
        )
        if optimal_batch_size != original_batch_size:
            self.logger.info(f"Adjusted batch size from {original_batch_size} to {optimal_batch_size} for device {self.device}")
            config['training']['batch_size'] = optimal_batch_size
        
        # Move models to device
        forward_model = self.device_manager.to_device(forward_model)
        if backward_model is not None:
            backward_model = self.device_manager.to_device(backward_model)
        
        # GFN Environment
        self.env = GFNEnvironment(
            vmin=config['quantization']['vmin'],
            vmax=config['quantization']['vmax'],
            k=config['quantization']['k_initial'],
            context_length=config['dataset']['context_length'],
            prediction_horizon=config['dataset']['prediction_horizon'],
            device=self.device,
        )
        
        # Forward policy
        self.forward_policy = ForwardPolicy(
            model=forward_model,
            use_ste=True,
            training_mode='sample',
            rand_action_prob=config['policy']['rand_action_prob'],
        )
        
        # Backward policy
        backward_policy_type = config['policy']['backward_policy_type']
        if backward_policy_type == 'uniform':
            self.backward_policy = BackwardPolicy(
                policy_type='uniform',
                prediction_horizon=config['dataset']['prediction_horizon']
            )
        else:  # 'learned'
            assert backward_model is not None, "Backward model must be provided for learned backward policy"
            self.backward_policy = BackwardPolicy(
                policy_type='learned',
                model=backward_model,
                prediction_horizon=config['dataset']['prediction_horizon']
            )
        
        # TB Loss
        self.tb_loss = TrajectoryBalanceLoss(
            Z_init=config['gfn']['Z_init'],
            Z_lr=config['gfn']['Z_lr'],
            lambda_entropy=config['gfn']['lambda_entropy']
        )
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(forward_model.parameters()) + 
            ([p for p in backward_model.parameters()] if backward_model is not None else []),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config['training'].get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Adaptive quantization
        if config['quantization'].get('adaptive', True):
            self.adaptive_quant = AdaptiveQuantization(
                k_initial=config['quantization']['k_initial'],
                k_max=config['quantization']['k_max'],
                vmin=config['quantization']['vmin'],
                vmax=config['quantization']['vmax'],
                lambda_adapt=config['quantization'].get('lambda_adapt', 0.9),
                epsilon_adapt=config['quantization'].get('epsilon_adapt', 0.02),
                delta_adapt=config['quantization'].get('delta_adapt', 5),
                update_interval=config['quantization'].get('update_interval', 1000),
            )
        else:
            self.adaptive_quant = None
        
        # Gradient clipping
        self.grad_clip_norm = config['training'].get('grad_clip_norm', None)
        
        # Logging and checkpointing
        self.log_interval = config['logging']['log_interval']
        self.checkpoint_interval = config['logging']['checkpoint_interval']
        self.results_dir = config['logging']['results_dir']
        
        # Create directories
        os.makedirs(os.path.join(self.results_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('TemporalGFN')
        self.logger.setLevel(logging.INFO)
        
        # Log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Log to file
        file_handler = logging.FileHandler(os.path.join(self.results_dir, 'logs', 'training.log'))
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # Training statistics
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        
        # Log device configuration
        self.logger.info(f"Trainer initialized with device: {self.device}")
        self.logger.info(f"Device type: {self.device.type}")
        if self.device_manager.is_gpu():
            memory_info = self.device_manager.get_memory_info()
            self.logger.info(f"GPU Memory: {memory_info['free_memory'] / 1024**3:.1f} GB free")
    
    def switch_device(self, new_device: Union[str, torch.device]):
        """
        Switch to a different device during training.
        
        Args:
            new_device: New device to switch to
        """
        old_device = self.device
        self.device_manager.set_device(new_device)
        self.device = self.device_manager.get_device()
        
        # Move models to new device
        self.forward_policy.model = self.device_manager.to_device(self.forward_policy.model)
        if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None:
            self.backward_policy.model = self.device_manager.to_device(self.backward_policy.model)
        
        # Update environment device
        self.env.device = self.device
        
        self.logger.info(f"Switched device from {old_device} to {self.device}")
        
        # Clear cache on old device if it was GPU
        if old_device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        return self.device_manager.get_memory_info()
    
    def clear_gpu_cache(self):
        """Clear GPU cache if using CUDA."""
        self.device_manager.clear_cache()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
    ):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Initial quantization bins K: {self.env.k}")
        
        # Log initial memory usage if GPU
        if self.device_manager.is_gpu():
            memory_info = self.get_memory_usage()
            self.logger.info(f"Initial GPU memory: {memory_info['allocated_memory'] / 1024**3:.2f} GB allocated")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            
            # Clear GPU cache periodically
            if self.device_manager.is_gpu() and epoch % 5 == 0:
                self.clear_gpu_cache()
                memory_info = self.get_memory_usage()
                self.logger.info(f"GPU memory after epoch {epoch}: {memory_info['allocated_memory'] / 1024**3:.2f} GB allocated")
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
                
                # Update learning rate scheduler if used
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                # Save checkpoint if best validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"New best validation loss: {val_loss:.6f}")
            
            # Regular checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed.")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation metrics: {self.best_val_metrics}")
        
        # Final memory cleanup
        if self.device_manager.is_gpu():
            self.clear_gpu_cache()
            final_memory = self.get_memory_usage()
            self.logger.info(f"Final GPU memory: {final_memory['allocated_memory'] / 1024**3:.2f} GB allocated")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Average loss and metrics for the epoch
        """
        # Set to training mode
        self.forward_policy.train()
        if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None:
            self.backward_policy.model.train()
        
        # Metrics for the epoch
        epoch_loss = 0.0
        epoch_forwards = 0.0
        epoch_backwards = 0.0
        epoch_rewards = 0.0
        epoch_entropies = 0.0
        epoch_metrics = {}
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        # Train on batches
        for batch_idx, batch in enumerate(pbar):
            # Get context and target windows
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            loss, metrics = self._training_step(context, target)
            
            # Update statistics
            epoch_loss += loss.item()
            epoch_forwards += metrics['forward']
            epoch_backwards += metrics['backward']
            epoch_rewards += metrics['reward']
            if 'entropy' in metrics:
                epoch_entropies += metrics['entropy']
            
            # Log to progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'k': self.env.k,
                'reward': metrics['reward']
            })
            
            # Increment global step
            self.global_step += 1
        
        # Compute epoch averages
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_forwards /= num_batches
        epoch_backwards /= num_batches
        epoch_rewards /= num_batches
        epoch_entropies /= num_batches
        
        # Log epoch statistics
        self.logger.info(
            f"Epoch {epoch} - Loss: {epoch_loss:.6f}, "
            f"Forward: {epoch_forwards:.6f}, Backward: {epoch_backwards:.6f}, "
            f"Reward: {epoch_rewards:.6f}, Entropy: {epoch_entropies:.6f}, "
            f"K: {self.env.k}"
        )
        
        # Return average loss and metrics
        return epoch_loss, {
            'forward': epoch_forwards,
            'backward': epoch_backwards,
            'reward': epoch_rewards,
            'entropy': epoch_entropies,
        }
    
    def _training_step(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform a single training step.
        
        Args:
            context: Context window
            target: Target values
            
        Returns:
            Loss tensor and metrics dictionary
        """
        self.optimizer.zero_grad()
        
        batch_size = context.shape[0]
        
        # Forward pass: sample trajectories and calculate probabilities
        states, actions, rewards, forward_logprobs, last_states, log_rewards = sample_trajectories_batch_training(
            self.env, self.forward_policy, batch_size, context, target
        )
        
        # Calculate forward flow
        # Convert lists to tensors for batch processing
        forward_flow = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            forward_flow[i] = forward_logprobs[i].sum() + log_rewards[i]
        
        # Calculate backward flow based on policy type
        backward_flow = torch.zeros(batch_size, device=self.device)
        if self.backward_policy.policy_type == 'uniform':
            # For uniform backward policy: product of 1/num_actions for each step
            log_step_prob = -torch.log(torch.tensor(self.env.k, dtype=torch.float32, device=self.device))
            traj_length = len(actions[0])  # Length of trajectory (same for all in batch)
            backward_flow = torch.full((batch_size,), traj_length * log_step_prob, device=self.device)
        else:  # 'learned'
            # For learned backward policy, we would need to calculate this differently
            # This part would need proper implementation for learned policies
            raise NotImplementedError("Learned backward policy flow calculation not implemented for batch training")
        
        # Calculate loss
        log_rewards_tensor = torch.tensor(log_rewards, device=self.device)
        
        # Create entropy tensor for batch
        batch_entropy = torch.zeros(batch_size, device=self.device)
        
        # Calculate TB loss (without entropy bonus for now)
        loss = self.tb_loss(forward_flow, backward_flow, log_rewards_tensor)
        
        # Add entropy bonus if needed
        entropy_bonus = 0.0
        if self.tb_loss.lambda_entropy > 0:
            # Calculate action entropy
            for i, logprobs in enumerate(forward_logprobs):
                if isinstance(logprobs, torch.Tensor) and logprobs.dim() > 0:
                    probs = torch.exp(logprobs)
                    batch_entropy[i] = -torch.sum(probs * logprobs)
            
            # Calculate entropy bonus
            entropy_bonus = self.tb_loss.lambda_entropy * batch_entropy.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if enabled
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.forward_policy.model.parameters()) + 
                ([p for p in self.backward_policy.model.parameters()] 
                 if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None else []),
                self.grad_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update adaptive quantization if enabled
        if self.adaptive_quant is not None:
            # Calculate reward statistics for detailed logging
            rewards_tensor = torch.tensor(rewards, device=self.device)
            reward_mean = rewards_tensor.mean().item()
            reward_var = rewards_tensor.var().item()
            reward_min = rewards_tensor.min().item()
            reward_max = rewards_tensor.max().item()
            
            # Calculate trajectory diversity metrics
            # Using action entropy as a proxy for trajectory diversity
            action_tensors = [torch.tensor(a, device=self.device) for a in actions]
            action_cat = torch.cat([a.float() for a in action_tensors])
            action_histogram = torch.histc(action_cat, bins=self.env.k, min=0, max=self.env.k-1)
            action_probs = action_histogram / action_histogram.sum()
            # Filter zero probabilities to avoid NaN in entropy calculation
            action_probs = action_probs[action_probs > 0]
            trajectory_diversity = -(action_probs * torch.log(action_probs)).sum().item()
            
            # Update the statistics in the adaptive quantization mechanism
            self.adaptive_quant.update_statistics(rewards_tensor, entropy_bonus)
            
            # Calculate additional quantization metrics
            unique_actions_count = len(torch.unique(action_cat))
            unique_actions_ratio = unique_actions_count / self.env.k
            
            # Count number of unique sequences
            unique_sequences = set()
            for seq in actions:
                unique_sequences.add(tuple(seq))
            unique_sequence_count = len(unique_sequences)
            
            # Calculate binning efficiency - how well are we using the available bins?
            # Compute normalized bin usage entropy
            if action_probs.shape[0] > 1:  # Need at least 2 bins with data
                bin_entropy = -(action_probs * torch.log(action_probs)).sum().item()
                # Normalize by maximum possible entropy (uniform distribution over all bins)
                max_entropy = np.log(self.env.k)
                normalized_bin_entropy = bin_entropy / max_entropy
            else:
                normalized_bin_entropy = 0.0
            
            # Log detailed quantization metrics
            self.logger.info(f"Trajectory stats: Diversity: {trajectory_diversity:.2f}, "
                            f"Unique actions: {unique_actions_count}/{self.env.k} ({unique_actions_ratio:.2f}), "
                            f"Unique sequences: {unique_sequence_count}, "
                            f"Bin entropy: {normalized_bin_entropy:.2f}, "
                            f"Min reward: {reward_min:.1f}, Max reward: {reward_max:.1f}, "
                            f"Mean reward: {reward_mean:.1f}, "
                            f"Log10 reward mean: {np.log10(max(1e-20, abs(reward_mean))):.2f}")
            
            # Check if we should adapt K
            if self.adaptive_quant.check_adaptation():
                old_k = self.env.k
                
                # Adapt K in the environment and policy models
                self.env.k = self.adaptive_quant.adapt(self.forward_policy)
                if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None:
                    self.backward_policy.model.update_action_space(self.env.k)
                
                # Get the adaptation details for logging
                latest_log = self.adaptive_quant.adaptation_log[-1]
                delta_t = 1 if old_k < self.env.k else -1 if old_k > self.env.k else 0
                
                # Log detailed adaptation information
                self.logger.info(f"Adaptive K updated: K: {old_k} -> {self.env.k}, "
                                f"Delta_t: {delta_t}, "
                                f"Reward variance: {reward_var:.2f}, "
                                f"Reward mean: {reward_mean:.2f}, "
                                f"Var/Mean ratio: {reward_var/abs(reward_mean):.4f}, "
                                f"Entropy EMA: {latest_log['entropy_ema']:.4f}, "
                                f"Learning indicator (eta_e): {latest_log['eta_e']:.6f}")
            
                # Sync the environment's K with adaptive quantization
                self.adaptive_quant.k = self.env.k
        
        # Create metrics dictionary
        metrics = {
            'loss': loss.item(),
            'forward': forward_flow.mean().item(),
            'backward': backward_flow.mean().item(),
            'reward': torch.tensor(rewards, device=self.device).mean().item(),
            'entropy': entropy_bonus.item() if entropy_bonus is not None else 0.0,
            'k': self.env.k
        }
        
        return loss, metrics
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss and metrics
        """
        # Set to evaluation mode
        self.forward_policy.eval()
        if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None:
            self.backward_policy.model.eval()
        
        # Metrics for validation
        val_loss = 0.0
        val_wql = 0.0
        val_crps = 0.0
        val_mase = 0.0
        
        # Number of validation samples per context
        num_samples = self.config['evaluation']['num_samples']
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get context and target windows
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Sample multiple forecasts
                forecast_samples = sample_trajectories_batch(
                    context_batch=context,
                    forward_policy=self.forward_policy,
                    backward_policy=self.backward_policy,
                    env=self.env,
                    num_samples=num_samples,
                    deterministic=False,
                    is_eval=True
                )
                
                # Calculate metrics
                metrics = calculate_metrics(
                    forecast_samples=forecast_samples,
                    targets=target,
                    insample=context,
                    quantiles=self.config['evaluation'].get('quantiles', None),
                    seasonality=self.config['evaluation'].get('seasonality', 1),
                )
                
                # Calculate loss
                # For each sample, calculate the TB loss
                batch_loss = 0.0
                for i in range(num_samples):
                    # Extract a single sample trajectory
                    sample = forecast_samples[:, i, :]
                    
                    # Quantize the continuous values to actions
                    actions = self.env.quantize(sample)
                    
                    # Reconstruct trajectory step by step
                    states = []
                    actions_list = []
                    for step in range(self.env.prediction_horizon):
                        if step == 0:
                            state = self.env.get_initial_state(context)
                        else:
                            state = self.env.step(states[-1], actions_list[-1])
                        states.append(state)
                        actions_list.append(actions[:, step])
                    
                    # Get the terminal state
                    terminal_state = states[-1]
                    
                    # Need to recompute forward logits to get the TB loss
                    action_logits_list = []
                    for step, state in enumerate(states[:-1]):
                        logits = self.forward_policy.model(
                            context=state['context'],
                            forecast=state['forecast'],
                            forecast_mask=state['mask'],
                            step=step
                        )
                        action_logits_list.append(logits)
                    
                    # Calculate forward log probabilities
                    forward_logprobs = calculate_forward_probs(action_logits_list, actions_list)
                    
                    # Calculate backward log probabilities
                    backward_logprobs = calculate_backward_uniform_probs(actions_list, self.env.k)
                    
                    # Calculate log rewards
                    log_rewards = self.env.log_reward(terminal_state, target)
                    
                    # Calculate TB loss (without entropy term for validation)
                    sample_loss = self.tb_loss(forward_logprobs, backward_logprobs, log_rewards)
                    batch_loss += sample_loss
                
                # Average loss over samples
                batch_loss /= num_samples
                
                # Update running totals
                val_loss += batch_loss.item()
                val_wql += metrics['wql'].mean().item()
                val_crps += metrics['crps'].mean().item()
                if 'mase' in metrics:
                    val_mase += metrics['mase'].mean().item()
        
        # Compute averages
        num_batches = len(val_loader)
        val_loss /= num_batches
        val_wql /= num_batches
        val_crps /= num_batches
        if 'mase' in metrics:
            val_mase /= num_batches
        
        # Log validation results
        self.logger.info(
            f"Validation - Loss: {val_loss:.6f}, WQL: {val_wql:.6f}, "
            f"CRPS: {val_crps:.6f}, MASE: {val_mase:.6f}"
        )
        
        # Return average loss and metrics
        return val_loss, {
            'wql': val_wql,
            'crps': val_crps,
            'mase': val_mase,
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'forward_model_state_dict': self.forward_policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tb_loss_state_dict': self.tb_loss.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'config': self.config,
            'k': self.env.k,
        }
        
        # Add backward model if applicable
        if hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None:
            checkpoint['backward_model_state_dict'] = self.backward_policy.model.state_dict()
        
        # Add scheduler if applicable
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add adaptive quantization if applicable
        if self.adaptive_quant is not None:
            checkpoint['adaptive_quant'] = self.adaptive_quant.get_stats()
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.results_dir, 'checkpoints', f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model if applicable
        if is_best:
            best_path = os.path.join(self.results_dir, 'checkpoints', 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.forward_policy.model.load_state_dict(checkpoint['forward_model_state_dict'])
        if (hasattr(self.backward_policy, 'model') and self.backward_policy.model is not None and
                'backward_model_state_dict' in checkpoint):
            self.backward_policy.model.load_state_dict(checkpoint['backward_model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load TB loss state
        self.tb_loss.load_state_dict(checkpoint['tb_loss_state_dict'])
        
        # Load scheduler if applicable
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update environment K
        if 'k' in checkpoint:
            self.env.k = checkpoint['k']
            if self.adaptive_quant is not None:
                self.adaptive_quant.k = checkpoint['k']
                self.forward_policy.model.update_action_space(checkpoint['k'])
        
        # Update other attributes
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metrics = checkpoint.get('best_val_metrics', {})
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return checkpoint.get('epoch', 0)

    def __del__(self):
        """Cleanup resources."""
        # Remove TensorBoard writer cleanup
        # # Close TensorBoard writer
        # if hasattr(self, 'tb_writer') and self.tb_writer is not None:
        #     self.tb_writer.close()

        # Remove TensorBoard logging
        # # Log to TensorBoard
        # if self.tb_writer:
        #     self.tb_writer.add_scalar('train/loss', loss.item(), self.global_step)
        #     self.tb_writer.add_scalar('train/tb_loss', tb_loss_value, self.global_step)
        #     self.tb_writer.add_scalar('train/entropy_loss', entropy_loss_value, self.global_step)
        #     self.tb_writer.add_scalar('train/reward_mean', reward_stats['mean'], self.global_step)
        #     self.tb_writer.add_scalar('train/reward_std', reward_stats['std'], self.global_step)
        #     self.tb_writer.add_scalar('train/trajectory_diversity', trajectory_stats['diversity'], self.global_step)
        #     self.tb_writer.add_scalar('train/unique_actions_ratio', trajectory_stats['unique_actions_ratio'], self.global_step)
        #     self.tb_writer.add_scalar('quantization/current_k', current_k, self.global_step)

        # Remove TensorBoard logging
        # # Also log to TensorBoard for visualization
        # if self.tb_writer:
        #     for key, value in metrics.items():
        #         if isinstance(value, (int, float)):
        #             self.tb_writer.add_scalar(f'eval/{key}', value, step)

        # Remove TensorBoard logging
        # # Log to TensorBoard
        # if self.tb_writer:
        #     self.tb_writer.add_scalar('eval/wql_mean', wql_mean, step)
        #     self.tb_writer.add_scalar('eval/mase', mase, step)
        #     for i, q in enumerate(quantiles):
        #         self.tb_writer.add_scalar(f'eval/wql_{q}', wql_values[i], step) 