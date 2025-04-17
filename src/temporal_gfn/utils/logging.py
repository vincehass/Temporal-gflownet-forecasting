"""
Logging utilities for the temporal GFN model.
Primarily uses Weights & Biases for experiment tracking and visualization.
"""
import os
import json
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List
import logging

# Configure logger
logger = logging.getLogger(__name__)

class Logger:
    """Logger class that primarily uses W&B with optional TensorBoard support."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_wandb: bool = True,
        wandb_entity: str = "nadhirvincenthassen",
        wandb_project: str = "temporal-gfn",
        use_tensorboard: bool = False,  # Disabled by default
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_wandb: Whether to use Weights & Biases (primary logging tool)
            wandb_entity: WandB entity name
            wandb_project: WandB project name
            use_tensorboard: Whether to use TensorBoard (optional secondary logging)
            config: Configuration dictionary for the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.config = config or {}
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize WandB (primary)
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                if wandb.login():
                    self.wandb_run = wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=experiment_name,
                        config=config,
                        dir=os.path.join(log_dir, 'wandb'),
                    )
                    logger.info(f"WandB initialized with entity={wandb_entity}, project={wandb_project}")
                else:
                    logger.warning("WandB login failed. WandB logging disabled.")
                    self.use_wandb = False
            except ImportError:
                logger.warning("wandb package not found. WandB logging disabled.")
                self.use_wandb = False
        
        # Initialize TensorBoard (optional)
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
                logger.info(f"TensorBoard logs will be saved to {os.path.join(log_dir, 'tensorboard')}")
            except ImportError:
                logger.warning("torch.utils.tensorboard not found. TensorBoard logging disabled.")
                self.use_tensorboard = False
        
        # Save config to file
        if config:
            config_path = os.path.join(log_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to W&B and optionally TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/iteration
        """
        # Log to WandB (primary)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)
        
        # Log to TensorBoard (optional)
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.tb_writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.tb_writer.add_scalar(key, value.item(), step)
            self.tb_writer.flush()
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """
        Log histogram data.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Current step/iteration
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # Log to WandB (primary)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step)
        
        # Log to TensorBoard (optional)
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image, step: int):
        """
        Log image data.
        
        Args:
            name: Name of the image
            image: Image to log (matplotlib figure or numpy array)
            step: Current step/iteration
        """
        # For matplotlib figures
        if 'matplotlib.figure.Figure' in str(type(image)):
            # Log to WandB (primary)
            if self.use_wandb and self.wandb_run:
                import wandb
                wandb.log({name: wandb.Image(image)}, step=step)
            
            # Log to TensorBoard (optional)
            if self.use_tensorboard and self.tb_writer:
                self.tb_writer.add_figure(name, image, step)
        else:
            # Assume numpy array or torch tensor
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            # Log to WandB (primary)
            if self.use_wandb and self.wandb_run:
                import wandb
                wandb.log({name: wandb.Image(image)}, step=step)
            
            # Log to TensorBoard (optional)
            if self.use_tensorboard and self.tb_writer:
                self.tb_writer.add_image(name, image, step)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        # Log to WandB (primary) - update config
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.config.update(params)
            
        # Log to TensorBoard (optional)
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_hparams(params, {})
    
    def close(self):
        """Close all loggers."""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
            
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()


def create_logger(
    log_dir: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    wandb_entity: str = "nadhirvincenthassen",
    wandb_project: str = "temporal-gfn",
    use_tensorboard: bool = False,  # Disabled by default
) -> Logger:
    """
    Create a logger instance.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        config: Configuration dictionary for the experiment
        use_wandb: Whether to use Weights & Biases (primary)
        wandb_entity: WandB entity name
        wandb_project: WandB project name
        use_tensorboard: Whether to use TensorBoard (optional)
        
    Returns:
        Logger instance
    """
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        use_tensorboard=use_tensorboard,
        config=config,
    ) 