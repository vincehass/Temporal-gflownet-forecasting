"""
Logging utilities for Temporal GFN.
"""
import os
import logging
import wandb
from typing import Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch

class Logger:
    """Logger class that uses W&B for experiment tracking with TensorBoard integration."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_mode: str = "online",
        use_tensorboard: bool = True,  # Enable by default for W&B integration
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            config: Configuration dictionary
            use_wandb: Whether to use W&B for logging
            wandb_entity: W&B entity name
            wandb_project: W&B project name
            wandb_name: W&B run name
            wandb_mode: W&B mode (online/offline)
            use_tensorboard: Whether to use TensorBoard (integrated with W&B)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, f'{experiment_name}.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Initialize W&B first
        self.wandb_run = None
        if use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_project or "temporal-gfn-forecasting",
                    entity=wandb_entity,
                    name=wandb_name or experiment_name,
                    config=config,
                    mode=wandb_mode,
                    dir=log_dir,
                )
                self.logger.info(f"W&B initialized. Run URL: {self.wandb_run.url}")
                
                # Enable TensorBoard integration with W&B
                if use_tensorboard:
                    try:
                        # Patch tensorboard to sync with W&B
                        wandb.tensorboard.patch(root_logdir=os.path.join(log_dir, 'tensorboard'))
                        self.logger.info("TensorBoard integration with W&B enabled")
                    except Exception as e:
                        self.logger.warning(f"Failed to enable TensorBoard-W&B integration: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # Initialize TensorBoard (will be synced to W&B if integration is enabled)
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_log_dir = os.path.join(log_dir, 'tensorboard')
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
                self.logger.info(f"TensorBoard logs will be saved to {tb_log_dir}")
                if self.use_wandb:
                    self.logger.info("TensorBoard logs will be automatically synced to W&B")
            except ImportError:
                self.logger.warning("torch.utils.tensorboard not found. TensorBoard logging disabled.")
                self.use_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log metrics to both W&B and TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to W&B: {e}")
        
        # Log to TensorBoard (will be synced to W&B automatically)
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step or 0)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log histogram to both W&B and TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Step number (optional)
        """
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log({name: wandb.Histogram(values)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log histogram to W&B: {e}")
        
        # Log to TensorBoard (will be synced to W&B automatically)
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(name, values, step or 0)
    
    def log_image(self, name: str, image: Union[plt.Figure, np.ndarray, torch.Tensor], step: Optional[int] = None):
        """
        Log image to both W&B and TensorBoard.
        
        Args:
            name: Name of the image
            image: Image to log (matplotlib figure, numpy array, or torch tensor)
            step: Step number (optional)
        """
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                if isinstance(image, plt.Figure):
                    self.wandb_run.log({name: wandb.Image(image)}, step=step)
                else:
                    self.wandb_run.log({name: wandb.Image(image)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log image to W&B: {e}")
        
        # Log to TensorBoard (will be synced to W&B automatically)
        if self.use_tensorboard and self.tb_writer:
            if isinstance(image, plt.Figure):
                self.tb_writer.add_figure(name, image, step or 0)
            elif isinstance(image, np.ndarray):
                self.tb_writer.add_image(name, image, step or 0, dataformats='HWC')
            elif isinstance(image, torch.Tensor):
                self.tb_writer.add_image(name, image, step or 0)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """
        Log text to both W&B and TensorBoard.
        
        Args:
            name: Name of the text log
            text: Text to log
            step: Step number (optional)
        """
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log({name: wandb.Html(text)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log text to W&B: {e}")
        
        # Log to TensorBoard (will be synced to W&B automatically)
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(name, text, step or 0)
    
    def close(self):
        """Close logger and cleanup resources."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
            
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish W&B run: {e}")


def create_logger(
    log_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
    use_wandb: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_mode: str = "online",
    use_tensorboard: bool = True,  # Enable by default for W&B integration
) -> Logger:
    """
    Create a logger instance.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        config: Configuration dictionary
        use_wandb: Whether to use W&B for logging
        wandb_entity: W&B entity name
        wandb_project: W&B project name
        wandb_name: W&B run name
        wandb_mode: W&B mode (online/offline)
        use_tensorboard: Whether to use TensorBoard (integrated with W&B)
    
    Returns:
        Logger instance
    """
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_mode=wandb_mode,
        use_tensorboard=use_tensorboard,
    ) 