"""
Weights & Biases utilities for Temporal GFN project.
Includes optimization settings for efficient memory usage and offline mode support.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
import wandb

logger = logging.getLogger(__name__)

class WandbManager:
    """
    Manager for Weights & Biases integration with efficiency optimizations.
    Supports offline and online modes with memory optimization.
    """
    
    def __init__(
        self,
        entity: str = "nadhirvincenthassen",
        project: str = "temporal-gfn",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        offline: bool = False,
        log_dir: Optional[str] = None,
        save_code: bool = True,
        watch_model: bool = True,
        log_freq: int = 100,
        log_gradients: bool = False,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize WandB Manager.
        
        Args:
            entity: W&B entity name
            project: W&B project name
            experiment_name: Name of the experiment run
            config: Configuration dictionary
            offline: Whether to use offline mode
            log_dir: Directory to save logs
            save_code: Whether to save code to W&B
            watch_model: Whether to watch model parameters
            log_freq: Frequency of gradient logging if watch_model is True
            log_gradients: Whether to log gradients (memory intensive)
            settings: Additional W&B settings
        """
        self.entity = entity
        self.project = project
        self.experiment_name = experiment_name
        self.config = config or {}
        self.offline = offline
        self.log_dir = log_dir
        self.save_code = save_code
        self.watch_model = watch_model
        self.log_freq = log_freq
        self.log_gradients = log_gradients
        self.settings = settings or {}
        
        # Set default settings for improved efficiency
        self._set_default_settings()
        
        # Internal state
        self.run = None
        self.is_initialized = False
        self.watched_models = []
    
    def _set_default_settings(self):
        """Set default W&B settings for improved efficiency."""
        # Memory optimization settings
        if 'silent' not in self.settings:
            self.settings['silent'] = True  # Less console output
        
        # Enable symlinks to save storage
        if 'symlink' not in self.settings:
            self.settings['symlink'] = True
        
        # Set offline mode if requested
        if self.offline:
            os.environ["WANDB_MODE"] = "offline"
            logger.info("W&B set to offline mode, data will be saved locally only")
            
            # Create the directory for offline data
            if self.log_dir:
                os.makedirs(os.path.join(self.log_dir, 'wandb'), exist_ok=True)
    
    def init(self):
        """Initialize W&B run."""
        if self.is_initialized:
            logger.warning("WandB already initialized")
            return self.run
        
        try:
            # Set up offline directory
            if self.offline and self.log_dir:
                os.environ["WANDB_DIR"] = os.path.join(self.log_dir, 'wandb')
            
            # Initialize W&B
            self.run = wandb.init(
                entity=self.entity,
                project=self.project,
                name=self.experiment_name,
                config=self.config,
                dir=self.log_dir,
                save_code=self.save_code,
                settings=self.settings,
                # Prevent resuming by default - specify resume=True to override
                resume="never" 
            )
            
            self.is_initialized = True
            logger.info(f"WandB initialized with run ID: {self.run.id}")
            return self.run
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {str(e)}")
            return None
    
    def watch(self, model, log_freq=None, log_gradients=None):
        """
        Watch model parameters and gradients.
        
        Args:
            model: PyTorch model to watch
            log_freq: How often to log (steps)
            log_gradients: Whether to log gradients (memory intensive)
        """
        if not self.is_initialized:
            logger.warning("W&B not initialized, can't watch model")
            return
        
        if not self.watch_model:
            return
            
        log_freq = log_freq or self.log_freq
        log_gradients = log_gradients if log_gradients is not None else self.log_gradients
        
        grad_save_mode = "all" if log_gradients else None
        
        try:
            wandb.watch(
                model, 
                log="all", 
                log_freq=log_freq, 
                log_graph=False,  # Log graph is memory intensive
                criterion=None,
                save_mode=grad_save_mode
            )
            self.watched_models.append(model)
            logger.info(f"Model added to W&B watch with log_freq={log_freq}")
        except Exception as e:
            logger.warning(f"Error watching model: {str(e)}")
    
    def log(self, metrics, step=None, commit=True):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/iteration
            commit: Whether to commit the step immediately
        """
        if not self.is_initialized:
            logger.warning("W&B not initialized, can't log metrics")
            return
            
        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Error logging to W&B: {str(e)}")
    
    def save_checkpoint(self, checkpoint, checkpoint_name, metadata=None):
        """
        Save a model checkpoint to W&B.
        Use with caution as this can grow storage usage quickly.
        
        Args:
            checkpoint: Path to checkpoint or dict
            checkpoint_name: Name of checkpoint file
            metadata: Metadata to associate with the checkpoint
        """
        if not self.is_initialized:
            logger.warning("W&B not initialized, can't save checkpoint")
            return
            
        try:
            artifact = wandb.Artifact(
                name=f"checkpoint-{wandb.run.id}",
                type="model",
                metadata=metadata
            )
            
            if isinstance(checkpoint, str):
                # Path to file
                artifact.add_file(checkpoint, name=checkpoint_name)
            else:
                # Dictionary checkpoint
                path = os.path.join(wandb.run.dir, checkpoint_name)
                wandb.save(path)
                artifact.add_file(path, name=checkpoint_name)
                
            wandb.log_artifact(artifact)
            logger.info(f"Checkpoint {checkpoint_name} saved to W&B")
        except Exception as e:
            logger.warning(f"Error saving checkpoint to W&B: {str(e)}")
    
    def finish(self):
        """Finish the W&B run."""
        if not self.is_initialized:
            return
            
        try:
            wandb.finish()
            self.is_initialized = False
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {str(e)}")
    
    def sync_offline_runs(self):
        """
        Sync offline runs to W&B servers.
        Call this when internet connection is available after offline runs.
        """
        if not self.offline:
            logger.warning("Not in offline mode, no need to sync")
            return
            
        try:
            wandb.init()
            logger.info("Syncing offline W&B runs...")
            wandb.sync()
            logger.info("Offline W&B runs synced successfully")
        except Exception as e:
            logger.error(f"Error syncing offline runs: {str(e)}")
    
    def __enter__(self):
        """Support for context manager."""
        self.init()
        return self
    
    def __exit__(self, *args):
        """Cleanup when exiting context."""
        self.finish()


def create_wandb_manager(
    entity: str = "nadhirvincenthassen",
    project: str = "temporal-gfn",
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    offline: bool = False,
    log_dir: Optional[str] = None,
    save_code: bool = True,
    watch_model: bool = True,
    log_freq: int = 100,
    log_gradients: bool = False,
) -> WandbManager:
    """
    Create a W&B manager with memory optimized settings.
    
    Args:
        entity: W&B entity name
        project: W&B project name
        experiment_name: Name of the experiment run
        config: Configuration dictionary
        offline: Whether to use offline mode
        log_dir: Directory to save logs
        save_code: Whether to save code to W&B
        watch_model: Whether to watch model parameters
        log_freq: Frequency of gradient logging if watch_model is True
        log_gradients: Whether to log gradients (memory intensive)
        
    Returns:
        WandbManager instance
    """
    # Memory-efficient settings
    settings = {
        # Use symlinks to reduce storage usage
        'symlink': True,
        
        # Don't save the full program by default
        'program_relpath': False,
        
        # Control console output
        'silent': "false", 
        
        # Only include essential files
        'save_code': save_code
    }
    
    manager = WandbManager(
        entity=entity,
        project=project,
        experiment_name=experiment_name,
        config=config,
        offline=offline,
        log_dir=log_dir,
        save_code=save_code,
        watch_model=watch_model,
        log_freq=log_freq,
        log_gradients=log_gradients,
        settings=settings
    )
    
    return manager 