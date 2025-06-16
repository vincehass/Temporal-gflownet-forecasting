"""
Device management utilities for Temporal GFN.
Provides modular CPU/GPU switching for local development and HPC clusters like Cedar.
"""
import os
import torch
import logging
from typing import Optional, Union, List, Dict, Any

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages device selection and configuration for training.
    Supports seamless switching between CPU and GPU(s).
    """
    
    def __init__(self, 
                 device: Optional[Union[str, torch.device]] = None,
                 force_cpu: bool = False,
                 gpu_id: Optional[int] = None,
                 multi_gpu: bool = False):
        """
        Initialize device manager.
        
        Args:
            device: Specific device to use ('cpu', 'cuda', 'cuda:0', etc.)
            force_cpu: Force CPU usage even if GPU is available
            gpu_id: Specific GPU ID to use
            multi_gpu: Enable multi-GPU training if available
        """
        self.force_cpu = force_cpu
        self.gpu_id = gpu_id
        self.multi_gpu = multi_gpu
        self.device = self._setup_device(device)
        self.device_info = self._get_device_info()
        
    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Setup and validate device configuration."""
        
        # Force CPU if requested
        if self.force_cpu:
            logger.info("Forcing CPU usage as requested")
            return torch.device('cpu')
        
        # Use specific device if provided
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            
            # Validate CUDA device
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return torch.device('cpu')
                
                if device.index is not None and device.index >= torch.cuda.device_count():
                    logger.warning(f"GPU {device.index} not available, using GPU 0")
                    return torch.device('cuda:0')
            
            return device
        
        # Auto-detect best device
        return self._auto_detect_device()
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            
            # Use specific GPU if requested
            if self.gpu_id is not None:
                if self.gpu_id < gpu_count:
                    device = torch.device(f'cuda:{self.gpu_id}')
                    logger.info(f"Using specified GPU {self.gpu_id}")
                else:
                    logger.warning(f"GPU {self.gpu_id} not available, using GPU 0")
                    device = torch.device('cuda:0')
            else:
                # Use first GPU by default
                device = torch.device('cuda:0')
                logger.info("Using GPU 0 (default)")
            
            return device
        
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple MPS device")
            return torch.device('mps')
        
        # Fall back to CPU
        else:
            logger.info("No GPU available, using CPU")
            return torch.device('cpu')
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            'device': str(self.device),
            'type': self.device.type,
            'available_gpus': 0,
            'gpu_names': [],
            'memory_info': {}
        }
        
        if torch.cuda.is_available():
            info['available_gpus'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) 
                               for i in range(torch.cuda.device_count())]
            
            if self.device.type == 'cuda':
                gpu_id = self.device.index or 0
                info['memory_info'] = {
                    'total': torch.cuda.get_device_properties(gpu_id).total_memory,
                    'allocated': torch.cuda.memory_allocated(gpu_id),
                    'cached': torch.cuda.memory_reserved(gpu_id)
                }
        
        return info
    
    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.device
    
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self.device.type in ['cuda', 'mps']
    
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.device.type == 'cpu'
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if self.device.type == 'cuda':
            gpu_id = self.device.index or 0
            return {
                'device': str(self.device),
                'total_memory': torch.cuda.get_device_properties(gpu_id).total_memory,
                'allocated_memory': torch.cuda.memory_allocated(gpu_id),
                'cached_memory': torch.cuda.memory_reserved(gpu_id),
                'free_memory': torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            }
        else:
            return {'device': str(self.device), 'memory_info': 'Not available for CPU/MPS'}
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def set_device(self, device: Union[str, torch.device]):
        """Change the current device."""
        old_device = self.device
        self.device = self._setup_device(device)
        self.device_info = self._get_device_info()
        logger.info(f"Device changed from {old_device} to {self.device}")
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or model to the configured device."""
        return tensor_or_model.to(self.device)
    
    def setup_for_training(self) -> Dict[str, Any]:
        """Setup device configuration for training."""
        config = {
            'device': str(self.device),
            'device_type': self.device.type,
            'is_gpu': self.is_gpu(),
            'multi_gpu_available': torch.cuda.device_count() > 1 if torch.cuda.is_available() else False,
            'device_info': self.device_info
        }
        
        # Set CUDA optimizations if using GPU
        if self.device.type == 'cuda':
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Enable cuDNN deterministic for reproducibility (optional)
            # torch.backends.cudnn.deterministic = True
            
            config['cuda_optimizations'] = {
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic
            }
        
        return config
    
    def log_device_info(self):
        """Log detailed device information."""
        logger.info("=== Device Configuration ===")
        logger.info(f"Selected device: {self.device}")
        logger.info(f"Device type: {self.device.type}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"    Compute Capability: {props.major}.{props.minor}")
            
            if self.device.type == 'cuda':
                gpu_id = self.device.index or 0
                memory_info = self.get_memory_info()
                logger.info(f"Current GPU memory usage:")
                logger.info(f"  Allocated: {memory_info['allocated_memory'] / 1024**3:.2f} GB")
                logger.info(f"  Cached: {memory_info['cached_memory'] / 1024**3:.2f} GB")
                logger.info(f"  Free: {memory_info['free_memory'] / 1024**3:.2f} GB")
        else:
            logger.info("CUDA available: False")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available: True")
        
        logger.info("============================")


def create_device_manager(
    device: Optional[Union[str, torch.device]] = None,
    force_cpu: bool = False,
    gpu_id: Optional[int] = None,
    multi_gpu: bool = False,
    log_info: bool = True
) -> DeviceManager:
    """
    Create a device manager instance.
    
    Args:
        device: Specific device to use
        force_cpu: Force CPU usage
        gpu_id: Specific GPU ID
        multi_gpu: Enable multi-GPU
        log_info: Log device information
    
    Returns:
        DeviceManager instance
    """
    manager = DeviceManager(
        device=device,
        force_cpu=force_cpu,
        gpu_id=gpu_id,
        multi_gpu=multi_gpu
    )
    
    if log_info:
        manager.log_device_info()
    
    return manager


def get_device_from_config(config: Dict[str, Any]) -> torch.device:
    """
    Get device from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device instance
    """
    device_config = config.get('device', {})
    
    if isinstance(device_config, str):
        # Simple string device specification
        return torch.device(device_config)
    
    # Detailed device configuration
    device_manager = create_device_manager(
        device=device_config.get('device'),
        force_cpu=device_config.get('force_cpu', False),
        gpu_id=device_config.get('gpu_id'),
        multi_gpu=device_config.get('multi_gpu', False),
        log_info=device_config.get('log_info', True)
    )
    
    return device_manager.get_device()


def setup_distributed_training(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training for multi-GPU/multi-node training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    logger.info(f"Distributed training setup: rank {rank}/{world_size}, backend {backend}")


# Environment variable helpers for HPC clusters
def setup_slurm_environment():
    """Setup environment variables for SLURM (used by Cedar)."""
    slurm_vars = {}
    
    # Common SLURM variables
    if 'SLURM_PROCID' in os.environ:
        slurm_vars['rank'] = int(os.environ['SLURM_PROCID'])
    if 'SLURM_NTASKS' in os.environ:
        slurm_vars['world_size'] = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_LOCALID' in os.environ:
        slurm_vars['local_rank'] = int(os.environ['SLURM_LOCALID'])
    if 'SLURM_NODEID' in os.environ:
        slurm_vars['node_id'] = int(os.environ['SLURM_NODEID'])
    
    # GPU information
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        slurm_vars['visible_gpus'] = os.environ['CUDA_VISIBLE_DEVICES']
    
    return slurm_vars


def get_optimal_batch_size(device: torch.device, model: torch.nn.Module, 
                          base_batch_size: int = 32) -> int:
    """
    Get optimal batch size based on device memory.
    
    Args:
        device: Target device
        model: Model to estimate memory for
        base_batch_size: Base batch size to scale from
        
    Returns:
        Optimal batch size
    """
    if device.type == 'cpu':
        # For CPU, use smaller batch sizes
        return min(base_batch_size, 16)
    
    elif device.type == 'cuda':
        # Estimate based on GPU memory
        gpu_id = device.index or 0
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        
        # Rough estimation: scale batch size with memory
        if total_memory > 20 * 1024**3:  # > 20GB
            return base_batch_size * 2
        elif total_memory > 10 * 1024**3:  # > 10GB
            return base_batch_size
        else:  # < 10GB
            return max(base_batch_size // 2, 8)
    
    else:  # MPS or other
        return base_batch_size 