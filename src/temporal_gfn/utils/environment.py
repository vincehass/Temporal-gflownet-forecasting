"""
Environment detection and automatic configuration for seamless local/cluster switching.
Automatically detects whether running locally or on Compute Canada and configures accordingly.
"""

import os
import socket
import torch
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """
    Automatically detect execution environment and configure device settings.
    Supports seamless switching between local development and cluster computing.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize environment detector.
        
        Args:
            config_path: Path to auto-detection config file
        """
        self.config_path = config_path or Path("configs/device/auto_config.yaml")
        self.auto_config = self._load_auto_config()
        self.detected_env = self._detect_environment()
        self.device_config = self._get_device_config()
        
    def _load_auto_config(self) -> Dict[str, Any]:
        """Load auto-detection configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Auto-config not found: {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if auto-config file is missing."""
        return {
            'device': {'auto_detect': True, 'prefer_gpu': True, 'force_cpu': False},
            'environment': {
                'local': {
                    'device': {'force_cpu': True, 'multi_gpu': False},
                    'training': {'batch_size': 8, 'num_workers': 2, 'pin_memory': False},
                    'testing': {'epochs': 3, 'context_length': 48, 'prediction_horizon': 12}
                },
                'cluster': {
                    'device': {'force_cpu': False, 'multi_gpu': True},
                    'training': {'batch_size': 64, 'num_workers': 8, 'pin_memory': True},
                    'testing': {'epochs': 50, 'context_length': 96, 'prediction_horizon': 24}
                }
            },
            'detection_rules': {
                'slurm': {'env_vars': ['SLURM_JOB_ID'], 'config_override': 'cluster'},
                'hostname': {'patterns': ['cedar', 'graham', 'beluga'], 'config_override': 'cluster'},
                'gpu_detection': {'min_gpu_memory_gb': 8, 'config_override': 'cluster', 'fallback': 'local'}
            }
        }
    
    def _detect_environment(self) -> str:
        """
        Detect current execution environment.
        
        Returns:
            Environment type: 'local' or 'cluster'
        """
        detection_rules = self.auto_config.get('detection_rules', {})
        
        # Check for SLURM environment
        if self._check_slurm_environment(detection_rules.get('slurm', {})):
            logger.info("Detected SLURM environment (Compute Canada cluster)")
            return 'cluster'
        
        # Check hostname patterns
        if self._check_hostname_patterns(detection_rules.get('hostname', {})):
            logger.info("Detected cluster hostname pattern")
            return 'cluster'
        
        # Check GPU availability and specs
        gpu_env = self._check_gpu_environment(detection_rules.get('gpu_detection', {}))
        if gpu_env:
            logger.info(f"GPU detection result: {gpu_env}")
            return gpu_env
        
        # Default to local
        logger.info("No cluster environment detected, using local configuration")
        return 'local'
    
    def _check_slurm_environment(self, slurm_config: Dict[str, Any]) -> bool:
        """Check for SLURM environment variables."""
        env_vars = slurm_config.get('env_vars', [])
        for var in env_vars:
            if os.getenv(var):
                logger.info(f"Found SLURM environment variable: {var}={os.getenv(var)}")
                return True
        return False
    
    def _check_hostname_patterns(self, hostname_config: Dict[str, Any]) -> bool:
        """Check if hostname matches cluster patterns."""
        patterns = hostname_config.get('patterns', [])
        hostname = socket.gethostname().lower()
        
        for pattern in patterns:
            if pattern.lower() in hostname:
                logger.info(f"Hostname '{hostname}' matches cluster pattern '{pattern}'")
                return True
        return False
    
    def _check_gpu_environment(self, gpu_config: Dict[str, Any]) -> Optional[str]:
        """Check GPU availability and determine environment based on specs."""
        if not torch.cuda.is_available():
            return gpu_config.get('fallback', 'local')
        
        min_memory_gb = gpu_config.get('min_gpu_memory_gb', 8)
        gpu_count = torch.cuda.device_count()
        
        suitable_gpus = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            
            if memory_gb >= min_memory_gb:
                suitable_gpus += 1
                logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f} GB) - suitable for cluster")
            else:
                logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f} GB) - insufficient for cluster")
        
        if suitable_gpus > 0:
            return gpu_config.get('config_override', 'cluster')
        else:
            return gpu_config.get('fallback', 'local')
    
    def _get_device_config(self) -> Dict[str, Any]:
        """Get device configuration for detected environment."""
        env_config = self.auto_config.get('environment', {}).get(self.detected_env, {})
        base_config = self.auto_config.get('device', {})
        
        # Merge base config with environment-specific config
        device_config = base_config.copy()
        if 'device' in env_config:
            device_config.update(env_config['device'])
        
        return device_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for detected environment."""
        env_config = self.auto_config.get('environment', {}).get(self.detected_env, {})
        return env_config.get('training', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration for detected environment."""
        env_config = self.auto_config.get('environment', {}).get(self.detected_env, {})
        return env_config.get('testing', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration for detected environment."""
        env_config = self.auto_config.get('environment', {}).get(self.detected_env, {})
        return env_config.get('dataset', {})
    
    def apply_optimizations(self):
        """Apply environment-specific optimizations."""
        if self.detected_env == 'local' or self.device_config.get('force_cpu', False):
            self._apply_cpu_optimizations()
        else:
            self._apply_gpu_optimizations()
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations."""
        cpu_opts = self.auto_config.get('cpu_optimizations', {})
        
        for key, value in cpu_opts.items():
            if key == 'omp_num_threads':
                os.environ['OMP_NUM_THREADS'] = str(value)
            elif key == 'mkl_num_threads':
                os.environ['MKL_NUM_THREADS'] = str(value)
            elif key == 'numexpr_num_threads':
                os.environ['NUMEXPR_NUM_THREADS'] = str(value)
            elif key == 'torch_num_threads':
                torch.set_num_threads(value)
        
        logger.info(f"Applied CPU optimizations: {cpu_opts}")
    
    def _apply_gpu_optimizations(self):
        """Apply GPU-specific optimizations."""
        gpu_opts = self.auto_config.get('gpu_optimizations', {})
        
        if 'cudnn_benchmark' in gpu_opts:
            torch.backends.cudnn.benchmark = gpu_opts['cudnn_benchmark']
        
        if 'cudnn_deterministic' in gpu_opts:
            torch.backends.cudnn.deterministic = gpu_opts['cudnn_deterministic']
        
        logger.info(f"Applied GPU optimizations: {gpu_opts}")
    
    def override_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base configuration with environment-detected settings.
        
        Args:
            base_config: Base configuration dictionary
            
        Returns:
            Updated configuration with environment overrides
        """
        config = base_config.copy()
        
        # Override device settings
        if 'device' not in config:
            config['device'] = {}
        config['device'].update(self.device_config)
        
        # Override training settings
        training_config = self.get_training_config()
        if training_config:
            if 'training' not in config:
                config['training'] = {}
            config['training'].update(training_config)
        
        # Override testing settings (for ablation studies)
        testing_config = self.get_testing_config()
        if testing_config:
            # Apply to relevant sections
            for section in ['training', 'quantization', 'dataset']:
                if section not in config:
                    config[section] = {}
                
                if section == 'training' and 'epochs' in testing_config:
                    config[section]['epochs'] = testing_config['epochs']
                elif section == 'quantization' and 'k_initial' in testing_config:
                    config[section]['k_initial'] = testing_config['k_initial']
                elif section == 'dataset':
                    dataset_config = self.get_dataset_config()
                    if dataset_config:
                        config[section].update(dataset_config)
        
        # Add environment metadata
        config['environment_info'] = {
            'detected_env': self.detected_env,
            'hostname': socket.gethostname(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'auto_config_path': str(self.config_path)
        }
        
        return config
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detected environment and configuration."""
        return {
            'environment': self.detected_env,
            'hostname': socket.gethostname(),
            'device_config': self.device_config,
            'training_config': self.get_training_config(),
            'testing_config': self.get_testing_config(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'slurm_job_id': os.getenv('SLURM_JOB_ID')
        }
    
    def log_environment_info(self):
        """Log detailed environment information."""
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info("ENVIRONMENT DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Detected Environment: {summary['environment']}")
        logger.info(f"Hostname: {summary['hostname']}")
        logger.info(f"CUDA Available: {summary['cuda_available']}")
        logger.info(f"CUDA Device Count: {summary['cuda_device_count']}")
        
        if summary['slurm_job_id']:
            logger.info(f"SLURM Job ID: {summary['slurm_job_id']}")
        
        logger.info("Device Configuration:")
        for key, value in summary['device_config'].items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Training Configuration:")
        for key, value in summary['training_config'].items():
            logger.info(f"  {key}: {value}")
        
        if summary['testing_config']:
            logger.info("Testing Configuration:")
            for key, value in summary['testing_config'].items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)


def detect_and_configure(base_config: Optional[Dict[str, Any]] = None,
                        config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to detect environment and configure settings.
    
    Args:
        base_config: Base configuration to override
        config_path: Path to auto-detection config file
        
    Returns:
        Configuration with environment-specific overrides applied
    """
    detector = EnvironmentDetector(config_path)
    detector.apply_optimizations()
    
    if base_config is None:
        base_config = {}
    
    configured = detector.override_config(base_config)
    detector.log_environment_info()
    
    return configured


def get_environment_type() -> str:
    """
    Quick function to get detected environment type.
    
    Returns:
        Environment type: 'local' or 'cluster'
    """
    detector = EnvironmentDetector()
    return detector.detected_env


def is_cluster_environment() -> bool:
    """
    Quick function to check if running in cluster environment.
    
    Returns:
        True if running on cluster, False if local
    """
    return get_environment_type() == 'cluster' 