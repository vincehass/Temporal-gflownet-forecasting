# Modular Device System for Temporal GFN

## 🎯 **Overview**

This repository now includes a **fully modular device switching system** that seamlessly transitions between local CPU development and Compute Canada GPU clusters **without any manual code changes**.

## 🔄 **Key Features**

- **Automatic Environment Detection**: Detects local vs cluster environments
- **Zero Manual Configuration**: No code changes needed when switching environments
- **Optimal Resource Usage**: CPU-optimized for local, GPU-optimized for cluster
- **Seamless W&B Integration**: Offline locally, online on cluster
- **Production Ready**: Full-scale experiments on cluster, quick tests locally

## 📂 **System Components**

### 1. **Environment Detection** (`src/temporal_gfn/utils/environment.py`)

- Automatically detects execution environment
- Checks for SLURM variables, hostname patterns, GPU specs
- Provides environment-specific configurations

### 2. **Auto-Configuration** (`configs/device/auto_config.yaml`)

- Centralized configuration for all environments
- Environment-specific overrides
- Detection rules and optimization settings

### 3. **Device Management** (`src/temporal_gfn/utils/device.py`)

- Modular CPU/GPU switching
- Automatic device placement
- Memory optimization
- Performance tuning

### 4. **Universal Scripts**

- `run_universal_tests.sh`: Auto-adapting test runner
- `test_cpu_setup.py`: Local environment verification
- `run_cpu_tests.sh`: CPU-specific testing (legacy)

## 🚀 **Usage**

### **Single Command for All Environments**

```bash
# Works on both local CPU and cluster GPU automatically!
./run_universal_tests.sh
```

**That's it!** The script will:

- Detect your environment automatically
- Configure optimal settings for your hardware
- Run appropriate tests (quick locally, full on cluster)
- Save results with environment metadata

### **Manual Environment Detection**

```python
from temporal_gfn.utils.environment import get_environment_type, is_cluster_environment

# Check environment
env_type = get_environment_type()  # 'local' or 'cluster'
is_cluster = is_cluster_environment()  # True/False

# Auto-configure any base config
from temporal_gfn.utils.environment import detect_and_configure
config = detect_and_configure(base_config)
```

### **Device Manager Usage**

```python
from temporal_gfn.utils.device import create_device_manager

# Auto-detects and configures optimal device
device_manager = create_device_manager()
device = device_manager.get_device()

# Move models automatically
model = device_manager.to_device(model)
```

## 🔧 **Environment-Specific Configurations**

### **Local Development (CPU)**

```yaml
device:
  force_cpu: true
  multi_gpu: false
training:
  batch_size: 8 # Small for CPU
  epochs: 3 # Quick testing
  num_workers: 2 # CPU-optimized
dataset:
  num_series: 50 # Reduced dataset
  series_length: 100 # Faster processing
```

### **Cluster Computing (GPU)**

```yaml
device:
  force_cpu: false
  multi_gpu: true
training:
  batch_size: 64 # Large for GPU
  epochs: 50 # Full training
  num_workers: 8 # Multi-core optimized
dataset:
  num_series: 100 # Full dataset
  series_length: 200 # Production scale
```

## 🎛️ **Detection Rules**

The system detects environments using these rules (in order):

1. **SLURM Environment**: Checks for `SLURM_JOB_ID`, `SLURM_PROCID`
2. **Hostname Patterns**: Looks for `cedar`, `graham`, `beluga`, `narval`
3. **GPU Specifications**: Requires ≥8GB GPU memory for cluster classification
4. **Fallback**: Defaults to local environment

## 📊 **Performance Optimizations**

### **CPU Optimizations (Local)**

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
torch.set_num_threads(4)
```

### **GPU Optimizations (Cluster)**

```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Mixed precision enabled for modern GPUs
```

## 🧪 **Testing Scenarios**

### **Scenario 1: Local Development**

```bash
# On your laptop (CPU only)
./run_universal_tests.sh
```

**Result**:

- ✅ CPU-optimized settings automatically applied
- ✅ Quick tests with synthetic data (3 epochs)
- ✅ W&B offline mode
- ✅ Results in ~15-30 minutes

### **Scenario 2: Cluster Computing**

```bash
# On Compute Canada (GPU available)
./run_universal_tests.sh
```

**Result**:

- ✅ GPU-optimized settings automatically applied
- ✅ Full-scale tests with real datasets (50 epochs)
- ✅ W&B online logging
- ✅ Production-quality results

### **Scenario 3: Manual Environment Override**

```bash
# Force local mode even on cluster
TEMPORAL_GFN_ENV=local ./run_universal_tests.sh

# Force cluster mode for testing
TEMPORAL_GFN_ENV=cluster ./run_universal_tests.sh
```

## 🔍 **Verification & Debugging**

### **Test Environment Detection**

```bash
python test_cpu_setup.py
```

### **Check Current Configuration**

```python
from temporal_gfn.utils.environment import EnvironmentDetector

detector = EnvironmentDetector()
summary = detector.get_summary()
print(f"Environment: {summary['environment']}")
print(f"Device Config: {summary['device_config']}")
```

### **View Detection Logs**

```python
import logging
logging.basicConfig(level=logging.INFO)

from temporal_gfn.utils.environment import detect_and_configure
config = detect_and_configure()  # Will log detection process
```

## 📁 **File Structure**

```
├── configs/device/
│   ├── auto_config.yaml         # Main auto-detection config
│   ├── cpu_config.yaml          # CPU-specific settings
│   ├── gpu_config.yaml          # GPU-specific settings
│   ├── cpu_local_config.yaml    # Local development config
│   └── cedar_config.yaml        # Compute Canada config
├── src/temporal_gfn/utils/
│   ├── device.py                # Device management
│   └── environment.py           # Environment detection
├── run_universal_tests.sh       # Universal test runner
├── run_cpu_tests.sh             # Legacy CPU-only tests
├── test_cpu_setup.py            # Environment verification
└── MODULAR_DEVICE_SYSTEM.md     # This documentation
```

## 🔄 **Migration Guide**

### **From Manual Configuration**

**Before** (manual changes needed):

```bash
# Local
python train.py device=cpu batch_size=8 epochs=3

# Cluster
python train.py device=cuda batch_size=64 epochs=50
```

**After** (automatic):

```bash
# Works everywhere!
./run_universal_tests.sh
```

### **From Existing Scripts**

**Old approach**:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**New approach**:

```python
from temporal_gfn.utils.device import create_device_manager
device_manager = create_device_manager()
device = device_manager.get_device()
```

## 🚀 **Best Practices**

1. **Always use the universal script**: `./run_universal_tests.sh`
2. **Let auto-detection work**: Don't hardcode device settings
3. **Test locally first**: Quick validation before cluster submission
4. **Use environment metadata**: Check results for detection info
5. **Verify with test script**: Run `python test_cpu_setup.py` when in doubt

## 🎉 **Benefits**

- **🔄 Seamless Switching**: Same codebase works everywhere
- **⚡ Optimized Performance**: Best settings for each environment
- **🛠️ Developer Friendly**: No manual configuration
- **📊 Production Ready**: Full-scale experiments on clusters
- **🧪 Quick Testing**: Fast iteration on local machines
- **🔍 Transparent**: Full logging of detection process
- **📈 Scalable**: Easy to add new environments/optimizations

## 🏁 **Quick Start**

1. **Activate environment**: `conda activate temporal_gfn`
2. **Test setup**: `python test_cpu_setup.py`
3. **Run universal tests**: `./run_universal_tests.sh`
4. **Check results**: View auto-generated environment info in results

**That's it!** The system handles everything else automatically. 🎯

---

**Ready to switch between local development and cluster computing without any hassle!** 🚀
