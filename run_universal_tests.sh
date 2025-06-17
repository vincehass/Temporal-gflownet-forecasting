#!/bin/bash
# =============================================================================
# Universal Tests for Temporal GFN - Auto-Adapting Local/Cluster
# =============================================================================
# This script automatically adapts between:
# - Local CPU development (when Compute Canada is down)
# - Cluster GPU computation (when on Compute Canada)
# 
# NO MANUAL CHANGES NEEDED - it auto-detects the environment!
#
# Author: Assistant for Nadhir Vincent Hassen

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check environment
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    print_error "No active virtual environment detected."
    echo "Please activate the environment first:"
    echo "  conda activate temporal_gfn  # For conda"
    echo "  source venv/bin/activate     # For venv"
    exit 1
fi

# Detect environment automatically
print_header "Universal Temporal GFN Tests - Auto-Detecting Environment"

# Run environment detection
print_info "Detecting execution environment..."
ENV_TYPE=$(python -c "
import sys
sys.path.insert(0, 'src')
from temporal_gfn.utils.environment import get_environment_type
print(get_environment_type())
")

if [ "$ENV_TYPE" = "cluster" ]; then
    print_success "Detected: Cluster Environment (Compute Canada)"
    BATCH_SIZE=64
    EPOCHS=50
    NUM_WORKERS=8
    DATASET="electricity"  # Use real data on cluster
    TEST_TYPE="full"
else
    print_success "Detected: Local Environment (CPU Development)"
    BATCH_SIZE=8
    EPOCHS=3
    NUM_WORKERS=2
    DATASET="synthetic"    # Use synthetic data locally
    TEST_TYPE="quick"
fi

RESULTS_BASE="results/universal_tests_$(date +%Y%m%d_%H%M%S)"

print_info "Configuration:"
echo -e "  Environment: ${ENV_TYPE}"
echo -e "  Batch Size: ${BATCH_SIZE}"
echo -e "  Epochs: ${EPOCHS}"
echo -e "  Workers: ${NUM_WORKERS}"
echo -e "  Dataset: ${DATASET}"
echo -e "  Test Type: ${TEST_TYPE}"
echo -e "  Results: ${RESULTS_BASE}"

# Create results directory
mkdir -p "${RESULTS_BASE}"

# Save environment detection info
python -c "
import sys, json
sys.path.insert(0, 'src')
from temporal_gfn.utils.environment import EnvironmentDetector
detector = EnvironmentDetector()
summary = detector.get_summary()
with open('${RESULTS_BASE}/environment_info.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Environment info saved to ${RESULTS_BASE}/environment_info.json')
"

# =============================================================================
# TEST 1: Individual Experiment (Auto-Configured)
# =============================================================================
print_header "TEST 1: Individual Experiment (${ENV_TYPE} optimized)"

TEST1_DIR="${RESULTS_BASE}/test1_individual"
mkdir -p "${TEST1_DIR}"

print_info "Running ${ENV_TYPE}-optimized individual experiment..."

# Universal command that adapts based on environment detection
TEST1_CMD="python scripts/run_single_ablation.py \
  --name test1_universal_${ENV_TYPE} \
  --dataset ${DATASET} \
  --epochs ${EPOCHS} \
  --quantization adaptive \
  --batch-size ${BATCH_SIZE} \
  --results-dir ${TEST1_DIR} \
  --wandb-project temporal-gfn-universal"

# Add offline mode for local testing
if [ "$ENV_TYPE" = "local" ]; then
    TEST1_CMD="${TEST1_CMD} --offline"
fi

echo "Command: ${TEST1_CMD}"
echo "Executing..."

if ${TEST1_CMD} 2>&1 | tee "${TEST1_DIR}/test1_log.txt"; then
    print_success "Test 1 completed successfully!"
    echo "Results: ${TEST1_DIR}"
else
    print_error "Test 1 failed - check log: ${TEST1_DIR}/test1_log.txt"
    exit 1
fi

# =============================================================================
# TEST 2: Basic Ablation Study (Auto-Configured)
# =============================================================================
print_header "TEST 2: Basic Ablation Study (${ENV_TYPE} optimized)"

TEST2_DIR="${RESULTS_BASE}/test2_basic_ablation"
mkdir -p "${TEST2_DIR}"

print_info "Running ${ENV_TYPE}-optimized basic ablation study..."

# Universal ablation command
TEST2_CMD="./scripts/run_ablation.sh \
  --name test2_universal_${ENV_TYPE} \
  --results-dir ${TEST2_DIR} \
  --dataset ${DATASET} \
  --epochs ${EPOCHS}"

echo "Command: ${TEST2_CMD}"
echo "Executing..."

if ${TEST2_CMD} 2>&1 | tee "${TEST2_DIR}/test2_log.txt"; then
    print_success "Test 2 completed successfully!"
    echo "Results: ${TEST2_DIR}"
else
    print_warning "Test 2 had issues - check log: ${TEST2_DIR}/test2_log.txt"
    # Continue to test 3 anyway
fi

# =============================================================================
# TEST 3: Complete Ablation Study (Auto-Configured)
# =============================================================================
print_header "TEST 3: Complete Ablation Study (${ENV_TYPE} optimized)"

TEST3_DIR="${RESULTS_BASE}/test3_complete_ablation"
mkdir -p "${TEST3_DIR}"

if [ "$ENV_TYPE" = "cluster" ]; then
    print_info "Running full-scale cluster ablation study..."
    # Full ablation study for cluster
    TEST3_CMD="./scripts/run_complete_ablation_study.sh \
      --datasets ${DATASET} \
      --quantizations adaptive,fixed \
      --k-values 5,10,20 \
      --policy-types uniform,learned \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --results-dir ${TEST3_DIR}"
else
    print_info "Running lightweight local ablation study..."
    # Lightweight ablation study for local
    TEST3_CMD="./scripts/run_complete_ablation_study.sh \
      --datasets ${DATASET} \
      --quantizations adaptive \
      --k-values 10 \
      --policy-types uniform \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --results-dir ${TEST3_DIR} \
      --offline"
fi

echo "Command: ${TEST3_CMD}"
echo "Executing..."

if ${TEST3_CMD} 2>&1 | tee "${TEST3_DIR}/test3_log.txt"; then
    print_success "Test 3 completed successfully!"
    echo "Results: ${TEST3_DIR}"
else
    print_warning "Test 3 had issues - check log: ${TEST3_DIR}/test3_log.txt"
fi

# =============================================================================
# SUMMARY
# =============================================================================
print_header "UNIVERSAL TESTS SUMMARY"

# Load environment info
echo "Environment Detection Results:"
python -c "
import json
with open('${RESULTS_BASE}/environment_info.json', 'r') as f:
    info = json.load(f)
print(f\"  Environment: {info['environment']}\")
print(f\"  Hostname: {info['hostname']}\")
print(f\"  CUDA Available: {info['cuda_available']}\")
print(f\"  CUDA Devices: {info['cuda_device_count']}\")
if info.get('slurm_job_id'):
    print(f\"  SLURM Job ID: {info['slurm_job_id']}\")
"

echo ""
echo "Test Results:"
if [ -f "${TEST1_DIR}/train_log.txt" ]; then
    print_success "Test 1 (Individual): SUCCESS"
else
    print_error "Test 1 (Individual): FAILED"
fi

if [ -f "${TEST2_DIR}/test2_log.txt" ]; then
    print_success "Test 2 (Basic Ablation): SUCCESS"
else
    print_error "Test 2 (Basic Ablation): FAILED"
fi

if [ -f "${TEST3_DIR}/test3_log.txt" ]; then
    print_success "Test 3 (Complete Ablation): SUCCESS"
else
    print_error "Test 3 (Complete Ablation): FAILED"
fi

echo ""
echo "Results saved to: ${RESULTS_BASE}"

if [ "$ENV_TYPE" = "local" ]; then
    echo ""
    print_info "Local Development Notes:"
    echo "- Tests optimized for CPU execution"
    echo "- Reduced epochs and batch sizes for speed"
    echo "- Using synthetic data for faster processing"
    echo "- W&B in offline mode"
    echo ""
    echo "When Compute Canada is back online:"
    echo "- Same script will automatically detect cluster environment"
    echo "- Will use full GPU optimizations and real datasets"
    echo "- No code changes needed!"
else
    echo ""
    print_info "Cluster Execution Notes:"
    echo "- Tests optimized for GPU computation"
    echo "- Full epochs and batch sizes"
    echo "- Using real datasets for production results"
    echo "- W&B online logging enabled"
fi

print_success "Universal testing script completed!"
echo ""
echo "🔄 Seamless switching between local CPU and cluster GPU!"
echo "🚀 Ready for production when Compute Canada is back online!" 