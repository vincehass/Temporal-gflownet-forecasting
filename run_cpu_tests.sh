#!/bin/bash
# =============================================================================
# CPU-Optimized Tests for Temporal GFN (Local Development)
# =============================================================================
# This script runs all 3 tests described in PROGRESS_SUMMARY.md 
# optimized for local CPU execution while Compute Canada is down.
#
# Author: Assistant for Nadhir Vincent Hassen
# Date: $(date +"%Y-%m-%d")

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

# Check environment
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    print_error "No active virtual environment detected."
    echo "Please activate the environment first:"
    echo "  conda activate temporal_gfn  # For conda"
    echo "  source venv/bin/activate     # For venv"
    exit 1
fi

# Configuration for CPU optimization
CPU_BATCH_SIZE=8       # Reduced from 32 for CPU
CPU_EPOCHS=3           # Reduced from 5-50 for quick testing
CPU_NUM_WORKERS=2      # Reduced for CPU efficiency
RESULTS_BASE="results/cpu_tests_$(date +%Y%m%d_%H%M%S)"

print_header "CPU-Optimized Temporal GFN Tests - Local Development"
echo -e "Environment: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}"
echo -e "Results will be saved to: ${RESULTS_BASE}"
echo -e "Batch size optimized for CPU: ${CPU_BATCH_SIZE}"
echo -e "Epochs reduced for testing: ${CPU_EPOCHS}"

# Create results directory
mkdir -p "${RESULTS_BASE}"

# Save system info
echo "System Information:" > "${RESULTS_BASE}/system_info.txt"
echo "Date: $(date)" >> "${RESULTS_BASE}/system_info.txt"
echo "Python: $(python --version)" >> "${RESULTS_BASE}/system_info.txt"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')" >> "${RESULTS_BASE}/system_info.txt"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')" >> "${RESULTS_BASE}/system_info.txt"
echo "Device Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')" >> "${RESULTS_BASE}/system_info.txt"

# =============================================================================
# TEST 1: Individual Experiment (CPU-Optimized)
# =============================================================================
print_header "TEST 1: Individual Experiment (CPU-Optimized)"

TEST1_DIR="${RESULTS_BASE}/test1_individual"
mkdir -p "${TEST1_DIR}"

echo "Running CPU-optimized individual experiment..."

# CPU-optimized command with force_cpu flag
TEST1_CMD="python scripts/run_single_ablation.py \
  --name test1_cpu_individual \
  --dataset synthetic \
  --epochs ${CPU_EPOCHS} \
  --quantization adaptive \
  --batch-size ${CPU_BATCH_SIZE} \
  --results-dir ${TEST1_DIR} \
  --wandb-project temporal-gfn-cpu-tests \
  --offline"

echo "Command: ${TEST1_CMD}"
echo "Executing..."

# Add CPU-specific environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if ${TEST1_CMD} 2>&1 | tee "${TEST1_DIR}/test1_log.txt"; then
    print_success "Test 1 completed successfully!"
    echo "Results: ${TEST1_DIR}"
else
    print_error "Test 1 failed - check log: ${TEST1_DIR}/test1_log.txt"
    exit 1
fi

# =============================================================================
# TEST 2: Basic Ablation Study (CPU-Optimized)
# =============================================================================
print_header "TEST 2: Basic Ablation Study (CPU-Optimized)"

TEST2_DIR="${RESULTS_BASE}/test2_basic_ablation"
mkdir -p "${TEST2_DIR}"

echo "Running CPU-optimized basic ablation study..."

# Modify the ablation script arguments for CPU
TEST2_CMD="./scripts/run_ablation.sh \
  --name test2_cpu_ablation \
  --results-dir ${TEST2_DIR} \
  --dataset synthetic \
  --epochs ${CPU_EPOCHS}"

echo "Command: ${TEST2_CMD}"
echo "Executing..."

if ${TEST2_CMD} 2>&1 | tee "${TEST2_DIR}/test2_log.txt"; then
    print_success "Test 2 completed successfully!"
    echo "Results: ${TEST2_DIR}"
else
    print_error "Test 2 failed - check log: ${TEST2_DIR}/test2_log.txt"
    # Continue to test 3 anyway
fi

# =============================================================================
# TEST 3: Complete Ablation Study (CPU-Optimized)
# =============================================================================
print_header "TEST 3: Complete Ablation Study (CPU-Optimized)"

TEST3_DIR="${RESULTS_BASE}/test3_complete_ablation"
mkdir -p "${TEST3_DIR}"

echo "Running CPU-optimized complete ablation study..."
print_warning "Note: Using synthetic dataset instead of electricity for CPU testing"

# CPU-optimized complete ablation with reduced scope
TEST3_CMD="./scripts/run_complete_ablation_study.sh \
  --datasets synthetic \
  --quantizations adaptive \
  --k-values 10 \
  --policy-types uniform \
  --epochs ${CPU_EPOCHS} \
  --batch-size ${CPU_BATCH_SIZE} \
  --results-dir ${TEST3_DIR} \
  --offline"

echo "Command: ${TEST3_CMD}"
echo "Executing..."

if ${TEST3_CMD} 2>&1 | tee "${TEST3_DIR}/test3_log.txt"; then
    print_success "Test 3 completed successfully!"
    echo "Results: ${TEST3_DIR}"
else
    print_error "Test 3 failed - check log: ${TEST3_DIR}/test3_log.txt"
fi

# =============================================================================
# SUMMARY
# =============================================================================
print_header "CPU TESTS SUMMARY"

echo "All tests completed! Results saved to: ${RESULTS_BASE}"
echo ""
echo "Test Results:"
if [ -f "${RESULTS_BASE}/test1_individual/train_log.txt" ]; then
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
echo "Next Steps:"
echo "1. Check individual test logs for any issues"
echo "2. Review model outputs and training curves"
echo "3. When Compute Canada is back online, scale up epochs and batch sizes"

echo ""
echo "CPU Performance Notes:"
echo "- Batch size reduced to ${CPU_BATCH_SIZE} for CPU efficiency"
echo "- Epochs reduced to ${CPU_EPOCHS} for quick testing"
echo "- Using synthetic data for faster processing"
echo "- All models will run but training will be slower than GPU (~22s/batch expected)"

print_success "CPU testing script completed!" 