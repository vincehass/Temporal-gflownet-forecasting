#!/bin/bash
# Comprehensive script to run ablation studies with full wandb integration

# Default parameters
DATASETS="synthetic"
QUANTIZATIONS="adaptive,fixed"
K_VALUES="10,20"
ENTROPY_VALUES="0.01"
POLICY_TYPES="uniform,learned"
EPOCHS=50
BATCH_SIZE=64
RESULTS_DIR="results/integrated_ablation"
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
GPU_ID=0
WANDB_MODE="online"
SKIP_ANALYSIS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}           Temporal GFlowNet Ablation Study Runner                    ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --quantizations)
      QUANTIZATIONS="$2"
      shift 2
      ;;
    --k-values)
      K_VALUES="$2"
      shift 2
      ;;
    --entropy-values)
      ENTROPY_VALUES="$2"
      shift 2
      ;;
    --policy-types)
      POLICY_TYPES="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --offline)
      WANDB_MODE="offline"
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --datasets LIST          Comma-separated list of datasets (default: synthetic)"
      echo "  --quantizations LIST     Comma-separated list of quantization types (default: adaptive,fixed)"
      echo "  --k-values LIST          Comma-separated list of k values (default: 10,20)"
      echo "  --entropy-values LIST    Comma-separated list of entropy values (default: 0.01)"
      echo "  --policy-types LIST      Comma-separated list of policy types (default: uniform,learned)"
      echo "  --epochs N               Number of training epochs (default: 50)"
      echo "  --batch-size N           Batch size (default: 64)"
      echo "  --results-dir DIR        Results directory (default: results/integrated_ablation)"
      echo "  --wandb-project PROJECT  W&B project name (default: temporal-gfn-forecasting)"
      echo "  --wandb-entity ENTITY    W&B entity name (default: nadhirvincenthassen)"
      echo "  --gpu ID                 GPU ID to use (default: 0)"
      echo "  --offline                Run W&B in offline mode"
      echo "  --skip-analysis          Skip running analysis scripts after experiments"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Convert comma-separated lists to arrays
IFS=',' read -ra DATASETS_ARRAY <<< "$DATASETS"
IFS=',' read -ra QUANTIZATIONS_ARRAY <<< "$QUANTIZATIONS"
IFS=',' read -ra K_VALUES_ARRAY <<< "$K_VALUES"
IFS=',' read -ra ENTROPY_VALUES_ARRAY <<< "$ENTROPY_VALUES"
IFS=',' read -ra POLICY_TYPES_ARRAY <<< "$POLICY_TYPES"

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "- Datasets: ${DATASETS}"
echo -e "- Quantizations: ${QUANTIZATIONS}"
echo -e "- K values: ${K_VALUES}"
echo -e "- Entropy values: ${ENTROPY_VALUES}"
echo -e "- Policy types: ${POLICY_TYPES}"
echo -e "- Epochs: ${EPOCHS}"
echo -e "- Batch size: ${BATCH_SIZE}"
echo -e "- Results directory: ${RESULTS_DIR}"
echo -e "- W&B project: ${WANDB_PROJECT}"
echo -e "- W&B entity: ${WANDB_ENTITY}"
echo -e "- W&B mode: ${WANDB_MODE}"
echo -e "- GPU ID: ${GPU_ID}"
echo -e "- Skip analysis: ${SKIP_ANALYSIS}"

# Ensure the results directory exists
mkdir -p "${RESULTS_DIR}"

# Set up a log directory for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ablation_study_${TIMESTAMP}.log"

# Function to log messages
log() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Initialize an array to store experiment names
declare -a EXPERIMENT_NAMES

# Keep track of successful and failed experiments
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# Initialize the wandb group - using the current timestamp as group name
WANDB_GROUP="ablation_${TIMESTAMP}"

# Start time for the experiment
START_TIME=$(date +%s)

for DATASET in "${DATASETS_ARRAY[@]}"; do
  log "========================================================================"
  log "Processing dataset: ${DATASET}"
  log "========================================================================"
  
  # Create dataset-specific directory
  DATASET_RESULTS_DIR="${RESULTS_DIR}/${DATASET}"
  mkdir -p "${DATASET_RESULTS_DIR}"
  
  # Run experiments for each combination of parameters
  for QUANT in "${QUANTIZATIONS_ARRAY[@]}"; do
    for K in "${K_VALUES_ARRAY[@]}"; do
      for ENTROPY in "${ENTROPY_VALUES_ARRAY[@]}"; do
        for POLICY in "${POLICY_TYPES_ARRAY[@]}"; do
          # Skip invalid combinations (e.g., learned policy with fixed quantization)
          if [[ "$POLICY" == "learned" && "$QUANT" == "fixed" ]]; then
            log "${YELLOW}Skipping invalid combination: ${QUANT} quantization with ${POLICY} policy${NC}"
            continue
          fi
          
          # Create experiment name
          EXP_NAME="${DATASET}_${QUANT}_k${K}_${POLICY}_entropy${ENTROPY}"
          EXPERIMENT_NAMES+=("$EXP_NAME")
          
          log "----------------------------------------------------------------------"
          log "${GREEN}Running experiment: ${EXP_NAME}${NC}"
          log "----------------------------------------------------------------------"
          
          # Create experiment-specific directory
          EXP_DIR="${DATASET_RESULTS_DIR}/${EXP_NAME}"
          mkdir -p "${EXP_DIR}"
          
          # Set up wandb parameters
          WANDB_ARGS=""
          if [[ "$WANDB_MODE" == "offline" ]]; then
            WANDB_ARGS="--offline"
          fi
          
          # Build the command - removing the --wandb-group parameter
          CMD="./scripts/run_experiments.sh \
            --name ${EXP_NAME} \
            --dataset ${DATASET} \
            --quantization ${QUANT} \
            --k ${K} \
            --policy ${POLICY} \
            --entropy ${ENTROPY} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --results-dir ${EXP_DIR} \
            --gpu ${GPU_ID} \
            --use-wandb \
            --wandb-project ${WANDB_PROJECT} \
            --wandb-entity ${WANDB_ENTITY} \
            ${WANDB_ARGS}"
          
          log "Running command: ${CMD}"
          
          # Execute the command
          if $CMD >> "${LOG_FILE}" 2>&1; then
            log "${GREEN}✓ Experiment ${EXP_NAME} completed successfully${NC}"
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
          else
            log "${RED}✗ Experiment ${EXP_NAME} failed${NC}"
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
          fi
        done
      done
    done
  done
  
  # Run analysis for this dataset if not skipped
  if [ "$SKIP_ANALYSIS" = false ]; then
    log "========================================================================"
    log "${BLUE}Running analysis for dataset: ${DATASET}${NC}"
    log "========================================================================"
    
    ANALYSIS_DIR="${RESULTS_DIR}/analysis/${DATASET}"
    mkdir -p "${ANALYSIS_DIR}"
    
    # Build analysis command
    ANALYSIS_CMD="python scripts/plot_ablation_results.py \
        --results_dir ${DATASET_RESULTS_DIR} \
        --output_dir ${ANALYSIS_DIR} \
        --study_type all \
        --datasets ${DATASET} \
        --metrics wql crps mase \
        --plot_training_curves"
    
    log "Running analysis command: ${ANALYSIS_CMD}"
    
    # Execute the analysis command
    if $ANALYSIS_CMD >> "${LOG_FILE}" 2>&1; then
      log "${GREEN}✓ Analysis for dataset ${DATASET} completed successfully${NC}"
    else
      log "${RED}✗ Analysis for dataset ${DATASET} failed${NC}"
    fi
  fi
done

# Calculate end time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# Print summary
log "========================================================================"
log "${BLUE}Experiment Summary:${NC}"
log "========================================================================"
log "Total experiments: $((SUCCESSFUL_EXPERIMENTS + FAILED_EXPERIMENTS))"
log "Successful experiments: ${SUCCESSFUL_EXPERIMENTS}"
log "Failed experiments: ${FAILED_EXPERIMENTS}"
log "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log ""
log "Experiment Results:"
for EXP_NAME in "${EXPERIMENT_NAMES[@]}"; do
  if [ -d "${RESULTS_DIR}/${EXP_NAME}/evaluation" ]; then
    log "${EXP_NAME}: ${GREEN}Success${NC}"
  else
    log "${EXP_NAME}: ${RED}Failed${NC}"
  fi
done
log ""
log "All experiments and analysis completed!"
log "Results saved to: ${RESULTS_DIR}"
log "All results logged to W&B project: ${WANDB_PROJECT}"

# Save summary to a file
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
{
  echo "========================================================================"
  echo "Experiment Summary"
  echo "========================================================================"
  echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Total experiments: $((SUCCESSFUL_EXPERIMENTS + FAILED_EXPERIMENTS))"
  echo "Successful experiments: ${SUCCESSFUL_EXPERIMENTS}"
  echo "Failed experiments: ${FAILED_EXPERIMENTS}"
  echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
  echo ""
  echo "Configuration:"
  echo "- Datasets: ${DATASETS}"
  echo "- Quantizations: ${QUANTIZATIONS}"
  echo "- K values: ${K_VALUES}"
  echo "- Entropy values: ${ENTROPY_VALUES}"
  echo "- Policy types: ${POLICY_TYPES}"
  echo "- Epochs: ${EPOCHS}"
  echo "- Batch size: ${BATCH_SIZE}"
  echo ""
  echo "Experiment Results:"
  for EXP_NAME in "${EXPERIMENT_NAMES[@]}"; do
    if [ -d "${RESULTS_DIR}/${EXP_NAME}/evaluation" ]; then
      echo "${EXP_NAME}: Success"
    else
      echo "${EXP_NAME}: Failed"
    fi
  done
  echo ""
  echo "W&B Information:"
  echo "- Project: ${WANDB_PROJECT}"
  echo "- Entity: ${WANDB_ENTITY}"
  echo "- Group: ${WANDB_GROUP}"
  echo "- Mode: ${WANDB_MODE}"
} > "${SUMMARY_FILE}"

echo -e "${GREEN}Summary saved to: ${SUMMARY_FILE}${NC}" 