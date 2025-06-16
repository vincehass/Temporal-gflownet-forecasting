#!/bin/bash
# Master script to run a complete ablation study including experiments and analysis

# Default parameters
DATASETS="synthetic"
QUANTIZATIONS="adaptive,fixed"
K_VALUES="5,10,20"
POLICY_TYPES="uniform,learned"
ENTROPY_VALUE="0.01"
EPOCHS=50
BATCH_SIZE=64
RESULTS_DIR="results/complete_ablation_study"
OUTPUT_DIR="results/complete_ablation_study/plots"
WANDB_PROJECT="temporal-gfn-forecasting"
WANDB_ENTITY="nadhirvincenthassen"
GPU_ID=0
WANDB_MODE="online"
SKIP_EXPERIMENTS=false
SKIP_ANALYSIS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}         Complete Temporal GFlowNet Ablation Study Runner             ${NC}"
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
    --policy-types)
      POLICY_TYPES="$2"
      shift 2
      ;;
    --entropy-value)
      ENTROPY_VALUE="$2"
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
    --output-dir)
      OUTPUT_DIR="$2"
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
    --skip-experiments)
      SKIP_EXPERIMENTS=true
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --datasets LIST           Comma-separated list of datasets (default: synthetic)"
      echo "  --quantizations LIST      Comma-separated list of quantization types (default: adaptive,fixed)"
      echo "  --k-values LIST           Comma-separated list of k values (default: 5,10,20)"
      echo "  --policy-types LIST       Comma-separated list of policy types (default: uniform,learned)"
      echo "  --entropy-value VALUE     Entropy value (default: 0.01)"
      echo "  --epochs N                Number of training epochs (default: 50)"
      echo "  --batch-size N            Batch size (default: 64)"
      echo "  --results-dir DIR         Results directory (default: results/complete_ablation_study)"
      echo "  --output-dir DIR          Analysis output directory (default: results/complete_ablation_study/plots)"
      echo "  --wandb-project PROJECT   W&B project name (default: temporal-gfn-forecasting)"
      echo "  --wandb-entity ENTITY     W&B entity name (default: nadhirvincenthassen)"
      echo "  --gpu ID                  GPU ID to use (default: 0)"
      echo "  --offline                 Run W&B in offline mode"
      echo "  --skip-experiments        Skip running experiments, only do analysis"
      echo "  --skip-analysis           Skip running analysis after experiments"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "- Datasets: ${DATASETS}"
echo -e "- Quantizations: ${QUANTIZATIONS}"
echo -e "- K values: ${K_VALUES}"
echo -e "- Policy types: ${POLICY_TYPES}"
echo -e "- Entropy value: ${ENTROPY_VALUE}"
echo -e "- Epochs: ${EPOCHS}"
echo -e "- Batch size: ${BATCH_SIZE}"
echo -e "- Results directory: ${RESULTS_DIR}"
echo -e "- Analysis output directory: ${OUTPUT_DIR}"
echo -e "- W&B project: ${WANDB_PROJECT}"
echo -e "- W&B entity: ${WANDB_ENTITY}"
echo -e "- W&B mode: ${WANDB_MODE}"
echo -e "- GPU ID: ${GPU_ID}"
echo -e "- Skip experiments: ${SKIP_EXPERIMENTS}"
echo -e "- Skip analysis: ${SKIP_ANALYSIS}"

# Ensure the results directory exists
mkdir -p "${RESULTS_DIR}"

# Set up a log directory for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/complete_study_${TIMESTAMP}.log"

# Function to log messages
log() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Start time for the entire study
START_TIME=$(date +%s)

# Initialize wandb group for this study
WANDB_GROUP="ablation_${TIMESTAMP}"

# Phase 1: Run experiments
if [ "$SKIP_EXPERIMENTS" = false ]; then
  log "========================================================================"
  log "${BLUE}Phase 1: Running Experiments${NC}"
  log "========================================================================"
  
  # Check if run_integrated_ablation_study.sh exists
  if [ -f "scripts/run_integrated_ablation_study.sh" ]; then
    # Run the integrated ablation study script
    log "Running with integrated ablation study script..."
    
    WANDB_ARGS=""
    if [[ "$WANDB_MODE" == "offline" ]]; then
      WANDB_ARGS="--offline"
    fi
    
    ABLATION_CMD="./scripts/run_integrated_ablation_study.sh \
      --datasets ${DATASETS} \
      --quantizations ${QUANTIZATIONS} \
      --k-values ${K_VALUES} \
      --policy-types ${POLICY_TYPES} \
      --entropy-values ${ENTROPY_VALUE} \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --results-dir ${RESULTS_DIR} \
      --wandb-project ${WANDB_PROJECT} \
      --wandb-entity ${WANDB_ENTITY} \
      --gpu ${GPU_ID} \
      ${WANDB_ARGS}"
    
    log "Running command: ${ABLATION_CMD}"
    
    if $ABLATION_CMD >> "${LOG_FILE}" 2>&1; then
      log "${GREEN}✓ Integrated ablation study completed successfully${NC}"
    else
      log "${RED}✗ Integrated ablation study failed - check log for details${NC}"
      if [ "$SKIP_ANALYSIS" = false ]; then
        log "${YELLOW}Proceeding to analysis phase anyway...${NC}"
      else
        exit 1
      fi
    fi
  else
    log "${YELLOW}Warning: scripts/run_integrated_ablation_study.sh not found${NC}"
    log "Running individual experiments using run_experiments.sh..."
    
    # Convert comma-separated lists to arrays
    IFS=',' read -ra DATASETS_ARRAY <<< "$DATASETS"
    IFS=',' read -ra QUANTIZATIONS_ARRAY <<< "$QUANTIZATIONS"
    IFS=',' read -ra K_VALUES_ARRAY <<< "$K_VALUES"
    IFS=',' read -ra POLICY_TYPES_ARRAY <<< "$POLICY_TYPES"
    
    # Run experiments manually
    for DATASET in "${DATASETS_ARRAY[@]}"; do
      log "Processing dataset: ${DATASET}"
      
      for QUANT in "${QUANTIZATIONS_ARRAY[@]}"; do
        for K in "${K_VALUES_ARRAY[@]}"; do
          for POLICY in "${POLICY_TYPES_ARRAY[@]}"; do
            # Skip invalid combinations (e.g., learned policy with fixed quantization)
            if [[ "$POLICY" == "learned" && "$QUANT" == "fixed" ]]; then
              log "${YELLOW}Skipping invalid combination: ${QUANT} quantization with ${POLICY} policy${NC}"
              continue
            fi
            
            # Create experiment name
            EXP_NAME="${DATASET}_${QUANT}_k${K}_${POLICY}"
            
            log "Running experiment: ${EXP_NAME}"
            
            WANDB_ARGS=""
            if [[ "$WANDB_MODE" == "offline" ]]; then
              WANDB_ARGS="--offline"
            fi
            
            # Check if we can use run_experiments.sh
            if [ -f "scripts/run_experiments.sh" ]; then
              EXP_CMD="./scripts/run_experiments.sh \
                --name ${EXP_NAME} \
                --dataset ${DATASET} \
                --quantization ${QUANT} \
                --k ${K} \
                --policy ${POLICY} \
                --entropy ${ENTROPY_VALUE} \
                --epochs ${EPOCHS} \
                --batch-size ${BATCH_SIZE} \
                --results-dir ${RESULTS_DIR}/${EXP_NAME} \
                --gpu ${GPU_ID} \
                --use-wandb \
                --wandb-project ${WANDB_PROJECT} \
                --wandb-entity ${WANDB_ENTITY} \
                --wandb-group ${WANDB_GROUP} \
                ${WANDB_ARGS}"
              
              log "Running command: ${EXP_CMD}"
              
              if $EXP_CMD >> "${LOG_FILE}" 2>&1; then
                log "${GREEN}✓ Experiment ${EXP_NAME} completed successfully${NC}"
              else
                log "${RED}✗ Experiment ${EXP_NAME} failed${NC}"
              fi
            else
              log "${RED}Error: scripts/run_experiments.sh not found${NC}"
              log "Cannot run experiments manually without run_experiments.sh"
              exit 1
            fi
          done
        done
      done
    done
  fi
else
  log "Skipping experiment phase as requested."
fi

# Phase 2: Run analysis
if [ "$SKIP_ANALYSIS" = false ]; then
  log "========================================================================"
  log "${BLUE}Phase 2: Running Analysis${NC}"
  log "========================================================================"
  
  # Set up output directory if not already done
  mkdir -p "${OUTPUT_DIR}"
  
  # Check if run_ablation_analysis.sh exists
  if [ -f "scripts/run_ablation_analysis.sh" ]; then
    log "Running analysis with run_ablation_analysis.sh..."
    
    # Use space-separated list for datasets for the analysis script
    ANALYSIS_DATASETS=$(echo $DATASETS | tr ',' ' ')
    
    ANALYSIS_CMD="./scripts/run_ablation_analysis.sh \
      --results-dir ${RESULTS_DIR} \
      --output-dir ${OUTPUT_DIR} \
      --datasets \"${ANALYSIS_DATASETS}\" \
      --metrics \"wql crps mase\""
    
    log "Running command: ${ANALYSIS_CMD}"
    
    if $ANALYSIS_CMD >> "${LOG_FILE}" 2>&1; then
      log "${GREEN}✓ Ablation analysis completed successfully${NC}"
    else
      log "${RED}✗ Ablation analysis failed - check log for details${NC}"
    fi
  else
    log "${YELLOW}Warning: scripts/run_ablation_analysis.sh not found${NC}"
    log "Looking for alternative analysis scripts..."
    
    # Try to use plot_ablation_results.py directly if it exists
    if [ -f "scripts/plot_ablation_results.py" ]; then
      log "Running analysis with plot_ablation_results.py..."
      
      # Use space-separated list for datasets
      ANALYSIS_DATASETS=$(echo $DATASETS | tr ',' ' ')
      
      ANALYSIS_CMD="python scripts/plot_ablation_results.py \
        --results_dir ${RESULTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --study_type all \
        --datasets ${ANALYSIS_DATASETS} \
        --metrics wql crps mase \
        --plot_training_curves"
      
      log "Running command: ${ANALYSIS_CMD}"
      
      if $ANALYSIS_CMD >> "${LOG_FILE}" 2>&1; then
        log "${GREEN}✓ Ablation analysis completed successfully${NC}"
      else
        log "${RED}✗ Ablation analysis failed - check log for details${NC}"
      fi
    elif [ -f "scripts/visualize_wandb_ablations.py" ]; then
      log "Running analysis with visualize_wandb_ablations.py..."
      
      # Use space-separated list for datasets
      ANALYSIS_DATASETS=$(echo $DATASETS | tr ',' ' ')
      
      ANALYSIS_CMD="python scripts/visualize_wandb_ablations.py \
        --results_dir ${RESULTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --datasets ${ANALYSIS_DATASETS} \
        --metrics wql crps mase \
        --generate_configs_plot"
      
      log "Running command: ${ANALYSIS_CMD}"
      
      if $ANALYSIS_CMD >> "${LOG_FILE}" 2>&1; then
        log "${GREEN}✓ Ablation analysis completed successfully${NC}"
      else
        log "${RED}✗ Ablation analysis failed - check log for details${NC}"
      fi
    else
      log "${RED}Error: No analysis scripts found${NC}"
      log "Cannot run analysis without any of: run_ablation_analysis.sh, plot_ablation_results.py, or visualize_wandb_ablations.py"
    fi
  fi
  
  # Create a README.md in the results directory
  README_FILE="${RESULTS_DIR}/README.md"
  log "Creating README.md in results directory..."
  
  {
    echo "# Temporal GFlowNet Ablation Study"
    echo ""
    echo "## Study Overview"
    echo ""
    echo "This ablation study was run on $(date) and explores the impact of different model configurations:"
    echo ""
    echo "- **Datasets:** ${DATASETS}"
    echo "- **Quantization Methods:** ${QUANTIZATIONS}"
    echo "- **K Values:** ${K_VALUES}"
    echo "- **Policy Types:** ${POLICY_TYPES}"
    echo "- **Entropy Value:** ${ENTROPY_VALUE}"
    echo "- **Training Parameters:** ${EPOCHS} epochs, batch size ${BATCH_SIZE}"
    echo ""
    echo "## Results"
    echo ""
    echo "The analysis results are available in the [plots directory](./plots)."
    echo ""
    echo "## W&B Information"
    echo ""
    echo "- **Project:** ${WANDB_PROJECT}"
    echo "- **Entity:** ${WANDB_ENTITY}"
    echo "- **Group:** ${WANDB_GROUP}"
    echo ""
    echo "## Run Information"
    echo ""
    echo "- **Run Date:** $(date)"
    echo "- **Log File:** [logs/complete_study_${TIMESTAMP}.log](./logs/complete_study_${TIMESTAMP}.log)"
  } > "${README_FILE}"
  
  log "${GREEN}Created README.md at ${README_FILE}${NC}"
else
  log "Skipping analysis phase as requested."
fi

# Calculate end time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# Print summary
log "========================================================================"
log "${BLUE}Complete Ablation Study Summary:${NC}"
log "========================================================================"
log "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log ""
log "Study details:"
log "- Datasets: ${DATASETS}"
log "- Quantizations: ${QUANTIZATIONS}"
log "- K values: ${K_VALUES}"
log "- Policy types: ${POLICY_TYPES}"
log "- Epochs: ${EPOCHS}"
log "- Results saved to: ${RESULTS_DIR}"
log "- Analysis saved to: ${OUTPUT_DIR}"
log ""

if [ "$SKIP_EXPERIMENTS" = false ]; then
  log "Experiments have been run and logged to W&B project: ${WANDB_PROJECT}"
fi

if [ "$SKIP_ANALYSIS" = false ]; then
  log "Analysis has been performed and results saved."
  log "To view the results, check the plots in: ${OUTPUT_DIR}"
fi

log ""
log "Complete ablation study completed successfully!"
log "========================================================================" 