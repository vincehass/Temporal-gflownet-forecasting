#!/bin/bash
# Script to analyze ablation studies and generate visualizations

# Default parameters
RESULTS_DIR="results/wandb_ablations"
OUTPUT_DIR="results/ablation_plots"
DATASETS="synthetic"
METRICS="wql crps mase"
PLOT_TRAINING=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                Temporal GFlowNet Ablation Analysis                   ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --no-training-curves)
      PLOT_TRAINING=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --results-dir DIR      Directory containing ablation studies (default: results/wandb_ablations)"
      echo "  --output-dir DIR       Directory to save plots (default: results/ablation_plots)"
      echo "  --datasets LIST        Space-separated list of datasets (default: synthetic)"
      echo "  --metrics LIST         Space-separated list of metrics (default: wql crps mase)"
      echo "  --no-training-curves   Skip plotting training curves"
      echo "  --help                 Show this help message"
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
echo -e "- Results directory: ${RESULTS_DIR}"
echo -e "- Output directory: ${OUTPUT_DIR}"
echo -e "- Datasets: ${DATASETS}"
echo -e "- Metrics: ${METRICS}"
echo -e "- Plot training curves: ${PLOT_TRAINING}"

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Set up a log file
LOG_FILE="${OUTPUT_DIR}/analysis_log.txt"
echo "Ablation Analysis Log - $(date)" > "${LOG_FILE}"
echo "------------------------------------------------------------------------------" >> "${LOG_FILE}"

# Check if the visualize_wandb_ablations.py script exists
if [ ! -f "scripts/visualize_wandb_ablations.py" ]; then
  echo -e "${RED}Error: scripts/visualize_wandb_ablations.py not found${NC}"
  echo "Error: scripts/visualize_wandb_ablations.py not found" >> "${LOG_FILE}"
  exit 1
fi

# Run the wandb ablation visualization script
echo -e "${BLUE}Running wandb ablation visualization...${NC}"
echo "Running wandb ablation visualization..." >> "${LOG_FILE}"

TRAIN_CURVES_ARG=""
if [ "$PLOT_TRAINING" = true ]; then
  TRAIN_CURVES_ARG="--generate_configs_plot"
fi

CMD="python scripts/visualize_wandb_ablations.py \
  --results_dir ${RESULTS_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --datasets ${DATASETS} \
  --metrics ${METRICS} \
  ${TRAIN_CURVES_ARG}"

echo "Command: ${CMD}" >> "${LOG_FILE}"
eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}Wandb ablation visualization completed successfully${NC}"
  echo "Wandb ablation visualization completed successfully" >> "${LOG_FILE}"
else
  echo -e "${RED}Wandb ablation visualization failed${NC}"
  echo "Wandb ablation visualization failed" >> "${LOG_FILE}"
fi

# Try to run the existing plot_ablation_results.py script if it exists
if [ -f "scripts/plot_ablation_results.py" ]; then
  echo -e "${BLUE}Running existing ablation results plotting script...${NC}"
  echo "Running existing ablation results plotting script..." >> "${LOG_FILE}"

  TRAIN_CURVES_ARG=""
  if [ "$PLOT_TRAINING" = true ]; then
    TRAIN_CURVES_ARG="--plot_training_curves"
  fi

  CMD="python scripts/plot_ablation_results.py \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${OUTPUT_DIR}/legacy_plots \
    --study_type all \
    --datasets ${DATASETS} \
    --metrics ${METRICS} \
    ${TRAIN_CURVES_ARG}"

  echo "Command: ${CMD}" >> "${LOG_FILE}"
  eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Legacy ablation results plotting completed successfully${NC}"
    echo "Legacy ablation results plotting completed successfully" >> "${LOG_FILE}"
  else
    echo -e "${YELLOW}Legacy ablation results plotting may have had issues - check the log file${NC}"
    echo "Legacy ablation results plotting may have had issues" >> "${LOG_FILE}"
  fi
fi

# Create a README for the output directory
echo -e "${BLUE}Creating README in output directory...${NC}"
cat > "${OUTPUT_DIR}/README.md" << EOF
# Temporal GFlowNet Ablation Study Results

This directory contains visualizations of ablation studies for the Temporal GFlowNet forecasting model.

## Overview

The ablation studies compare different configurations of:
- Quantization strategies (adaptive vs fixed, different k values)
- Policy types (learned vs uniform)
- Other model parameters

## Analysis Details

- Analysis run on: $(date)
- Results directory: ${RESULTS_DIR}
- Datasets analyzed: ${DATASETS}
- Metrics compared: ${METRICS}

## Files

- \`experiment_summary.csv\`: Summary of all experiment configurations
- \`experiment_summary.md\`: Markdown version of the summary table
- \`configuration_heatmap.png\`: Heatmap visualization of numeric configuration parameters
- \`config_*.png\`: Individual configuration parameter comparisons
- \`*_metrics_comparison.png\`: (If available) Comparison of evaluation metrics
- \`*_training_curve.png\`: (If available) Training curves for different experiments
EOF

echo -e "${GREEN}Analysis completed! Results saved to ${OUTPUT_DIR}${NC}"
echo "Analysis completed! Results saved to ${OUTPUT_DIR}" >> "${LOG_FILE}" 