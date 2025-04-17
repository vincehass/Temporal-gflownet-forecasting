#!/bin/bash
# =============================================================================
# Temporal GFN Experiments Runner
# =============================================================================
# This script provides a modular way to run experiments with the 
# Temporal GFN model for time series forecasting, supporting:
#   - Various datasets
#   - Model configurations (transformer architecture)
#   - Quantization strategies (fixed vs adaptive)
#   - Training settings (TB loss, entropy bonus)
#   - Backward policy options (uniform vs learned)
#   - Evaluation metrics and visualization
#
# Author: Nadhir Vincent Hassen

set -e  # Exit on error

# =============================================================================
# Configuration Variables (can be overridden with command line arguments)
# =============================================================================
EXPERIMENT_NAME="default_experiment"
DATASET="eeg"  # Options: eeg, ehr, ecg, etc.
MODEL_TYPE="transformer"  # Only transformer for now, can be extended
QUANTIZATION_TYPE="adaptive"  # Options: adaptive, fixed
QUANTIZATION_K=10  # Initial bins for quantization
POLICY_TYPE="uniform"  # Options: uniform, learned 
ENTROPY_BONUS=0.01  # Lambda entropy value
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
CONTEXT_LENGTH=96
PREDICTION_HORIZON=24
USE_WANDB=false
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="temporal-gfn"
WANDB_OFFLINE=false
GPU_ID=0  # -1 for CPU, 0+ for specific GPU
EVAL_ONLY=false
CHECKPOINT_PATH=""
ABLATION_MODE=false
ABLATION_TYPE=""
RESULTS_DIR="results"

# =============================================================================
# Helper Functions
# =============================================================================

function print_header() {
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

function print_config() {
    echo "Configuration:"
    echo "- Experiment name: $EXPERIMENT_NAME"
    echo "- Dataset: $DATASET"
    echo "- Model type: $MODEL_TYPE"
    echo "- Quantization: $QUANTIZATION_TYPE (K=$QUANTIZATION_K)"
    echo "- Policy type: $POLICY_TYPE"
    echo "- Entropy bonus: $ENTROPY_BONUS"
    echo "- Epochs: $EPOCHS"
    echo "- Batch size: $BATCH_SIZE"
    echo "- Learning rate: $LEARNING_RATE"
    echo "- Context length: $CONTEXT_LENGTH"
    echo "- Prediction horizon: $PREDICTION_HORIZON"
    echo "- Use W&B: $USE_WANDB"
    
    if [ "$USE_WANDB" = true ]; then
        echo "  - W&B entity: $WANDB_ENTITY"
        echo "  - W&B project: $WANDB_PROJECT"
        echo "  - W&B offline mode: $WANDB_OFFLINE"
    fi
    
    echo "- GPU ID: $GPU_ID"
    echo "- Results directory: $RESULTS_DIR/$EXPERIMENT_NAME"
    
    if [ "$EVAL_ONLY" = true ]; then
        echo "- Evaluation only mode with checkpoint: $CHECKPOINT_PATH"
    fi
    
    if [ "$ABLATION_MODE" = true ]; then
        echo "- Ablation study mode: $ABLATION_TYPE"
    fi
}

function check_environment() {
    # Check if the conda environment exists and is activated
    if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
        echo "ERROR: No active virtual environment detected."
        echo "Please activate the environment first:"
        echo "  source venv/bin/activate  # For venv"
        echo "  conda activate temporal_gfn  # For conda"
        exit 1
    fi
    
    # Verify Python can import key packages
    python -c "import torch; import numpy; import wandb" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Missing required Python packages."
        echo "Please make sure all dependencies are installed."
        exit 1
    fi
    
    echo "✓ Environment check passed"
}

function create_experiment_dir() {
    local exp_dir="$RESULTS_DIR/$EXPERIMENT_NAME"
    mkdir -p "$exp_dir/checkpoints"
    mkdir -p "$exp_dir/plots"
    mkdir -p "$exp_dir/logs"
    mkdir -p "$exp_dir/evaluation"
    
    # Save experiment configuration
    {
        echo "experiment_name: $EXPERIMENT_NAME"
        echo "dataset: $DATASET"
        echo "model_type: $MODEL_TYPE"
        echo "quantization:"
        echo "  type: $QUANTIZATION_TYPE"
        echo "  k_initial: $QUANTIZATION_K"
        echo "policy_type: $POLICY_TYPE"
        echo "entropy_bonus: $ENTROPY_BONUS"
        echo "epochs: $EPOCHS"
        echo "batch_size: $BATCH_SIZE"
        echo "learning_rate: $LEARNING_RATE"
        echo "context_length: $CONTEXT_LENGTH"
        echo "prediction_horizon: $PREDICTION_HORIZON"
        echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    } > "$exp_dir/config.yaml"
    
    echo "✓ Created experiment directory: $exp_dir"
    echo "✓ Saved experiment configuration"
}

function prepare_dataset_args() {
    local dataset_args=""
    
    case $DATASET in
        "eeg")
            dataset_args="dataset=eeg_config context_length=$CONTEXT_LENGTH prediction_horizon=$PREDICTION_HORIZON"
            ;;
        "ehr")
            dataset_args="dataset=ehr_config context_length=$CONTEXT_LENGTH prediction_horizon=$PREDICTION_HORIZON"
            ;;
        "ecg")
            dataset_args="dataset=ecg_config context_length=$CONTEXT_LENGTH prediction_horizon=$PREDICTION_HORIZON"
            ;;
        "synthetic")
            dataset_args="dataset=synthetic_config context_length=$CONTEXT_LENGTH prediction_horizon=$PREDICTION_HORIZON"
            ;;
        *)
            echo "WARNING: Unknown dataset '$DATASET'. Using default configuration."
            dataset_args="dataset=default_config context_length=$CONTEXT_LENGTH prediction_horizon=$PREDICTION_HORIZON"
            ;;
    esac
    
    echo "$dataset_args"
}

function prepare_model_args() {
    local model_args=""
    
    case $MODEL_TYPE in
        "transformer")
            model_args="model=transformer_config"
            ;;
        *)
            echo "WARNING: Unknown model type '$MODEL_TYPE'. Using default transformer configuration."
            model_args="model=transformer_config"
            ;;
    esac
    
    echo "$model_args"
}

function prepare_quant_args() {
    local quant_args=""
    
    case $QUANTIZATION_TYPE in
        "adaptive")
            quant_args="quantization=adaptive_config quantization.k_initial=$QUANTIZATION_K quantization.adaptive=true"
            ;;
        "fixed")
            quant_args="quantization=fixed_config quantization.k_initial=$QUANTIZATION_K quantization.adaptive=false"
            ;;
        *)
            echo "WARNING: Unknown quantization type '$QUANTIZATION_TYPE'. Using default adaptive configuration."
            quant_args="quantization=adaptive_config quantization.k_initial=$QUANTIZATION_K quantization.adaptive=true"
            ;;
    esac
    
    echo "$quant_args"
}

function prepare_policy_args() {
    local policy_args=""
    
    case $POLICY_TYPE in
        "uniform")
            policy_args="policy=uniform_config policy.backward_policy_type=uniform"
            ;;
        "learned")
            policy_args="policy=learned_config policy.backward_policy_type=learned"
            ;;
        *)
            echo "WARNING: Unknown policy type '$POLICY_TYPE'. Using default uniform configuration."
            policy_args="policy=uniform_config policy.backward_policy_type=uniform"
            ;;
    esac
    
    echo "$policy_args"
}

function prepare_training_args() {
    local training_args="training=base_config training.epochs=$EPOCHS training.batch_size=$BATCH_SIZE training.learning_rate=$LEARNING_RATE gfn.lambda_entropy=$ENTROPY_BONUS"
    echo "$training_args"
}

function prepare_wandb_args() {
    local wandb_args=""
    
    if [ "$USE_WANDB" = true ]; then
        wandb_args="use_wandb=true wandb_entity=$WANDB_ENTITY wandb_project=$WANDB_PROJECT"
        
        if [ "$WANDB_OFFLINE" = true ]; then
            wandb_args="$wandb_args wandb_mode=offline"
        fi
    fi
    
    echo "$wandb_args"
}

function prepare_device_args() {
    local device_args=""
    
    if [ "$GPU_ID" -ge 0 ]; then
        device_args="gpu=$GPU_ID"
    else
        device_args="device=cpu"
    fi
    
    echo "$device_args"
}

function run_training() {
    local exp_dir="$RESULTS_DIR/$EXPERIMENT_NAME"
    local dataset_args=$(prepare_dataset_args)
    local model_args=$(prepare_model_args)
    local quant_args=$(prepare_quant_args)
    local policy_args=$(prepare_policy_args)
    local training_args=$(prepare_training_args)
    local wandb_args=$(prepare_wandb_args)
    local device_args=$(prepare_device_args)
    
    # Assemble the full command, ensuring proper format for Hydra
    local cmd="python scripts/train.py \
        --config-name base_config \
        $wandb_args \
        $device_args \
        ${dataset_args} \
        ${model_args} \
        ${quant_args} \
        ${policy_args} \
        ${training_args} \
        results_dir=$exp_dir"
    
    print_header "Starting Training"
    echo "Command: $cmd"
    echo ""
    
    # Execute the command
    eval "$cmd"
}

function run_evaluation() {
    local exp_dir="$RESULTS_DIR/$EXPERIMENT_NAME"
    local device_args=$(prepare_device_args)
    local wandb_args=$(prepare_wandb_args)
    
    local checkpoint=""
    if [ "$EVAL_ONLY" = true ] && [ ! -z "$CHECKPOINT_PATH" ]; then
        checkpoint="$CHECKPOINT_PATH"
    else
        checkpoint="$exp_dir/checkpoints/best_model.pt"
    fi
    
    # Make sure the checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "ERROR: Checkpoint file not found: $checkpoint"
        exit 1
    fi
    
    local config_path="$exp_dir/config.yaml"
    if [ ! -f "$config_path" ] && [ "$EVAL_ONLY" = true ]; then
        config_path=$(dirname "$CHECKPOINT_PATH")/config.yaml
    fi
    
    # Make sure the config exists
    if [ ! -f "$config_path" ]; then
        echo "ERROR: Config file not found: $config_path"
        exit 1
    fi
    
    # Assemble the full command
    local cmd="python scripts/evaluate.py \
        --checkpoint_path $checkpoint \
        --config_path $config_path \
        --output_dir $exp_dir/evaluation \
        $wandb_args \
        $device_args"
    
    print_header "Starting Evaluation"
    echo "Command: $cmd"
    echo ""
    
    # Execute the command
    eval "$cmd"
}

function run_visualization() {
    local exp_dir="$RESULTS_DIR/$EXPERIMENT_NAME"
    local wandb_args=""
    
    # Prepare wandb arguments
    if [ "$USE_WANDB" = true ]; then
        wandb_args="--use-wandb --wandb-entity $WANDB_ENTITY --wandb-project $WANDB_PROJECT --wandb-name ${EXPERIMENT_NAME}_visualization"
        
        if [ "$WANDB_OFFLINE" = true ]; then
            wandb_args="$wandb_args --offline"
        fi
    fi
    
    # Assemble the full command
    local cmd="./scripts/visualize_results.sh \
        --results-dir $RESULTS_DIR \
        --output-dir $exp_dir/plots \
        --experiments $EXPERIMENT_NAME \
        $wandb_args"
    
    print_header "Generating Visualizations"
    echo "Command: $cmd"
    echo ""
    
    # Execute the command
    eval "$cmd"
}

function run_ablation_study() {
    print_header "Running Ablation Study: $ABLATION_TYPE"
    local base_exp_name="$EXPERIMENT_NAME"
    local wandb_args=""
    
    # Prepare wandb arguments
    if [ "$USE_WANDB" = true ]; then
        wandb_args="--use-wandb --wandb-entity $WANDB_ENTITY --wandb-project $WANDB_PROJECT"
        
        if [ "$WANDB_OFFLINE" = true ]; then
            wandb_args="$wandb_args --offline"
        fi
    fi
    
    case $ABLATION_TYPE in
        "quantization")
            # Run with different quantization settings
            for quant_type in "adaptive" "fixed"; do
                for k_value in 5 10 20; do
                    local ablation_name="${base_exp_name}_${quant_type}_k${k_value}"
                    echo "Running experiment: $ablation_name"
                    EXPERIMENT_NAME="$ablation_name" QUANTIZATION_TYPE="$quant_type" QUANTIZATION_K="$k_value" run_single_experiment
                done
            done
            
            # Combine results for comparison
            local exp_list=""
            for quant_type in "adaptive" "fixed"; do
                for k_value in 5 10 20; do
                    local ablation_name="${base_exp_name}_${quant_type}_k${k_value}"
                    exp_list="$exp_list $ablation_name"
                done
            done
            
            ./scripts/visualize_results.sh $wandb_args \
                --results-dir "$RESULTS_DIR" \
                --output-dir "$RESULTS_DIR/${base_exp_name}_ablation_quantization/plots" \
                --experiments $exp_list \
                --wandb-name "${base_exp_name}_ablation_quantization"
            ;;
            
        "policy")
            # Run with different policy types
            for policy in "uniform" "learned"; do
                local ablation_name="${base_exp_name}_policy_${policy}"
                echo "Running experiment: $ablation_name"
                EXPERIMENT_NAME="$ablation_name" POLICY_TYPE="$policy" run_single_experiment
            done
            
            # Combine results for comparison
            ./scripts/visualize_results.sh $wandb_args \
                --results-dir "$RESULTS_DIR" \
                --output-dir "$RESULTS_DIR/${base_exp_name}_ablation_policy/plots" \
                --experiments "${base_exp_name}_policy_uniform" "${base_exp_name}_policy_learned" \
                --wandb-name "${base_exp_name}_ablation_policy"
            ;;
            
        "entropy")
            # Run with different entropy bonus values
            for entropy in 0.0 0.001 0.01 0.1; do
                local ablation_name="${base_exp_name}_entropy_${entropy//./_}"
                echo "Running experiment: $ablation_name"
                EXPERIMENT_NAME="$ablation_name" ENTROPY_BONUS="$entropy" run_single_experiment
            done
            
            # Combine results for comparison
            local exp_list=""
            for entropy in 0.0 0.001 0.01 0.1; do
                local ablation_name="${base_exp_name}_entropy_${entropy//./_}"
                exp_list="$exp_list $ablation_name"
            done
            
            ./scripts/visualize_results.sh $wandb_args \
                --results-dir "$RESULTS_DIR" \
                --output-dir "$RESULTS_DIR/${base_exp_name}_ablation_entropy/plots" \
                --experiments $exp_list \
                --wandb-name "${base_exp_name}_ablation_entropy"
            ;;
            
        *)
            echo "ERROR: Unknown ablation type '$ABLATION_TYPE'"
            echo "Available options: quantization, policy, entropy"
            exit 1
            ;;
    esac
}

function run_single_experiment() {
    # Create experiment directory and save configuration
    create_experiment_dir
    
    # Run training (unless eval_only is true)
    if [ "$EVAL_ONLY" = false ]; then
        run_training
    fi
    
    # Run evaluation
    run_evaluation
    
    # Generate visualizations
    run_visualization
}

function parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --name)
                EXPERIMENT_NAME="$2"
                shift
                shift
                ;;
            --dataset)
                DATASET="$2"
                shift
                shift
                ;;
            --model)
                MODEL_TYPE="$2"
                shift
                shift
                ;;
            --quantization)
                QUANTIZATION_TYPE="$2"
                shift
                shift
                ;;
            --k)
                QUANTIZATION_K="$2"
                shift
                shift
                ;;
            --policy)
                POLICY_TYPE="$2"
                shift
                shift
                ;;
            --entropy)
                ENTROPY_BONUS="$2"
                shift
                shift
                ;;
            --epochs)
                EPOCHS="$2"
                shift
                shift
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift
                shift
                ;;
            --lr)
                LEARNING_RATE="$2"
                shift
                shift
                ;;
            --context-length)
                CONTEXT_LENGTH="$2"
                shift
                shift
                ;;
            --pred-horizon)
                PREDICTION_HORIZON="$2"
                shift
                shift
                ;;
            --use-wandb)
                USE_WANDB=true
                shift
                ;;
            --wandb-entity)
                WANDB_ENTITY="$2"
                shift
                shift
                ;;
            --wandb-project)
                WANDB_PROJECT="$2"
                shift
                shift
                ;;
            --offline)
                WANDB_OFFLINE=true
                shift
                ;;
            --gpu)
                GPU_ID="$2"
                shift
                shift
                ;;
            --cpu)
                GPU_ID="-1"
                shift
                ;;
            --eval-only)
                EVAL_ONLY=true
                shift
                ;;
            --checkpoint)
                CHECKPOINT_PATH="$2"
                shift
                shift
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                shift
                shift
                ;;
            --ablation)
                ABLATION_MODE=true
                ABLATION_TYPE="$2"
                shift
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --name NAME               Experiment name"
                echo "  --dataset DATASET         Dataset to use (eeg, ehr, ecg, synthetic)"
                echo "  --model MODEL             Model type (transformer)"
                echo "  --quantization TYPE       Quantization type (adaptive, fixed)"
                echo "  --k VALUE                 Initial number of quantization bins"
                echo "  --policy TYPE             Backward policy type (uniform, learned)"
                echo "  --entropy VALUE           Entropy bonus lambda value"
                echo "  --epochs N                Number of training epochs"
                echo "  --batch-size N            Batch size"
                echo "  --lr VALUE                Learning rate"
                echo "  --context-length N        Context window length"
                echo "  --pred-horizon N          Prediction horizon length"
                echo "  --use-wandb               Enable W&B logging"
                echo "  --wandb-entity ENTITY     W&B entity name"
                echo "  --wandb-project PROJECT   W&B project name"
                echo "  --offline                 Use W&B in offline mode"
                echo "  --gpu ID                  GPU ID to use (default: 0)"
                echo "  --cpu                     Use CPU instead of GPU"
                echo "  --eval-only               Run evaluation only (no training)"
                echo "  --checkpoint PATH         Path to model checkpoint for evaluation"
                echo "  --results-dir DIR         Results directory (default: results)"
                echo "  --ablation TYPE           Run ablation study (quantization, policy, entropy)"
                echo "  --help                    Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $key"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Main Script Execution
# =============================================================================

# Parse command line arguments
parse_args "$@"

# Check environment
check_environment

# Print configuration
print_config

# Run experiments
if [ "$ABLATION_MODE" = true ]; then
    run_ablation_study
else
    run_single_experiment
fi

print_header "Experiment Completed Successfully"
echo "Results saved to: $RESULTS_DIR/$EXPERIMENT_NAME"
echo "Thank you for using Temporal GFN!" 