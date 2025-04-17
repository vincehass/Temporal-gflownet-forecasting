# Temporal Generative Flow Networks for Probabilistic Time Series Forecasting

This repository contains the implementation and experimental setup for the paper "Temporal Generative Flow Networks for Probabilistic Time Series Forecasting".

## Features

- GFN framework applied to time series forecasting (using Trajectory Balance Loss).
- Transformer-based forward (and optionally backward) policies.
- Adaptive curriculum-based quantization mechanism for the action space.
- Straight-Through Estimator (STE) for handling discrete actions during backpropagation.
- Uniform and Learned backward policy options.
- Data loading and preprocessing utilities adapted from Chronos (scaling, windowing).
- Standard time series evaluation metrics (WQL, MASE) adapted from Chronos.
- Configuration system for managing experiments and hyperparameters.
- Scripts for training, evaluation, and result plotting.

## Setup

### Quick Setup (Recommended)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vincehass/temporal-gflownet-forecasting.git
   cd temporal-gflownet-forecasting
   ```

2. **Run the setup script:**

   ```bash
   ./setup.sh
   ```

   This script will:

   - Create a Python virtual environment
   - Install all dependencies
   - Make scripts executable
   - Run functionality tests to verify the installation

   If you want to disable Weights & Biases logging during setup tests:

   ```bash
   ./setup.sh --disable-wandb
   ```

3. **Important: Always activate the virtual environment before running any scripts:**

   ```bash
   source venv/bin/activate  # For bash/zsh
   # OR
   conda activate temporal_gfn  # If using conda
   ```

   The prompt should change to indicate the environment is active.

### Manual Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vincehass/temporal-gflownet-forecasting.git
   cd temporal-gflownet-forecasting
   ```

2. **Create Environment:**

   **Using conda:**

   ```bash
   conda env create -f environment.yml
   conda activate temporal_gfn
   ```

   **Using pip/venv:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Make scripts executable:**

   ```bash
   chmod +x scripts/*.py
   chmod +x scripts/*.sh
   ```

4. **Run functionality tests:**
   ```bash
   python scripts/test_functionality.py
   ```

### Datasets

Datasets used in the Chronos paper for pretraining and evaluation (both in-domain and zero-shot) are available through the HuggingFace repos: autogluon/chronos_datasets and autogluon/chronos_datasets_extra. Check out these repos for instructions on how to download and use the datasets.

### Configuration

Experiments are managed via YAML configuration files located in the `configs/` directory.

- `configs/base_config.yaml`: Contains default settings for the model, GFN, training, etc.
- `configs/dataset/*.yaml`: Contains dataset-specific parameters (path, sequence lengths, evaluation horizon).
- `configs/experiment/*.yaml`: Contains overrides for specific experimental setups or ablations (e.g., fixed vs. adaptive quantization, learned vs. uniform backward policy).

You typically run experiments by specifying a base config and potentially dataset/experiment override configs. Key parameters include:

- `model`: Transformer hyperparameters (layers, heads, dim).
- `gfn`: GFN parameters (loss type (TB), Z learning rate, entropy bonus `lambda_entropy`).
- `quantization`: Initial `K`, `K_max`, adaptive params (`lambda_adapt`, `epsilon_adapt`, `delta_adapt`), `vmin`, `vmax`. Set `adaptive: false` for fixed K ablation.
- `policy`: `backward_policy_type` (`uniform` or `learned`).
- `training`: Batch size, learning rate, epochs, warmup epochs, gradient clipping.
- `dataset`: Name, path, context length `T`, prediction horizon `T_prime`.
- `logging`: Directories for saving checkpoints, logs, results.

## Running Scripts

**⚠️ Important: Always activate the virtual environment before running any scripts:**

```bash
# If using venv:
source venv/bin/activate

# If using conda:
conda activate temporal_gfn
```

### Training

Use the `scripts/train.py` script. Configuration is typically passed via command-line arguments using Hydra or a similar library.

```bash
# Activate environment first!
source venv/bin/activate

# Then run the training script
python scripts/train.py --config-name base_config \
    dataset=eeg_config \
    experiment=adaptive_quant_config \
    training.epochs=100 \
    gfn.lambda_entropy=0.01 \
    +results_dir=results/eeg_adaptive_run1
```

Training will save model checkpoints and logs (e.g., TensorBoard logs) to the specified `results_dir`.

### Evaluation

Use the `scripts/evaluate.py` script to evaluate a trained model on the test set.

```bash
# Activate environment first!
source venv/bin/activate

# Then run evaluation
python scripts/evaluate.py \
    --checkpoint_path results/eeg_adaptive_run1/checkpoints/best_model.pt \
    --config_path results/eeg_adaptive_run1/config.yaml \
    --output_dir results/eeg_adaptive_run1/evaluation/
```

Evaluation will calculate metrics (WQL, MASE) and potentially save generated forecasts and plots to the `output_dir`.

### Comprehensive Ablation Studies

This project includes a comprehensive suite of ablation studies to evaluate various aspects of the Temporal GFN approach to probabilistic time series forecasting. These studies are designed to analyze the impact of different components and hyperparameters on model performance.

#### Ablation Study Components

Our ablation studies systematically evaluate three key components:

1. **Quantization Methods (Fixed vs. Adaptive)**:

   - **Fixed Quantization**: Uses a constant number of bins (K) throughout training
   - **Adaptive Quantization**: Dynamically adjusts K based on reward statistics

   _Mathematical formulation_: For adaptive quantization, K is adjusted according to:

   ```
   K_{t+1} = min(K_max, K_t + ⌈λ_adapt * Δ_t⌉)
   ```

   where Δ_t is determined by comparing the reward variance with a threshold:

   ```
   Δ_t = {
     +1 if σ²(r_t) > ε_adapt * mean(r_t)
     -1 if σ²(r_t) < δ_adapt * mean(r_t)
      0 otherwise
   }
   ```

2. **Backward Policy (Uniform vs. Learned)**:

   - **Uniform Policy**: Uses a uniform distribution p*B(s_t | s*{t+1}) = 1/K
   - **Learned Policy**: Trains a neural network to model p*B(s_t | s*{t+1})

   _Mathematical formulation_: The GFlowNet training optimizes the trajectory balance objective:

   ```
   L_TB = Σ_τ (PF(τ)R(τ) - PF(s0→s')PB(s'→sT))²
   ```

   where PF is the forward policy, PB is the backward policy, and R(τ) is the reward.

3. **Entropy Regularization (0.0, 0.001, 0.01, 0.1)**:

   - Controls the exploration-exploitation trade-off
   - Higher values encourage more exploration of the state space

   _Mathematical formulation_: The entropy-regularized objective is:

   ```
   L = L_TB - λ_entropy * H(PF)
   ```

   where H(PF) is the entropy of the forward policy.

#### Combinations and Experimental Setup

The full ablation study systematically evaluates 28 distinct combinations:

- 2 quantization methods (fixed, adaptive)
- 2 backward policy types (uniform, learned)
- 4 entropy regularization values (0.0, 0.001, 0.01, 0.1)
- 3 quantization bin counts (K=5, K=10, K=20)

Note: The learned backward policy is only applicable with adaptive quantization, as fixed quantization with a learned backward policy is not a valid combination.

#### Interpreting Ablation Results

When analyzing ablation results, consider these key insights:

1. **Adaptive Quantization vs. Fixed**:

   - Adaptive quantization typically improves performance by allocating bins where they're most needed
   - The refinement of the discretization adapts to the complexity of the underlying distribution
   - Expected to show greater improvements on datasets with complex, multimodal distributions

2. **Learned vs. Uniform Backward Policy**:

   - A learned backward policy can potentially capture more complex dependencies in the data
   - Especially important for non-Markovian series with long-range dependencies
   - Trades increased computational cost for improved modeling capacity

3. **Entropy Regularization Effects**:

   - Low values (0.0, 0.001): Focus on exploitation of high-reward trajectories
   - Medium values (0.01): Balance between exploration and exploitation
   - High values (0.1): Emphasize exploration, potentially at the cost of immediate reward

4. **Quantization Bin Count (K)**:
   - Lower K (5): Coarser discretization, faster training, but potentially less expressive
   - Medium K (10): Balanced approach for most datasets
   - Higher K (20): Finer discretization, potentially more expressive but slower training and risk of overfitting

#### Running Ablation Studies

To run the comprehensive ablation studies with proper W&B integration, use:

```bash
# Activate environment first!
source venv/bin/activate

# Run the direct ablation script for comprehensive evaluation
./scripts/direct_ablation_all.sh
```

This script will:

1. Run 3 representative experiments first:
   - Adaptive quantization with k=10 and uniform policy
   - Fixed quantization with k=10 and uniform policy
   - Adaptive quantization with k=10 and learned policy
2. Ask if you want to continue with the full ablation study (28 experiments)
3. If confirmed, run all valid combinations of parameters

All experiments are automatically logged to Weights & Biases for easy comparison and visualization.

#### For Individual Ablation Experiments

To run a single experiment with specific parameters:

```bash
python scripts/direct_ablation.py \
    --name "eeg_adaptive_k10_uniform" \
    --dataset eeg \
    --quantization adaptive \
    --k 10 \
    --policy uniform \
    --entropy 0.01 \
    --epochs 100
```

This approach provides the most reliable W&B integration by:

- Directly initializing W&B at the Python level
- Running training with proper parameter passing
- Parsing output in real-time to extract metrics
- Automatically running evaluation after training
- Logging all metrics (training and evaluation) to W&B

### Visualizing Ablation Results

After running the ablation studies, you can visualize the results using the provided plotting scripts:

```bash
# Plot results from ablation studies
python scripts/plot_ablation_results.py \
    --results_dir results/eeg_direct_ablation \
    --output_dir results/ablation_plots \
    --study_type all
```

The script generates several types of comparison plots:

- Overall metrics comparison across all configurations
- Per-horizon metrics showing performance at different prediction steps
- Quantile metrics for probabilistic forecast evaluation
- Training curves showing convergence behavior

For more advanced visualization and interactive exploration, use the W&B dashboard:

```bash
# Open the W&B dashboard in your browser
wandb ui
```

### Running Individual Components

If you need to run just specific parts of the pipeline:

#### Training Only

```bash
python scripts/train.py --config-name base_config dataset=eeg_config quantization=adaptive_config
```

#### Evaluation Only

```bash
./scripts/run_experiments.sh --name "my_experiment" --eval-only --checkpoint path/to/checkpoint.pt
```

#### Visualization Only

```bash
python scripts/plot_results.py --exp_dirs results/experiment1/evaluation results/experiment2/evaluation --output_dir results/comparison_plots
```

## Experiment Tracking and Visualization

The project uses Weights & Biases (W&B) for experiment tracking, visualization, and comparison. All metrics, hyperparameters, and models are automatically logged to W&B.

### W&B Integration

**NOTE: W&B logging is now enabled by default** for all experiments in this project through the base configuration. You don't need to explicitly specify `--use-wandb` in most cases.

#### Recommended Approach: Direct W&B Integration

For the most reliable W&B integration, we strongly recommend using our direct integration scripts:

```bash
# For a single experiment with reliable W&B integration
python scripts/direct_ablation.py --name "my_experiment" --dataset eeg --epochs 50

# For comprehensive ablation studies with reliable W&B integration
./scripts/direct_ablation_all.sh
```

These scripts initialize W&B at the Python level rather than through environment variables, ensuring proper logging of all metrics. See the [Direct Ablation Scripts](#direct-ablation-scripts-with-improved-wandb-integration) section for more details.

#### Standard W&B Integration

To use the standard W&B integration:

1. **Sign up for a W&B account** at [wandb.ai](https://wandb.ai/) if you don't have one.

2. **Login to W&B:**

   ```bash
   wandb login
   ```

3. **Run experiments:**
   W&B logging is automatically enabled for all scripts. Simply run your experiments as usual:

   ```bash
   # Activate environment first!
   source venv/bin/activate

   # W&B logging is enabled by default
   python scripts/train.py --config-name base_config
   ```

4. **Using the W&B wrapper:**
   You can also use the provided wrapper script to ensure W&B is enabled for any command:

   ```bash
   # Run any command with W&B enabled
   ./scripts/run_with_wandb.sh python scripts/train.py --config-name base_config

   # Set custom W&B project name
   ./scripts/run_with_wandb.sh --wandb-project=my-project python scripts/train.py

   # Run shell scripts with W&B enabled
   ./scripts/run_with_wandb.sh ./scripts/run_ablation.sh --dataset eeg --epochs 5
   ```

5. **View your results** at your W&B dashboard.

You can customize the W&B entity and project name:

```bash
# Using the raw script
python scripts/train.py --config-name base_config wandb_entity="your-username" wandb_project="your-project"

# Or using the wrapper
./scripts/run_with_wandb.sh --wandb-entity=your-username --wandb-project=your-project python scripts/train.py
```

By default, W&B logging uses entity "nadhirvincenthassen" and project "temporal-gfn-forecasting".

### Setting Default W&B Configuration

You can set your default W&B settings to avoid specifying them for each command:

```bash
# Set default entity and project
./scripts/run_with_wandb.sh --wandb-entity=your-username --wandb-project=your-custom-project --set-defaults
```

### Offline Mode for Limited Connectivity

For training in environments with limited or no internet connectivity, you can use W&B in offline mode:

```bash
# Using the wrapper script
./scripts/run_with_wandb.sh --offline python scripts/train.py --config-name base_config

# Or directly in the command
python scripts/train.py --config-name base_config wandb_mode=offline
```

This will save all the logs locally. You can sync them to W&B servers later when you have internet access:

```bash
# Manually sync offline runs
wandb sync path/to/logs/wandb
```

### Memory Optimization

The project includes optimized W&B settings to minimize memory usage and improve efficiency:

- **Gradient Logging Control**: By default, gradients are not logged to save memory. Enable with `--log_gradients` if needed.
- **Selective Artifact Saving**: Model checkpoints are saved locally but not automatically uploaded to W&B.
- **Optimized Logging Frequency**: Parameter tracking frequency is configurable with `--log_freq`.

To further reduce memory usage during training:

```bash
python scripts/train.py --config-name base_config wandb_log_freq=100 wandb_save_code=false
```

### Common W&B Integration Issues and Solutions

When working with W&B in this project, you might encounter these common issues:

1. **Failed experiments with shell scripts**: The legacy approach using shell scripts (`run_experiments.sh`, `run_ablation.sh`) may fail to properly initialize W&B, resulting in experiments showing as "failed" in the summary.

   - **Solution**: Use the direct ablation scripts (`direct_ablation.py`, `direct_ablation_all.sh`) which initialize W&B properly at the Python level.

2. **Missing metrics in W&B dashboard**: If metrics aren't appearing in your W&B dashboard:

   - **Solution**: Use `direct_ablation.py` which parses the output in real-time and explicitly logs metrics to W&B.

3. **No evaluation metrics in W&B**: Sometimes only training metrics appear in W&B:

   - **Solution**: The `direct_ablation.py` script automatically runs evaluation after training completes and logs evaluation metrics with the prefix "eval\_".

4. **Hydra configuration errors**: You might see errors like "Key 'context_length' is not in struct":

   - **Solution**: The direct scripts properly handle the Hydra configuration format, avoiding these parameter passing issues.

5. **W&B entity/project issues**: If runs are appearing in the wrong project:
   - **Solution**: Explicitly specify the W&B entity and project using the `--wandb-entity` and `--wandb-project` arguments in the direct scripts.

The direct integration approach (`direct_ablation.py` and `direct_ablation_all.sh`) was specifically developed to address these issues and provide a more reliable way to run experiments with proper W&B integration.

### Testing W&B Integration

To verify that W&B integration is working correctly:

```bash
# Activate environment first!
source venv/bin/activate

# Test online mode
python scripts/test_wandb.py

# Test offline mode
python scripts/test_wandb.py --offline
```

This will create a sample W&B run with synthetic metrics and plots to demonstrate the integration.

## Workflow and Implementation Notes

1.  **Core GFN Logic (`src/temporal_gfn/gfn/`):**

    - Adapt the TB loss implementation from `gflownet-tlm`.
    - Define the `GFNEnvironment` (`env.py`) specifically for time series:
      - `state`: Represents the time window (e.g., a tensor).
      - `action`: Represents the discrete bin index (integer `0` to `K-1`).
      - `step(state, action)`: Implements the state transition (sliding window).
      - `get_actions(state)`: Returns the set of possible actions (all `K` bins).
      - `reward(state)`: Calculates reward based on the final generated trajectory (requires access to ground truth).
    - Implement forward/backward policy wrappers (`policies.py`) that take the neural network models (`src/temporal_gfn/models/`) and handle STE.
    - Implement trajectory sampling (`sampling.py`).

2.  **Models (`src/temporal_gfn/models/`):**

    - Implement a standard Transformer Encoder architecture. The output layer size needs to be adjustable based on the current `K`.
    - If using a learned backward policy, implement its architecture (could be another Transformer or MLP).

3.  **Data Handling (`src/temporal_gfn/data/`):**

    - Leverage Chronos's approach for handling diverse time series datasets.
    - Implement robust windowing (`windowing.py`) to create `(context_window, future_target)` pairs.
    - Implement scaling (`scaling.py`), applying it per time series (like Chronos's mean scaling). Store scaling factors for inverse transform during evaluation/reward calculation.
    - The dataloader should yield batches suitable for the GFN (initial states `s0` and corresponding targets `y`).

4.  **Quantization (`src/temporal_gfn/quantization/`):**

    - `base.py`: Implement `quantize(value, vmin, vmax, K)` and `dequantize(bin_index, vmin, vmax, K)`.
    - `adaptive.py`: Implement the logic to calculate `eta_e` based on logged reward/entropy statistics and update `K`. This module will interact with the trainer and the policy models (to signal layer resizing).

5.  **STE (`src/temporal_gfn/estimators.py`):**

    - Implement the STE logic, likely as a custom PyTorch function or integrated directly into the forward policy's forward pass where hard/soft samples are generated.

6.  **Trainer (`src/temporal_gfn/trainers.py`):**

    - Orchestrates the main training loop.
    - Handles sampling trajectories using the GFN components.
    - Calculates rewards (needs dequantized values and ground truth).
    - Computes the TB loss (+ entropy).
    - Performs backpropagation and optimizer steps.
    - Manages the adaptive quantization updates (calling logic from `quantization/adaptive.py`).
    - Handles logging (metrics, hyperparameters, `K` value) and checkpointing.

7.  **Evaluator (`src/temporal_gfn/evaluators.py`):**
    - Loads a trained model checkpoint.
    - Generates multiple forecast trajectories for the test set (sampling from the learned policy).
    - Dequantizes the generated bin indices and applies inverse scaling.
    - Calculates probabilistic metrics (e.g., WQL over multiple trajectories) and point metrics (e.g., MASE on the median forecast) using utility functions adapted from Chronos (`utils.py`).

## Code Structure (`src/temporal_gfn/`)

- `data/`: Handles all time series input, preprocessing, and batching.
- `models/`: Defines the neural network architectures (policy networks).
- `gfn/`: Encapsulates the core GFN logic (environment, loss, sampling).
- `quantization/`: Specific code for adaptive quantization.
- `estimators.py`: STE implementation.
- `trainers.py`/`evaluators.py`: High-level training/evaluation orchestration.
- `utils.py`: Common utilities, logging, config loading, metrics.

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- NumPy
- Pandas (for data handling)
- Scikit-learn (potentially for metrics or baseline scaling)
- Hydra-core (or another config library like OmegaConf/YAML)
- TensorBoard (for logging)
- Matplotlib / Seaborn (for plotting)
- `tqdm`

(See `requirements.txt` / `environment.yml` for specific versions)

## Contributing

Please refer to CONTRIBUTING.md (if applicable). Open issues for bugs or feature requests. Pull requests are welcome.
