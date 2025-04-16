# Temporal Generative Flow Networks for Probabilistic Time Series Forecasting

This repository contains the implementation and experimental setup for the paper "Temporal Generative Flow Networks for Probabilistic Time Series Forecasting".


## Features

*   GFN framework applied to time series forecasting (using Trajectory Balance Loss).
*   Transformer-based forward (and optionally backward) policies.
*   Adaptive curriculum-based quantization mechanism for the action space.
*   Straight-Through Estimator (STE) for handling discrete actions during backpropagation.
*   Uniform and Learned backward policy options.
*   Data loading and preprocessing utilities adapted from Chronos (scaling, windowing).
*   Standard time series evaluation metrics (WQL, MASE) adapted from Chronos.
*   Configuration system for managing experiments and hyperparameters.
*   Scripts for training, evaluation, and result plotting.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vincehass/temporal-gflownet-forecasting.git
    cd temporal-gflownet-forecasting
    ```

2.  **Create Environment (Recommended: Conda):**
    ```bash
    conda env create -f environment.yml
    conda activate temporal_gfn
    ```
    *Alternatively, use pip:*
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a compatible version of PyTorch installed (potentially with CUDA support).*

3.  **Download Data:**
    *   Benchmark datasets used in Chronos can be downloaded using the provided script or instructions within it:
        ```bash
        cd data
        bash download_chronos_data.sh # Adapt this script as needed
        cd ..
        ```
    *   For custom datasets (EEG, EHR), place them in the `data/` directory or modify the dataloader paths in the configurations.

## Usage


### Datasets

Datasets used in the Chronos paper for pretraining and evaluation (both in-domain and zero-shot) are available through the HuggingFace repos: autogluon/chronos_datasets and autogluon/chronos_datasets_extra. Check out these repos for instructions on how to download and use the datasets.

### Configuration

Experiments are managed via YAML configuration files located in the `configs/` directory.
*   `configs/base_config.yaml`: Contains default settings for the model, GFN, training, etc.
*   `configs/dataset/*.yaml`: Contains dataset-specific parameters (path, sequence lengths, evaluation horizon).
*   `configs/experiment/*.yaml`: Contains overrides for specific experimental setups or ablations (e.g., fixed vs. adaptive quantization, learned vs. uniform backward policy).

You typically run experiments by specifying a base config and potentially dataset/experiment override configs. Key parameters include:
*   `model`: Transformer hyperparameters (layers, heads, dim).
*   `gfn`: GFN parameters (loss type (TB), Z learning rate, entropy bonus `lambda_entropy`).
*   `quantization`: Initial `K`, `K_max`, adaptive params (`lambda_adapt`, `epsilon_adapt`, `delta_adapt`), `vmin`, `vmax`. Set `adaptive: false` for fixed K ablation.
*   `policy`: `backward_policy_type` (`uniform` or `learned`).
*   `training`: Batch size, learning rate, epochs, warmup epochs, gradient clipping.
*   `dataset`: Name, path, context length `T`, prediction horizon `T_prime`.
*   `logging`: Directories for saving checkpoints, logs, results.

### Training

Use the `scripts/train.py` script. Configuration is typically passed via command-line arguments using Hydra or a similar library (needs integration in `train.py`).

```bash
python scripts/train.py --config-name base_config \
    dataset=eeg_config \
    experiment=adaptive_quant_config \
    training.epochs=100 \
    gfn.lambda_entropy=0.01 \
    +results_dir=results/eeg_adaptive_run1
```
*(Note: The exact command structure depends on the chosen configuration library, e.g., Hydra. The example uses '+' for command-line overrides).*

Training will save model checkpoints and logs (e.g., TensorBoard logs) to the specified `results_dir`.

### Evaluation

Use the `scripts/evaluate.py` script to evaluate a trained model on the test set.

```bash
python scripts/evaluate.py \
    --checkpoint_path results/eeg_adaptive_run1/checkpoints/best_model.pt \
    --config_path results/eeg_adaptive_run1/config.yaml \
    --output_dir results/eeg_adaptive_run1/evaluation/
```
*(Note: Assumes the training config is saved alongside the checkpoint. The script needs to load the model and dataset based on the config and checkpoint).*

Evaluation will calculate metrics (WQL, MASE) and potentially save generated forecasts and plots to the `output_dir`.

### Ablation Studies

Conduct ablation studies by creating specific configuration files in `configs/experiment/` that modify only the component being ablated compared to the main configuration (`adaptive_quant_config.yaml`).

Examples:
*   **Fixed Quantization:** Create `fixed_quant_config.yaml` setting `quantization.adaptive: false` and a specific `quantization.k_initial`. Run training using this config.
*   **Uniform Backward Policy:** Create `uniform_pb_config.yaml` setting `policy.backward_policy_type: uniform`. Run training.
*   **No Entropy Bonus:** Create `no_entropy_config.yaml` setting `gfn.lambda_entropy: 0.0`. Run training.

Compare the results from these runs against the full model run using `scripts/plot_results.py` or analysis notebooks.

## Workflow and Implementation Notes

1.  **Core GFN Logic (`src/temporal_gfn/gfn/`):**
    *   Adapt the TB loss implementation from `gflownet-tlm`.
    *   Define the `GFNEnvironment` (`env.py`) specifically for time series:
        *   `state`: Represents the time window (e.g., a tensor).
        *   `action`: Represents the discrete bin index (integer `0` to `K-1`).
        *   `step(state, action)`: Implements the state transition (sliding window).
        *   `get_actions(state)`: Returns the set of possible actions (all `K` bins).
        *   `reward(state)`: Calculates reward based on the final generated trajectory (requires access to ground truth).
    *   Implement forward/backward policy wrappers (`policies.py`) that take the neural network models (`src/temporal_gfn/models/`) and handle STE.
    *   Implement trajectory sampling (`sampling.py`).

2.  **Models (`src/temporal_gfn/models/`):**
    *   Implement a standard Transformer Encoder architecture. The output layer size needs to be adjustable based on the current `K`.
    *   If using a learned backward policy, implement its architecture (could be another Transformer or MLP).

3.  **Data Handling (`src/temporal_gfn/data/`):**
    *   Leverage Chronos's approach for handling diverse time series datasets.
    *   Implement robust windowing (`windowing.py`) to create `(context_window, future_target)` pairs.
    *   Implement scaling (`scaling.py`), applying it per time series (like Chronos's mean scaling). Store scaling factors for inverse transform during evaluation/reward calculation.
    *   The dataloader should yield batches suitable for the GFN (initial states `s0` and corresponding targets `y`).

4.  **Quantization (`src/temporal_gfn/quantization/`):**
    *   `base.py`: Implement `quantize(value, vmin, vmax, K)` and `dequantize(bin_index, vmin, vmax, K)`.
    *   `adaptive.py`: Implement the logic to calculate `eta_e` based on logged reward/entropy statistics and update `K`. This module will interact with the trainer and the policy models (to signal layer resizing).

5.  **STE (`src/temporal_gfn/estimators.py`):**
    *   Implement the STE logic, likely as a custom PyTorch function or integrated directly into the forward policy's forward pass where hard/soft samples are generated.

6.  **Trainer (`src/temporal_gfn/trainers.py`):**
    *   Orchestrates the main training loop.
    *   Handles sampling trajectories using the GFN components.
    *   Calculates rewards (needs dequantized values and ground truth).
    *   Computes the TB loss (+ entropy).
    *   Performs backpropagation and optimizer steps.
    *   Manages the adaptive quantization updates (calling logic from `quantization/adaptive.py`).
    *   Handles logging (metrics, hyperparameters, `K` value) and checkpointing.

7.  **Evaluator (`src/temporal_gfn/evaluators.py`):**
    *   Loads a trained model checkpoint.
    *   Generates multiple forecast trajectories for the test set (sampling from the learned policy).
    *   Dequantizes the generated bin indices and applies inverse scaling.
    *   Calculates probabilistic metrics (e.g., WQL over multiple trajectories) and point metrics (e.g., MASE on the median forecast) using utility functions adapted from Chronos (`utils.py`).

## Code Structure (`src/temporal_gfn/`)

*   `data/`: Handles all time series input, preprocessing, and batching.
*   `models/`: Defines the neural network architectures (policy networks).
*   `gfn/`: Encapsulates the core GFN logic (environment, loss, sampling).
*   `quantization/`: Specific code for adaptive quantization.
*   `estimators.py`: STE implementation.
*   `trainers.py`/`evaluators.py`: High-level training/evaluation orchestration.
*   `utils.py`: Common utilities, logging, config loading, metrics.

## Dependencies

*   PyTorch
*   Transformers (Hugging Face)
*   NumPy
*   Pandas (for data handling)
*   Scikit-learn (potentially for metrics or baseline scaling)
*   Hydra-core (or another config library like OmegaConf/YAML)
*   TensorBoard (for logging)
*   Matplotlib / Seaborn (for plotting)
*   `tqdm`

(List specific versions in `requirements.txt` / `environment.yml`)

## Contributing

Please refer to CONTRIBUTING.md (if applicable). Open issues for bugs or feature requests. Pull requests are welcome.

```

**Workflow Explanation and Directions:**

1.  **Foundation:** Start by integrating the basic GFN training loop structure (sampling, loss calculation, backprop) from `gflownet-tlm` into `src/temporal_gfn/trainers.py` and `src/temporal_gfn/gfn/`. Use the Transformer from `src/temporal_gfn/models/transformer.py` as the policy network.
2.  **Time Series Integration:** Adapt the data loading, windowing, and scaling from Chronos into `src/temporal_gfn/data/`. Ensure the dataloader provides initial states (`s0`) and targets (`y`) needed by the trainer.
3.  **GFN Environment:** Define the `GFNEnvironment` in `src/temporal_gfn/gfn/env.py` for the time series task. State is the window tensor, actions are bin indices `0..K-1`. Implement the `step` function for the sliding window transition.
4.  **Quantization (Fixed K First):** Implement the basic `quantize` and `dequantize` functions in `src/temporal_gfn/quantization/base.py`. Integrate them into the action selection (forward policy) and reward calculation (trainer). Start with a fixed `K`.
5.  **STE Implementation:** Implement the STE logic in `src/temporal_gfn/estimators.py` and integrate it into the policy's forward pass (`src/temporal_gfn/gfn/policies.py`) to generate `ahard` and `asoft`. Ensure gradients flow correctly through `asoft`.
6.  **Reward Function:** Implement the forecasting reward function (Eq. \ref{eq:reward_method}) within the trainer or environment, using the (dequantized, inversely scaled) `asoft` sequence and ground truth `y`.
7.  **Basic Training Loop:** Get a basic version running with fixed `K`, uniform backward policy, and TB loss (+ optional entropy). Debug and ensure losses decrease and basic forecasting occurs.
8.  **Adaptive Quantization:** Implement the adaptive logic in `src/temporal_gfn/quantization/adaptive.py` and integrate the `K` update call into the trainer's epoch loop. Ensure the policy network's output layer dynamically resizes when `K` changes.
9.  **Learned Backward Policy (Optional):** If needed, implement the backward policy network and integrate its training into the main loss calculation.
10. **Evaluation:** Adapt evaluation metrics (WQL, MASE) from Chronos into `src/temporal_gfn/utils.py`. Implement the evaluation script (`scripts/evaluate.py`) to generate multiple samples, dequantize/unscale, and compute metrics.
11. **Configuration & Experiments:** Set up the Hydra (or alternative) configuration system. Create config files for datasets and ablation studies. Use `scripts/run_experiment.sh` to manage multiple runs.
12. **Analysis:** Use notebooks or `scripts/plot_results.py` to analyze logs and compare results across different configurations and ablations.
