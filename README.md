# Transformer-Based White Matter Tractography Segmentation

Deep learning for automated white matter tractography segmentation using Transformer architectures, spherical-coordinate streamline representations, and supervised contrastive learning.

## Overview

White matter tractography reconstructs the brain's white matter pathways from diffusion MRI (dMRI). Automatically identifying and segmenting these pathways is challenging due to the complexity of their geometry and the anatomical variability across subjects.

This project investigates deep learning approaches for automated tractography segmentation, focusing on Transformer architectures and training strategies designed to improve classification performance, representation quality, and probability calibration.

The pipeline represents tractography streamlines using normalized spherical coordinates, avoiding computationally expensive rigid registration. Recurrent neural networks and Transformer-based architectures are benchmarked to investigate their suitability for processing sequential streamline data.

## Key Features

- **Spherical-coordinate representations:** Process streamline geometry without relying on computationally expensive rigid registration.
- **Architecture benchmarking:** Compare recurrent neural networks, including LSTM and GRU variants, with Transformer-based models.
- **Configurable Transformer models:** Experiment with architectural parameters, pooling strategies, and feedforward dimensions.
- **Supervised contrastive learning:** Explore contrastive pretraining to improve the organization of learned representations.
- **Focal loss:** Investigate alternative optimization objectives for improving classification performance and probability calibration.
- **Flexible experimentation:** Configure model, training, sampling, and optimization parameters through `src/config.py` and the available command-line arguments.

## Methodology

The experimental pipeline explores the following components:

1. **Streamline representation:** Transform tractography streamlines into normalized spherical-coordinate representations.
2. **Model training:** Train and compare recurrent and Transformer-based architectures for tract classification.
3. **Training optimization:** Investigate hyperparameters, learning-rate warmup, batch size, gradient accumulation, and alternative loss functions.
4. **Contrastive pretraining:** Learn structured representations using supervised contrastive learning before downstream fine-tuning.
5. **Evaluation and analysis:** Assess classification performance, probability calibration, learned representations, and anatomical consistency.

The contrastive approach combines supervised contrastive pretraining with subsequent fine-tuning using focal loss.

## Results

The experiments were evaluated on **TractoInferno**, a large-scale, multicenter tractography dataset.

The contrastive pretraining + focal loss approach achieved the following results:

| Metric | Result |
|---|---:|
| Macro F1-score | 91.85% |
| Weighted Dice score (wDice) | 85.97% |
| Expected Calibration Error (ECE) | 2.91% |

Analysis of the learned representations using t-SNE showed more compact clusters and clearer hemispheric separation with contrastive pretraining. The remaining classification errors were concentrated particularly in anatomically complex regions with substantial pathway convergence and tissue transitions.

These results reflect the evaluated experimental configuration and should not be interpreted as evidence of clinical deployment or production readiness.

## Getting Started

### Prerequisites

- Python version compatible with the project dependencies.
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
- Access to the required tractography dataset and any preprocessing outputs expected by the training pipeline.

### Installation

Clone the repository and enter the project directory:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Install the project dependencies using `uv`:

```bash
uv sync
```

Make sure the dataset and any required configuration or preprocessing files are available at the paths expected by the project.

## Configuration

Experiment parameters are defined in `src/config.py`. Use this file as the starting point for understanding the available configuration options and their default values.

Parameters can also be supplied through command-line arguments where supported, allowing experiments to be configured without modifying the source code.

Examples of configurable parameters include:

- **Architecture:** model dimension, feedforward dimension, and pooling strategy.
- **Training:** number of epochs, batch size, and gradient accumulation.
- **Optimization:** warmup steps, patience, and loss-related settings.
- **Data sampling:** epoch sampling percentage and number of data-loader workers.
- **Contrastive learning:** temperature and checkpoint output directory.

To inspect the command-line options exposed by each training script, run:

```bash
uv run src/train.py --help
uv run src/contrastive_train.py --help
```

The exact set of supported arguments depends on the corresponding script and configuration implementation.

## Running Experiments

### 1. Supervised training

Run a Transformer experiment using CLS pooling and a feedforward dimension of 1024:

```bash
uv run src/train.py \
  --pooling cls \
  --epochs 40 \
  --patience 10 \
  --warmup_steps 1500 \
  --d_model 256 \
  --batch_size 512 \
  --accumulation_steps 4 \
  --dim_feedforward 1024
```

### 2. Contrastive pretraining

Run the contrastive training pipeline with CLS pooling and a temperature of 0.07:

```bash
uv run src/contrastive_train.py \
  --pooling cls \
  --epochs 30 \
  --patience 10 \
  --warmup_steps 1500 \
  --save_dir checkpoints/contrastive \
  --temperature 0.07
```

These commands illustrate two example experimental configurations. Adjust the arguments to explore alternative architectures and training strategies.

### Running multiple experiments

The following Bash script runs both example configurations sequentially:

```bash
#!/bin/bash
set -e

echo "Starting experiments..."
echo "======================="

echo "Experiment 1: CLS pooling + 1024 feedforward dimension"

uv run src/train.py \
  --pooling cls \
  --epochs 40 \
  --patience 10 \
  --warmup_steps 1500 \
  --d_model 256 \
  --batch_size 512 \
  --accumulation_steps 4 \
  --dim_feedforward 1024

echo "Experiment 2: Supervised contrastive learning"

uv run src/contrastive_train.py \
  --pooling cls \
  --epochs 30 \
  --patience 10 \
  --warmup_steps 1500 \
  --save_dir checkpoints/contrastive \
  --temperature 0.07

echo "======================="
echo "All experiments completed!"
```

Save the script as `run_experiments.sh` and execute it with:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

The script stops if an experiment exits with an error. Ensure the dataset, dependencies, and required configuration are available before running it.

## Reproducibility

For reproducible comparisons between experiments:

- Keep track of the configuration and command-line arguments used for each run.
- Use consistent dataset splits and evaluation procedures.
- Record random seeds where supported by the implementation.
- Preserve model checkpoints and evaluation outputs.
- Compare multiple metrics rather than relying on a single classification score.

## Project Structure

The main entry points currently include:

```text
.
├── src/
│   ├── config.py
│   ├── train.py
│   └── contrastive_train.py
├── checkpoints/
└── README.md
```

Additional modules, data-processing scripts, evaluation utilities, and configuration files may be present elsewhere in the repository.

## Research Context

This project was developed as part of a Master's thesis in Artificial Intelligence at the University of Alicante. It investigates Transformer-based architectures and contrastive learning strategies for automated white matter tractography segmentation.

The work is intended for research and experimentation. Clinical use would require further validation and assessment within the relevant clinical setting.

## Acknowledgements

The experiments use TractoInferno as the evaluation dataset. Please consult the dataset's official documentation and associated publication for details on data access, citation, and usage conditions.

## Final Note
The project is currently been improved and more experiments are been carried on with publication purposes. The full thesis with all results and conclusions can be found in [Master's thesis.pdf](https://github.com/SamLopBru/TractographyClassificationThesis/blob/main/MasterThesis.pdf)
