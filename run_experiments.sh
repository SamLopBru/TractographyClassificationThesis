#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

# Common settings for fast iteration (bs=1024, pct=0.05 based on previous findings)
COMMON_ARGS="--batch_size 1024 --epoch_sampling_pct 0.05 --epochs 20 --num_workers 8"

echo "Starting Experiments..."
echo "======================="

# 1. CLS + more data (highest expected gain)
echo "Running Experiment 1: CLS + more data"
uv run src/train.py $COMMON_ARGS --pooling cls --epoch_sampling_pct 0.15 --val_sampling_pct 0.15

# 2. CLS + longer training
echo "Running Experiment 2: CLS + longer training"
uv run src/train.py $COMMON_ARGS --pooling cls --epochs 40 --patience 10

# 3. LSTM L4
echo "Running Experiment 3: LSTM L4 + CLS"
uv run src/train.py $COMMON_ARGS --encoder_type lstm --num_layers 4 --pooling cls

echo "======================="
echo "All experiments completed!"
