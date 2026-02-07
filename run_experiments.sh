#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

# Common settings for fast iteration (bs=1024, pct=0.05 based on previous findings)
COMMON_ARGS="--batch_size 1024 --epoch_sampling_pct 0.05 --epochs 20 --num_workers 8"

echo "Starting Experiments..."
echo "======================="

# 1. Pooling Strategies (Baseline is 'mean')
echo "Running Experiment A: Pooling Strategies"

echo "1.1: CLS Pooling"
uv run src/train.py $COMMON_ARGS --encoder_type transformer --pooling cls --d_model 256 --num_layers 6

echo "1.2: Max Pooling"
uv run src/train.py $COMMON_ARGS --encoder_type transformer --pooling max --d_model 256 --num_layers 6

# 2. Scale Up Width (Baseline d_model=256)
echo "Running Experiment B: Scale Up Width"

echo "2.1: d_model=384"
uv run src/train.py $COMMON_ARGS --encoder_type transformer --pooling mean --d_model 384 --num_layers 6

echo "2.2: d_model=512"
uv run src/train.py $COMMON_ARGS --encoder_type transformer --pooling mean --d_model 512 --num_layers 6

# 3. Scale Up Depth (Baseline num_layers=6)
echo "Running Experiment C: Scale Up Depth"

echo "3.1: num_layers=10"
uv run src/train.py $COMMON_ARGS --encoder_type transformer --pooling mean --d_model 256 --num_layers 10

# 4. Lightweight LSTM (Baseline num_layers=6 -> 8.5M params)
echo "Running Experiment D: Lightweight LSTM"

echo "4.1: LSTM 2 Layers"
uv run src/train.py $COMMON_ARGS --encoder_type lstm --d_model 256 --num_layers 2

# 5. More Data (Baseline epoch_sampling_pct=0.05)
echo "Running Experiment E: More Data"

echo "5.1: 15% Data per Epoch"
uv run src/train.py --batch_size 1024 --epoch_sampling_pct 0.15 --val_sampling_pct 0.20 --epochs 20 --num_workers 8 --encoder_type transformer --pooling mean --d_model 256 --num_layers 6

echo "======================="
echo "All experiments completed!"
