#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

# Common settings for fast iteration (bs=1024, pct=0.05 based on previous findings)
COMMON_ARGS="--batch_size 1024 --epoch_sampling_pct 0.05 --epochs 20 --num_workers 8"

echo "Starting Experiments..."
echo "======================="

# 1. CLS + more data (highest expected gain)
echo "Running Experiment 1: CLS + more time + more warmup"
uv run src/train.py --pooling cls --epochs 40 --patience 10 --epoch_sampling_pct 0.05 --warmup_steps 1500 --save_dir checkpoints/40

echo "Running Experiment 2: CLS + LR 5e-5 + more warmup"
uv run src/train.py --pooling cls --epochs 30 --patience 10 --base_lr 5e-5 --warmup_steps 1000 --save_dir checkpoints/30


echo "======================="
echo "All experiments completed!"
