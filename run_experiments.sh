#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

# Common settings for fast iteration (bs=1024, pct=0.05 based on previous findings)
COMMON_ARGS="--batch_size 1024 --epoch_sampling_pct 0.05 --epochs 20 --num_workers 8"

echo "Starting Experiments..."
echo "======================="

echo "Running Experiment 1: CLS + 1024 feedforward"
uv run src/train.py --pooling cls --epochs 40 --patience 10 --warmup_steps 1500 --d_model 256 --batch_size 512 --accumulation_steps 4 --dim_feedforward 1024

echo "Running Experiment 2: Contrastive Learning"
uv run src/contrastive_train.py --pooling cls --epochs 30 --patience 10 --warmup_steps 1500 --save_dir checkpoints/contrastive --temperature 0.07


echo "======================="
echo "All experiments completed!"
