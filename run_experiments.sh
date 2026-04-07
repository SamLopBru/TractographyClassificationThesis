#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

echo "Starting experiments..."
echo ""

EXPERIMENT_NAME="transformer_base_config_rmsnorm_retrain"

echo "Training: $EXPERIMENT_NAME"
uv run src/train.py \
  --experiment_name $EXPERIMENT_NAME \
  --encoder_type "transformer" \
  --warmup_steps 1500 \
  --patience 15 \
  --norm_layer "rmsnorm" \
  --epochs 50

echo ""
echo "Full Test: $EXPERIMENT_NAME"
uv run tests/test.py \
  --checkpoint checkpoints/${EXPERIMENT_NAME}/best_model.pt \
  --output_dir tests/best_results/${EXPERIMENT_NAME} \
  --sampling_pct_test 0.5 \
  --wdice \
  --bootstrap

uv run tests/test.py --compare


echo ""
echo "======================="
echo "All experiments completed!"