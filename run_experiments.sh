#!/bin/bash

# Experiment Runner Script
# Usage: ./run_experiments.sh

echo "Starting Experiments..."
echo "======================="

uv run tests/test.py --checkpoint checkpoints/contrastive_pretrained_proj_256/best_model.pt --wdice --output_dir test_results/contrastive_pretrained_proj_256

uv run tests/test.py --checkpoint checkpoints/contrastive_cosine_restarts/best_model.pt --wdice --output_dir test_results/contrastive_cosine_restarts

echo ""
echo "======================="
echo "All ex  periments completed!"
  