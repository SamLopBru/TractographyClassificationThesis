#!/bin/bash

# Tests Runner Script
# Usage: ./run_tests.sh


uv run tests/test.py --checkpoint checkpoints/transformer_pretrained_finetune_focal/best_model.pt \
--output_dir tests/test_results/transformer_pretrained_finetune_focal


uv run tests/test.py --checkpoint checkpoints/transformer_pretrained_finetune/best_model.pt \
--output_dir tests/test_results/transformer_pretrained_finetune
