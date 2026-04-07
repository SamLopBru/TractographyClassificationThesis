#!/bin/bash

# Tests Runner Script
# Usage: ./run_tests.sh

echo "========================================================="
echo "Batch Evaluation of All Fine-Tuned Checkpoints"
echo "========================================================="

CHECKPOINTS_DIR="checkpoints"
RESULTS_DIR="tests/test_results_2"

for exp_dir in "$CHECKPOINTS_DIR"/*/; do
    # Strip trailing slash to get the exact folder name
    exp_dir=${exp_dir%/}
    exp_name=$(basename "$exp_dir")
    
    checkpoint_path="$exp_dir/best_model.pt"
    
    # Only test if best_model.pt exists (this skips pure pretrained encoders)
    if [ -f "$checkpoint_path" ]; then
        echo -e "\n\n🚀 Evaluating Model: $exp_name"
        output_path="$RESULTS_DIR/$exp_name"
        
        # Run test script with wDice and Bootstrapping CIs
        uv run tests/test.py \
            --checkpoint "$checkpoint_path" \
            --output_dir "$output_path" \
            --sampling_pct_test 0.5
            
            
    else
        echo -e "\n⏭️ Skipping $exp_name (No best_model.pt found - likely a pretrained encoder)"
    fi
done

echo -e "\n✅ All batch evaluations complete!"
