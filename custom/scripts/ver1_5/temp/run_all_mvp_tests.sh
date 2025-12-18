#!/bin/bash
# Run all MVP tests for GR00T pipeline verification
# Usage: bash custom/scripts/run_all_mvp_tests.sh

set -e

echo "=============================================="
echo "  GR00T Pipeline MVP Test Suite"
echo "=============================================="
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate groot

cd /home/jrobot/project/Isaac-GR00T

echo "Running MVP Test 1: Data Loading..."
echo "----------------------------------------------"
python custom/scripts/test_data_loading_mvp.py
echo ""

echo "Running MVP Test 2: Normalization Pipeline..."
echo "----------------------------------------------"
python custom/scripts/test_normalization_mvp.py
echo ""

echo "=============================================="
echo "  All MVP Tests Complete"
echo "=============================================="
echo ""
echo "To run full model inference test (requires GPU):"
echo "  python custom/scripts/test_model_inference_mvp.py"
