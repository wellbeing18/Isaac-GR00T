#!/bin/bash
################################################################################
# GR00T N1.5 MVP Training Script for SO-101
# Purpose: Quick validation that GR00T LoRA training works with your dataset
# Duration: ~1-2 hours for 500 steps
# VRAM Usage: ~18-20GB with --no-tune_diffusion_model flag
################################################################################

set -e  # Exit on error

# Suppress torchvision video deprecation warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"

echo "========================================================================"
echo "GR00T N1.5 MVP Training Test - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Purpose: Validate GR00T LoRA training pipeline"
echo "Duration: ~1-2 hours (500 steps)"
echo "Dataset: Your existing 10 episodes"
echo "LoRA Rank: 16 (NVIDIA recommended)"
echo ""
echo "What this tests:"
echo "  ✓ Dataset compatible with GR00T"
echo "  ✓ LoRA training fits in 24GB VRAM"
echo "  ✓ Training runs without crashes"
echo "  ✓ Loss decreases consistently"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
# NOTE: If you converted v3 → v2, update DATASET_PATH to point to the v2 directory:
#       DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets_v2"

OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_mvp_test"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"

# Check if Isaac-GR00T is installed
if [ ! -d "$ISAAC_GROOT_ROOT" ]; then
    echo "❌ ERROR: Isaac-GR00T not found at $ISAAC_GROOT_ROOT"
    echo ""
    echo "Please install Isaac-GR00T first:"
    echo "  git clone https://github.com/NVIDIA/Isaac-GR00T"
    echo "  cd Isaac-GR00T"
    echo "  conda create -n groot python=3.10"
    echo "  conda activate groot"
    echo "  pip install -e .[base]"
    echo "  pip install --no-build-isolation flash-attn==2.7.1.post4"
    echo ""
    echo "Then set ISAAC_GROOT_ROOT:"
    echo "  export ISAAC_GROOT_ROOT=/path/to/Isaac-GR00T"
    exit 1
fi

# Activate conda environment
echo "[1/4] Activating groot conda environment..."
eval "$(conda shell.bash hook)"
conda activate groot

# Verify environment
echo "[2/4] Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check dataset
echo "[3/4] Verifying dataset..."
if [ ! -d "$DATASET_PATH/meta" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    echo "Expected structure:"
    echo "  $DATASET_PATH/meta/info.json"
    echo "  $DATASET_PATH/meta/modality.json"
    echo "  $DATASET_PATH/data/"
    echo "  $DATASET_PATH/videos/"
    exit 1
fi

# Check for modality.json (required by GR00T)
if [ ! -f "$DATASET_PATH/meta/modality.json" ]; then
    echo "❌ ERROR: meta/modality.json missing!"
    echo ""
    echo "GR00T requires a modality.json file that matches the --data-config."
    echo ""
    echo "To fix:"
    echo "  1. Copy so100_dualcam__modality.json template from Isaac-GR00T"
    echo "  2. Adapt camera keys to 'head' and 'left_wrist'"
    echo "  3. Save to: $DATASET_PATH/meta/modality.json"
    echo ""
    echo "See: jdocs/top_level/lora/COMPLETE_LORA_GUIDE.md (Dataset Preparation)"
    exit 1
fi

# Check dataset version
echo "Dataset found: $DATASET_PATH"
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")
    echo "Episodes: $TOTAL_EPISODES"

    # Warn if using LeRobot v3 (GR00T expects v2)
    DATASET_VERSION=$(python -c "import json; info=json.load(open('$DATASET_PATH/meta/info.json')); print(info.get('codebase_version', 'unknown'))" 2>/dev/null || echo "unknown")
    if [[ "$DATASET_VERSION" == v3* ]] || [[ "$DATASET_VERSION" == "3."* ]]; then
        echo ""
        echo "⚠️  WARNING: Dataset appears to be LeRobot v3"
        echo "GR00T tooling expects LeRobot v2 format."
        echo ""
        echo "If training fails, convert dataset to v2:"
        echo "  1. Run conversion (from LeRobot repo):"
        echo "     cd /home/jrobot/project/lerobot"
        echo "     python scripts/convert_dataset_v3_to_v2.py \\"
        echo "       --input $DATASET_PATH \\"
        echo "       --output ${DATASET_PATH}_v2"
        echo ""
        echo "  2. Update DATASET_PATH in this script to point to v2 directory"
        echo ""
        echo "See: https://github.com/huggingface/lerobot/pull/2109"
        echo ""
        echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
        read
    fi
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run MVP training test
echo "[4/4] Starting GR00T MVP training (500 steps)..."
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor GPU usage in another terminal: watch -n 5 nvidia-smi"
echo "Monitor CPU RAM: watch -n 10 'free -h'"
echo ""
echo "Training Configuration:"
echo "  - LoRA Rank: 16"
echo "  - Batch Size: 8"
echo "  - Steps: 500 (quick test)"
echo "  - Memory Optimization: --no-tune_diffusion_model (saves ~5GB VRAM)"
echo ""

cd $ISAAC_GROOT_ROOT

python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 500 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --no-tune_diffusion_model \
    --save-steps 500 \
    --gradient-accumulation-steps 1 \
    --warmup-ratio 0.05 \
    --report-to tensorboard

echo ""
echo "========================================================================"
echo "GR00T MVP Training Test Complete!"
echo "========================================================================"
echo ""
echo "✅ Pass Criteria:"
echo "   - Training completed 500 steps without crashes"
echo "   - VRAM stayed <22GB throughout"
echo "   - Loss decreased from initial value"
echo "   - Checkpoint saved successfully"
echo ""
echo "Check output at: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. If test passed: Collect 65-90 more episodes (reach 75-100 total)"
echo "  2. Then run full training: bash scripts/train_groot_so101_full.sh"
echo "  3. Deploy and evaluate on robot"
echo ""
echo "If test failed, see: jdocs/top_level/finetuning_pi0_5_issue.md"
echo ""
