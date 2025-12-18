#!/bin/bash
################################################################################
# GR00T N1.5 Full Training Script for SO-101
# Purpose: Full LoRA finetuning for production model
# Duration: ~6-8 hours for 10,000 steps
# VRAM Usage: ~18-20GB with --no-tune_diffusion_model flag
# Prerequisites: 75-100 episodes collected
################################################################################

set -e  # Exit on error

# Suppress torchvision video deprecation warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"

echo "========================================================================"
echo "GR00T N1.5 Full Training - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Duration: ~6-8 hours (10,000 steps)"
echo "Prerequisites: 75-100 episodes in dataset"
echo "Expected Results: 40-60% success (75 eps) to 65-80% (100 eps)"
echo ""
echo "IMPORTANT: This will use significant GPU resources."
echo "  - VRAM: 18-20GB / 24GB"
echo "  - Training time: 6-8 hours"
echo "  - Disk space: ~5-10GB for checkpoints"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
# NOTE: If you converted v3 → v2, update DATASET_PATH to point to the v2 directory:
#       DATASET_PATH="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets_v2"

OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_so101_v1"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"

# Training hyperparameters (NVIDIA recommended for SO-101)
MAX_STEPS=10000
BATCH_SIZE=16
LEARNING_RATE=1e-4
LORA_RANK=16            # NVIDIA's recommended rank
SAVE_STEPS=1000         # Save checkpoint every 1000 steps

# Check if Isaac-GR00T is installed
if [ ! -d "$ISAAC_GROOT_ROOT" ]; then
    echo "❌ ERROR: Isaac-GR00T not found at $ISAAC_GROOT_ROOT"
    echo ""
    echo "Please install Isaac-GR00T first or set ISAAC_GROOT_ROOT"
    exit 1
fi

# Activate conda environment
echo "[1/5] Activating groot conda environment..."
eval "$(conda shell.bash hook)"
conda activate groot

# Verify environment
echo "[2/5] Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify dataset
echo "[3/5] Verifying dataset..."
if [ ! -d "$DATASET_PATH/meta" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
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
TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")

echo "Dataset found: $DATASET_PATH"
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

# Validate episode count
if [ "$TOTAL_EPISODES" -lt 75 ]; then
    echo ""
    echo "⚠️  WARNING: Only $TOTAL_EPISODES episodes found!"
    echo "Recommended: 75-100 episodes for reliable performance"
    echo ""
    echo "You can:"
    echo "  1. Continue anyway (may underfit)"
    echo "  2. Collect more episodes first (recommended)"
    echo ""
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read
fi

# Create output directory
echo "[4/5] Creating output directory..."
mkdir -p $OUTPUT_DIR
echo "Checkpoints will be saved to: $OUTPUT_DIR"

# Display training configuration
echo "[5/5] Training Configuration:"
echo "  Model: GR00T N1.5 (3B params) with LoRA"
echo "  LoRA Rank: $LORA_RANK"
echo "  Steps: $MAX_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Save Frequency: Every $SAVE_STEPS steps"
echo "  Memory Optimization: --no-tune_diffusion_model"
echo ""
echo "⚠️  VRAM NOTE: Batch size 16 uses ~18-20GB VRAM"
echo "   If you see CUDA OOM errors:"
echo "   1. Edit this script: Change BATCH_SIZE=16 to BATCH_SIZE=8"
echo "   2. Optionally add: --gradient-accumulation-steps 2"
echo "   (This maintains effective batch size while using less VRAM)"
echo ""
echo "Starting training in 5 seconds..."
echo "Monitor GPU with: watch -n 5 nvidia-smi"
echo "Monitor CPU RAM with: watch -n 10 'free -h'"
echo ""
sleep 5

# Change to Isaac-GR00T directory
cd $ISAAC_GROOT_ROOT

# Run full training
python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps $MAX_STEPS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank $LORA_RANK \
    --no-tune_diffusion_model \
    --save-steps $SAVE_STEPS \
    --gradient-accumulation-steps 1 \
    --warmup-ratio 0.05 \
    --report-to tensorboard

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Available checkpoints:"
ls -lh $OUTPUT_DIR/checkpoint-*/
echo ""
echo "Next Steps:"
echo "  1. Evaluate finetuned model on robot"
echo "  2. Compare to pretrained baseline"
echo "  3. If performance good, deploy to production"
echo ""
echo "To run inference:"
echo "  cd $ISAAC_GROOT_ROOT"
echo "  python scripts/inference_service.py \\"
echo "    --model-path $OUTPUT_DIR/checkpoint-$MAX_STEPS \\"
echo "    --server \\"
echo "    --port 8000"
echo ""
echo "See jdocs/top_level/finetuning_pi0_5_issue.md for evaluation guide."
echo ""
