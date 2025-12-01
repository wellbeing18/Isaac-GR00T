#!/bin/bash
################################################################################
# GR00T Mini-MVP LoRA Training Script for SO-101
# Purpose: Quick validation that training pipeline works (100 steps)
# Duration: ~5-10 minutes
# Goal: Verify dataset, model loading, and training loop before full run
#
# For full 5K training, use: train_groot_mvp.sh
################################################################################

set -e  # Exit on error

# Suppress Python warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TORCHVISION_NO_DEPRECATION_WARNING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

# Training configuration - MINI (100 steps for validation)
MAX_STEPS=100        # Quick validation only
SAVE_STEPS=50        # Save at 50 and 100
BATCH_SIZE=4         # Batch size
LEARNING_RATE=1e-4   # Learning rate
LORA_RANK=16         # LoRA rank

echo "========================================================================"
echo "GR00T Mini-MVP LoRA Training - Quick Validation"
echo "========================================================================"
echo ""
echo "Purpose: Validate training pipeline works before full 5K run"
echo ""
echo "Training Configuration:"
echo "  - Steps: $MAX_STEPS (mini validation)"
echo "  - Save every: $SAVE_STEPS steps"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - LoRA Rank: $LORA_RANK"
echo ""
echo "Duration: ~5-10 minutes"
echo "Expected VRAM: 18-20GB"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Paths
DATASET_PATH_GROOT="/home/jrobot/project/XLeRobot/datasets_groot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S%3N)
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_${TIMESTAMP}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"
CUSTOM_SCRIPTS="$ISAAC_GROOT_ROOT/custom/scripts"

echo ""
echo "========================================================================"
echo "Step 1/4: Environment Check"
echo "========================================================================"

# Check if Isaac-GR00T is installed
if [ ! -d "$ISAAC_GROOT_ROOT" ]; then
    echo "ERROR: Isaac-GR00T not found at $ISAAC_GROOT_ROOT"
    exit 1
fi

# Activate groot environment
echo "Activating groot conda environment..."
eval "$(conda shell.bash hook)"
conda activate groot

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "========================================================================"
echo "Step 2/4: Dataset Check"
echo "========================================================================"

# Check GR00T dataset exists
if [ ! -d "$DATASET_PATH_GROOT/meta" ]; then
    echo "ERROR: GR00T dataset not found at $DATASET_PATH_GROOT"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH_GROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_PATH_GROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo "0")
echo "Dataset: $DATASET_PATH_GROOT"
echo "  Episodes: $TOTAL_EPISODES"
echo "  Frames: $TOTAL_FRAMES"

DATASET_PATH=$DATASET_PATH_GROOT

# Check modality.json exists
if [ ! -f "$DATASET_PATH/meta/modality.json" ]; then
    echo "ERROR: modality.json not found. Run convert_lerobot_v3_to_groot.py first."
    exit 1
fi

echo ""
echo "========================================================================"
echo "Step 3/4: Run Mini Training (100 steps)"
echo "========================================================================"

# Check CUDA availability
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits | head -1

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Starting training..."
sleep 2

# Change to Isaac-GR00T directory
cd $ISAAC_GROOT_ROOT

echo ""
echo "▶ Training started at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -W ignore scripts/gr00t_finetune.py \
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
    --report-to tensorboard \
    2>&1 | tee $OUTPUT_DIR/training.log

TRAINING_EXIT_CODE=$?

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶ Training finished at $(date)"

echo ""
echo "========================================================================"
echo "Step 4/4: Results"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Mini-MVP training completed successfully!"
    echo ""
    echo "Output: $OUTPUT_DIR"
    echo ""
    echo "Checkpoints:"
    ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | while read d; do echo "  - $(basename $d)"; done || echo "  (none)"
    echo ""
    echo "Next step: Run full 5K training with:"
    echo "  bash custom/scripts/train_groot_mvp.sh"
else
    echo "✗ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $OUTPUT_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi
