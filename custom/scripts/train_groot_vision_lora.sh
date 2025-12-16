#!/bin/bash
################################################################################
# GR00T Vision LoRA Training Script
# Purpose: LoRA finetuning of BOTH Action Head AND Vision Backbone
# Fixes: "Blind Policy" / Oscillation due to frozen vision encoder
################################################################################

set -e

# Configuration
MAX_STEPS=8000
SAVE_STEPS=500
BATCH_SIZE=16
GRAD_ACCUM=2
LEARNING_RATE=1e-4
LORA_RANK=16
NUM_WORKERS=4
VIDEO_BACKEND=torchvision_av

# Paths
DATASET_PATH="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_vision_lora_${TIMESTAMP}"
ISAAC_GROOT_ROOT="$HOME/project/Isaac-GR00T"

echo "========================================================================"
echo "GR00T Vision LoRA Training"
echo "========================================================================"
echo "  - Vision Encoder: TUNED (LoRA)"
echo "  - Action Head:    TUNED (LoRA)"
echo "  - Dataset:        $DATASET_PATH"
echo "  - Output:         $OUTPUT_DIR"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate groot

# Create output dir
mkdir -p $OUTPUT_DIR

# Run Training
cd $ISAAC_GROOT_ROOT

# NOTE: --lora-full-model enables LoRA on backbone (Vision/LLM) + Action Head
# NOTE: --tune-visual ensures the backbone is considered trainable
python -W ignore scripts/gr00t_finetune.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus 1 \
    --max-steps $MAX_STEPS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --data-config so100_dualcam \
    --video-backend $VIDEO_BACKEND \
    --lora-rank $LORA_RANK \
    --lora-full-model \
    --tune-visual \
    --save-steps $SAVE_STEPS \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --warmup-ratio 0.05 \
    --dataloader_num_workers $NUM_WORKERS \
    --report-to tensorboard \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training finished. Run evaluation/inference on: $OUTPUT_DIR/best"

