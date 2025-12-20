#!/bin/bash
# ===========================================================================
# GR00T 1.6 Finetuning for SO-101 with Augmented Dataset + Regularization
# ===========================================================================
#
# This script finetunes GR00T N1.6-3B on:
#   - AUGMENTED dataset (140 episodes: 70 original + 70 mirrored)
#   - Training regularization enabled (state dropout, rotation aug)
#
# Changes from train_groot_so101_1_6.sh:
#   1. Dataset: so101_pick_place_groot_augmented (bias-corrected)
#   2. Added --state-dropout-prob 0.1 (regularization)
#   3. Added --random-rotation-angle 5 (image augmentation)
#   4. 15k steps default (1 epoch with 2x data)
#
# Usage:
#   # First run (15k steps, ~5 hours)
#   bash train_groot_so101_augmented.sh
#
#   # Resume to 30k (in same output folder)
#   RESUME_FROM=outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-15000 \
#   bash train_groot_so101_augmented.sh
#
#   # Resume to 45k
#   RESUME_FROM=outputs/groot_1_6_augmented_YYYYMMDD_HHMMSS/checkpoint-30000 \
#   bash train_groot_so101_augmented.sh
#
#   # MVP test run (1500 steps)
#   MAX_STEPS=1500 bash train_groot_so101_augmented.sh
#
# Reference: custom/scripts/ver1_6/train_groot_so101_1_6.sh
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS - Modify these for tuning
# ===========================================================================

# --- Resume Configuration ---
# Set RESUME_FROM to resume from a checkpoint (e.g., outputs/.../checkpoint-15000)
RESUME_FROM="${RESUME_FROM:-}"

# --- Model Configuration ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# --- Dataset Configuration (AUGMENTED) ---
DATASET_PATH="${DATASET_PATH:-/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot_augmented}"
MODALITY_CONFIG="${MODALITY_CONFIG:-custom/scripts/ver1_6/so101_config_1_6.py}"

# --- Training Hyperparameters ---
MAX_STEPS="${MAX_STEPS:-15000}"           # 15k = 1 epoch with augmented data
LEARNING_RATE="${LEARNING_RATE:-1e-4}"    # NVIDIA recommended
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"   # 8 for 24GB VRAM
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Regularization (NEW for augmented training) ---
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.1}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-5}"

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-1500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

# --- Output Directory with Timestamp ---
# Generate timestamp for unique output directory (only used for fresh start)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot_1_6_augmented}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

# --- Data Augmentation ---
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"

# --- Resources ---
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- Logging ---
USE_WANDB="false"

# ===========================================================================
# RESUME LOGIC
# ===========================================================================

if [ -n "$RESUME_FROM" ]; then
    # Validate checkpoint exists
    if [ ! -d "$RESUME_FROM" ]; then
        echo "ERROR: Checkpoint not found at $RESUME_FROM"
        exit 1
    fi

    # Use checkpoint as base model
    BASE_MODEL="$RESUME_FROM"

    # Disable warmup for resume
    WARMUP_RATIO="0.0"

    # Use same output directory as checkpoint (parent of checkpoint folder)
    OUTPUT_DIR="$(dirname "$RESUME_FROM")"

    echo ">>> RESUME MODE"
    echo "    Checkpoint: $RESUME_FROM"
    echo "    Output dir: $OUTPUT_DIR"
    echo "    Warmup:     $WARMUP_RATIO (disabled for resume)"
else
    echo ">>> FRESH START"
    echo "    Base model: $BASE_MODEL"
    echo "    Output dir: $OUTPUT_DIR"
fi

# ===========================================================================
# END OF KEY HYPERPARAMETERS
# ===========================================================================

# Print configuration
echo "=================================================================="
echo "GR00T 1.6 Finetuning - AUGMENTED Dataset + Regularization"
echo "=================================================================="
echo "Model:              $BASE_MODEL"
echo "Dataset:            $DATASET_PATH"
echo "Output:             $OUTPUT_DIR"
echo "Max steps:          $MAX_STEPS"
echo "Batch size:         $GLOBAL_BATCH_SIZE"
echo "Learning rate:      $LEARNING_RATE"
echo "Warmup ratio:       $WARMUP_RATIO"
echo "Weight decay:       $WEIGHT_DECAY"
echo "------ Regularization ------"
echo "State dropout:      $STATE_DROPOUT_PROB"
echo "Random rotation:    $RANDOM_ROTATION_ANGLE deg"
echo "Color jitter:       $COLOR_JITTER"
echo "Save steps:         $SAVE_STEPS"
echo "Num GPUs:           $NUM_GPUS"
echo "=================================================================="

# Validate dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run augment_dataset_horizontal.py first"
    exit 1
fi

# Validate modality config exists
if [ ! -f "$MODALITY_CONFIG" ]; then
    echo "ERROR: Modality config not found at $MODALITY_CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Logging Setup ---
LOG_FILE="$OUTPUT_DIR/training.log"
echo "Logging to: $LOG_FILE"

# No wandb flag - logging disabled
WANDB_FLAG=""

# Export for multi-GPU (if needed)
export NUM_GPUS=$NUM_GPUS

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save training configuration to log
{
    echo "=================================================================="
    echo "Training Started: $(date)"
    echo "=================================================================="
    echo "Model:              $BASE_MODEL"
    echo "Dataset:            $DATASET_PATH"
    echo "Output:             $OUTPUT_DIR"
    echo "Max steps:          $MAX_STEPS"
    echo "Batch size:         $GLOBAL_BATCH_SIZE"
    echo "Learning rate:      $LEARNING_RATE"
    echo "Warmup ratio:       $WARMUP_RATIO"
    echo "Weight decay:       $WEIGHT_DECAY"
    echo "State dropout:      $STATE_DROPOUT_PROB"
    echo "Random rotation:    $RANDOM_ROTATION_ANGLE"
    echo "Color jitter:       $COLOR_JITTER"
    echo "Save steps:         $SAVE_STEPS"
    echo "Num GPUs:           $NUM_GPUS"
    if [ -n "$RESUME_FROM" ]; then
        echo "Resume from:        $RESUME_FROM"
    fi
    echo "=================================================================="
    echo ""
} >> "$LOG_FILE"

log_msg "Starting training..."

# Launch training with logging to both terminal and file
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "$BASE_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --modality_config_path "$MODALITY_CONFIG" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --num_gpus $NUM_GPUS \
    --output_dir "$OUTPUT_DIR" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    --state_dropout_prob $STATE_DROPOUT_PROB \
    --random_rotation_angle $RANDOM_ROTATION_ANGLE \
    $WANDB_FLAG \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params $COLOR_JITTER \
    --dataloader_num_workers $DATALOADER_WORKERS \
    2>&1 | tee -a "$LOG_FILE"

# Log completion
log_msg "Training complete!"

{
    echo ""
    echo "=================================================================="
    echo "Training Complete: $(date)"
    echo "=================================================================="
    echo "Checkpoint saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Open-loop evaluation:"
    echo "     python custom/scripts/ver1_6/eval_openloop_1_6.py --checkpoint $OUTPUT_DIR/checkpoint-$MAX_STEPS"
    echo ""
    echo "  2. Robot inference:"
    echo "     python custom/scripts/ver1_6/infer_groot_so101_1_6.py --checkpoint $OUTPUT_DIR/checkpoint-$MAX_STEPS"
    echo ""
    echo "  3. Resume training (add more steps):"
    echo "     RESUME_FROM=$OUTPUT_DIR/checkpoint-$MAX_STEPS MAX_STEPS=15000 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh"
    echo "=================================================================="
} | tee -a "$LOG_FILE"
