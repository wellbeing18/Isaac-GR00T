#!/bin/bash
# ===========================================================================
# GR00T 1.6 Finetuning for SO-101
# ===========================================================================
#
# This script finetunes GR00T N1.6-3B on SO-101 robot data.
#
# IMPORTANT: GR00T 1.6 uses SELECTIVE PARAMETER FREEZING, not LoRA.
# Default configuration:
#   - tune_llm=False, tune_visual=False: Freeze VLM backbone (~2.8B params)
#   - tune_projector=True, tune_diffusion_model=True: Train action processing (~214M params)
# This fits comfortably in 24GB VRAM without LoRA.
#
# Decision Notes:
#   - Batch size 16: Safe for 24GB VRAM (~20GB peak)
#   - Learning rate 1e-4: NVIDIA recommended for finetuning
#   - Action horizon 16: Standard for smooth trajectory prediction
#   - Relative actions: Enabled by default (use_relative_action=True in launch_finetune.py)
#
# VRAM Requirements (RTX 5090/4090 24GB):
#   - Model loading: ~7-8 GB
#   - Training batch_size=8: ~20-22GB peak (1.6B trainable params)
#   - If OOM: reduce GLOBAL_BATCH_SIZE to 4
#
# Usage:
#   # MVP run (1000 steps, ~1 hour) - compare with baseline to decide full training
#   MAX_STEPS=1000 bash train_groot_so101_1_6.sh
#
#   # Full training (~3-4 hours)
#   bash train_groot_so101_1_6.sh
#
# Reference: examples/SO100/finetune_so100.sh, gr00t/experiment/launch_finetune.py
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS - Modify these for tuning
# ===========================================================================

# --- Model Configuration ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# --- Dataset Configuration ---
DATASET_PATH="${DATASET_PATH:-/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot}"
MODALITY_CONFIG="${MODALITY_CONFIG:-custom/scripts/ver1_6/so101_config_1_6.py}"

# --- Training Hyperparameters ---
MAX_STEPS="${MAX_STEPS:-10000}"           # MVP: 1000, Full: 10000
LEARNING_RATE="${LEARNING_RATE:-1e-4}"    # NVIDIA recommended
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"   # 8 for 24GB VRAM (1.6B trainable params)
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Parameter Freezing (GR00T 1.6 default) ---
# These are default values in FinetuneConfig, uncomment to override
# TUNE_LLM="--tune_llm"             # Default: False (frozen)
# TUNE_VISUAL="--tune_visual"       # Default: False (frozen)
# TUNE_PROJECTOR="--tune_projector" # Default: True (trainable)
# TUNE_DIFFUSION="--tune_diffusion_model"  # Default: True (trainable)

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

# --- Output Directory with Timestamp ---
# Generate timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot_1_6_so101}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

# --- Data Augmentation ---
# Color jitter helps with lighting variations in real-world deployment
# Format: brightness contrast saturation hue
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"

# --- Resources ---
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- Logging ---
# No wandb/tensorboard - use terminal output + open-loop eval for metrics
USE_WANDB="false"

# ===========================================================================
# END OF KEY HYPERPARAMETERS
# ===========================================================================

# Print configuration
echo "=================================================================="
echo "GR00T 1.6 Finetuning Configuration"
echo "=================================================================="
echo "Model:          $BASE_MODEL"
echo "Dataset:        $DATASET_PATH"
echo "Output:         $OUTPUT_DIR"
echo "Max steps:      $MAX_STEPS"
echo "Batch size:     $GLOBAL_BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Warmup ratio:   $WARMUP_RATIO"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Save steps:     $SAVE_STEPS"
echo "Num GPUs:       $NUM_GPUS"
echo "=================================================================="

# Validate dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run convert_lerobot_v3_to_groot_1_6.py first"
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
    echo "Model:          $BASE_MODEL"
    echo "Dataset:        $DATASET_PATH"
    echo "Output:         $OUTPUT_DIR"
    echo "Max steps:      $MAX_STEPS"
    echo "Batch size:     $GLOBAL_BATCH_SIZE"
    echo "Learning rate:  $LEARNING_RATE"
    echo "Warmup ratio:   $WARMUP_RATIO"
    echo "Weight decay:   $WEIGHT_DECAY"
    echo "Save steps:     $SAVE_STEPS"
    echo "Num GPUs:       $NUM_GPUS"
    echo "Color jitter:   $COLOR_JITTER"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

log_msg "Starting training..."

# Launch training with logging to both terminal and file
# For single GPU: use plain python (fixed factory.py barrier bug)
# For multi-GPU: use torchrun --nproc_per_node=$NUM_GPUS --master_port=29500
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
    echo "=================================================================="
} | tee -a "$LOG_FILE"
