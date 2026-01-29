#!/bin/bash
# ===========================================================================
# GR00T 1.6 Cloud Training Script for Bimanual SO-101
# ===========================================================================
#
# This script finetunes GR00T N1.6-3B on bimanual SO-101 robot data.
# Optimized for cloud GPU instances (A100, H100).
#
# FINETUNING MODES:
#   1. Default (Projector + DiT only) - for RTX 4090 (24GB):
#      - tune_visual=False, tune_llm=False
#      - tune_projector=True, tune_diffusion_model=True
#      - ~214M trainable params, VRAM: ~25GB
#
#   2. Vision + Action (RECOMMENDED for bimanual) - for A100 40GB:
#      - tune_visual=True, tune_llm=False
#      - ~300M+ trainable params, VRAM: ~35GB
#      - Better visual-spatial learning for bimanual coordination
#
#   3. Full (all components) - for A100 80GB / H100:
#      - tune_visual=True, tune_llm=True (with LoRA)
#      - VRAM: ~40-60GB
#
# Memory Requirements:
#   - RTX 4090 (24GB): Default mode only, batch_size=8
#   - A100 40GB: Vision+Action mode, batch_size=16-32
#   - A100 80GB / H100: Full mode, batch_size=32-64
#
# Usage:
#   # Default (RTX 4090) - Projector + DiT only
#   bash train_groot_bimanual.sh
#
#   # Vision + Action (A100 40GB) - RECOMMENDED for bimanual
#   TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 bash train_groot_bimanual.sh
#
#   # Full finetuning (A100 80GB / H100)
#   TUNE_VISUAL=true TUNE_LLM=true GLOBAL_BATCH_SIZE=32 bash train_groot_bimanual.sh
#
#   # Custom dataset
#   DATASET_PATH=/path/to/groot/dataset bash train_groot_bimanual.sh
#
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS
# ===========================================================================

# --- Dataset Configuration ---
# Must be in GR00T format (use convert_bimanual_to_groot.py first)
DATASET_PATH="${DATASET_PATH:-/home/jrobot/project/Isaac-GR00T/datasets/picknplace_bimanual_groot}"
MODALITY_CONFIG="${MODALITY_CONFIG:-custom/scripts/cloud/so101_bimanual_config_1_6.py}"

# --- Model Configuration ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# --- Training Hyperparameters ---
MAX_STEPS="${MAX_STEPS:-10000}"           # MVP: 1000, Full: 10000
LEARNING_RATE="${LEARNING_RATE:-1e-4}"    # NVIDIA recommended
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"   # 8 for 24GB VRAM
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Parameter Freezing Configuration ---
# Set to "true" to unfreeze, "false" to freeze
TUNE_VISUAL="${TUNE_VISUAL:-false}"           # Unfreeze vision encoder (recommended for bimanual on A100+)
TUNE_LLM="${TUNE_LLM:-false}"                 # Unfreeze LLM backbone (requires 80GB+ VRAM)
TUNE_PROJECTOR="${TUNE_PROJECTOR:-true}"      # Train projector layers (always recommended)
TUNE_DIFFUSION="${TUNE_DIFFUSION:-true}"      # Train diffusion model (always recommended)
LORA_RANK="${LORA_RANK:-16}"                  # LoRA rank if using LoRA for LLM

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

# --- Output Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot16_bimanual}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

# --- Data Augmentation ---
# Color jitter helps with lighting variations
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"

# --- Resources ---
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- Logging ---
USE_WANDB="${USE_WANDB:-false}"

# ===========================================================================
# END OF KEY HYPERPARAMETERS
# ===========================================================================

echo "=================================================================="
echo "GR00T 1.6 Bimanual Training Configuration"
echo "=================================================================="
echo "Model:          $BASE_MODEL"
echo "Dataset:        $DATASET_PATH"
echo "Output:         $OUTPUT_DIR"
echo "Max steps:      $MAX_STEPS"
echo "Batch size:     $GLOBAL_BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo ""
echo "Finetuning Mode:"
echo "  Tune Visual:    $TUNE_VISUAL"
echo "  Tune LLM:       $TUNE_LLM"
echo "  Tune Projector: $TUNE_PROJECTOR"
echo "  Tune Diffusion: $TUNE_DIFFUSION"
if [ "$TUNE_LLM" = "true" ]; then
    echo "  LoRA Rank:      $LORA_RANK"
fi
echo ""
echo "Other Settings:"
echo "  Warmup ratio:   $WARMUP_RATIO"
echo "  Weight decay:   $WEIGHT_DECAY"
echo "  Save steps:     $SAVE_STEPS"
echo "  Num GPUs:       $NUM_GPUS"
echo "  Color jitter:   $COLOR_JITTER"
echo "=================================================================="

# Validate dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo ""
    echo "To convert LeRobot bimanual dataset to GR00T format:"
    echo "  python custom/scripts/cloud/convert_bimanual_to_groot.py \\"
    echo "      --input /path/to/lerobot/datasets_bimanuel/picknplace_300_144 \\"
    echo "      --output $DATASET_PATH"
    exit 1
fi

# Validate modality config exists
if [ ! -f "$MODALITY_CONFIG" ]; then
    echo "ERROR: Modality config not found at $MODALITY_CONFIG"
    echo "Using default bimanual config..."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Logging Setup ---
LOG_FILE="$OUTPUT_DIR/training.log"
echo "Logging to: $LOG_FILE"

# W&B flag
WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

# Export for multi-GPU
export NUM_GPUS=$NUM_GPUS

# Save training configuration
{
    echo "=================================================================="
    echo "GR00T 1.6 Bimanual Training"
    echo "Started: $(date)"
    echo "=================================================================="
    echo "Model:          $BASE_MODEL"
    echo "Dataset:        $DATASET_PATH"
    echo "Modality:       $MODALITY_CONFIG"
    echo "Output:         $OUTPUT_DIR"
    echo "Max steps:      $MAX_STEPS"
    echo "Batch size:     $GLOBAL_BATCH_SIZE"
    echo "Learning rate:  $LEARNING_RATE"
    echo "Warmup ratio:   $WARMUP_RATIO"
    echo "Weight decay:   $WEIGHT_DECAY"
    echo "Save steps:     $SAVE_STEPS"
    echo "Num GPUs:       $NUM_GPUS"
    echo "Color jitter:   $COLOR_JITTER"
    echo ""
    echo "Finetuning Mode:"
    echo "  Tune Visual:    $TUNE_VISUAL"
    echo "  Tune LLM:       $TUNE_LLM"
    echo "  Tune Projector: $TUNE_PROJECTOR"
    echo "  Tune Diffusion: $TUNE_DIFFUSION"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

echo "Starting training..."

# Build tuning flags based on configuration
TUNE_FLAGS=""
if [ "$TUNE_VISUAL" = "true" ]; then
    TUNE_FLAGS="$TUNE_FLAGS --tune_visual"
    echo "  >> Unfreezing vision encoder (tune_visual=true)"
fi
if [ "$TUNE_LLM" = "true" ]; then
    TUNE_FLAGS="$TUNE_FLAGS --tune_llm --lora_rank $LORA_RANK"
    echo "  >> Unfreezing LLM with LoRA rank=$LORA_RANK (tune_llm=true)"
fi
if [ "$TUNE_PROJECTOR" = "true" ]; then
    TUNE_FLAGS="$TUNE_FLAGS --tune_projector"
fi
if [ "$TUNE_DIFFUSION" = "true" ]; then
    TUNE_FLAGS="$TUNE_FLAGS --tune_diffusion_model"
fi

echo "  Tuning flags: $TUNE_FLAGS"
echo ""

# Launch training
# For single GPU: use plain python
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
    $TUNE_FLAGS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params $COLOR_JITTER \
    --dataloader_num_workers $DATALOADER_WORKERS \
    2>&1 | tee -a "$LOG_FILE"

# Log completion
{
    echo ""
    echo "=================================================================="
    echo "Training Complete: $(date)"
    echo "=================================================================="
    echo "Checkpoint saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Download checkpoint to local machine:"
    echo "     rsync -avz user@cloud:$OUTPUT_DIR ./outputs/"
    echo ""
    echo "  2. Run inference test:"
    echo "     python custom/scripts/cloud/infer_groot_bimanual.py \\"
    echo "         --checkpoint $OUTPUT_DIR/checkpoint-$MAX_STEPS"
    echo ""
    echo "  3. Test on robot (after download to local):"
    echo "     python custom/scripts/ver1_6/infer_groot_so101_1_6.py \\"
    echo "         --checkpoint ./outputs/groot16_bimanual_*/checkpoint-$MAX_STEPS"
    echo "=================================================================="
} | tee -a "$LOG_FILE"
