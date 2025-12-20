#!/bin/bash
# ===========================================================================
# GR00T 1.6 Finetuning for SO-101 with Augmented Dataset + Regularization
# ===========================================================================
#
# This script finetunes GR00T N1.6-3B on:
#   - AUGMENTED dataset (140 episodes: 70 original + 70 mirrored)
#   - Training regularization enabled (state dropout, rotation aug)
#
# =========================== RESUME MODES ===================================
#
# MODE 1: PLANNED TRAINING (Recommended)
#   - Set MAX_STEPS to your target upfront (e.g., 45k)
#   - Stop at any checkpoint, evaluate, then resume with SAME max_steps
#   - LR scheduler continues correctly
#
#   # Start training to 45k
#   MAX_STEPS=45000 SAVE_STEPS=3000 bash train_groot_so101_augmented.sh
#
#   # Stop after checkpoint-15000 is saved (Ctrl+C), evaluate
#   # Resume with SAME max_steps (LR continues from where it left off)
#   RESUME_FROM=outputs/.../checkpoint-15000 MAX_STEPS=45000 \
#   bash train_groot_so101_augmented.sh
#
# MODE 2: EXTEND TRAINING (Beyond original max_steps)
#   - If you need to train more than originally planned
#   - MUST use constant LR to avoid scheduler jump
#
#   # Original training was 15k, now want to extend to 45k
#   RESUME_FROM=outputs/.../checkpoint-15000 MAX_STEPS=45000 \
#   LR_SCHEDULER_TYPE=constant LEARNING_RATE=1e-5 \
#   bash train_groot_so101_augmented.sh
#
# MODE 3: UNKNOWN DURATION (Can extend indefinitely)
#   - Use constant LR from the start
#   - Can add more steps anytime without scheduler issues
#
#   LR_SCHEDULER_TYPE=constant MAX_STEPS=50000 bash train_groot_so101_augmented.sh
#
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS
# ===========================================================================

# --- Resume Configuration ---
# Set to checkpoint path to resume (e.g., outputs/.../checkpoint-15000)
RESUME_FROM="${RESUME_FROM:-}"

# --- Model Configuration ---
# ALWAYS use the original base model - checkpoint weights are loaded via resume
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# --- Dataset Configuration (AUGMENTED) ---
DATASET_PATH="${DATASET_PATH:-/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot_augmented}"
MODALITY_CONFIG="${MODALITY_CONFIG:-custom/scripts/ver1_6/so101_config_1_6.py}"

# --- Training Hyperparameters ---
MAX_STEPS="${MAX_STEPS:-15000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Regularization ---
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.1}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-5}"

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-1500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

# --- Output Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot_1_6_augmented}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

# --- Data Augmentation ---
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"

# --- Resources ---
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# ===========================================================================
# RESUME LOGIC
# ===========================================================================

RESUME_CHECKPOINT_FLAG=""

if [ -n "$RESUME_FROM" ]; then
    # Validate checkpoint exists
    if [ ! -d "$RESUME_FROM" ]; then
        echo "ERROR: Checkpoint not found at $RESUME_FROM"
        exit 1
    fi

    # Use same output directory as checkpoint (parent of checkpoint folder)
    OUTPUT_DIR="$(dirname "$RESUME_FROM")"

    # Disable warmup for resume (already warmed up)
    WARMUP_RATIO="0.0"

    # Set the resume checkpoint flag
    RESUME_CHECKPOINT_FLAG="--resume_from_checkpoint $RESUME_FROM"

    # Check if extending beyond original max_steps
    # Read original max_steps from trainer_state.json
    TRAINER_STATE="$RESUME_FROM/trainer_state.json"
    if [ -f "$TRAINER_STATE" ]; then
        ORIGINAL_MAX_STEPS=$(python3 -c "import json; print(json.load(open('$TRAINER_STATE'))['max_steps'])")
        CHECKPOINT_STEP=$(python3 -c "import json; print(json.load(open('$TRAINER_STATE'))['global_step'])")

        echo ">>> RESUME MODE"
        echo "    Checkpoint:         $RESUME_FROM"
        echo "    Checkpoint step:    $CHECKPOINT_STEP"
        echo "    Original max_steps: $ORIGINAL_MAX_STEPS"
        echo "    New max_steps:      $MAX_STEPS"

        if [ "$MAX_STEPS" -gt "$ORIGINAL_MAX_STEPS" ]; then
            echo ""
            echo "    WARNING: Extending beyond original max_steps!"
            echo "    LR scheduler will recalculate, which may cause instability."
            if [ "$LR_SCHEDULER_TYPE" = "cosine" ]; then
                echo ""
                echo "    RECOMMENDATION: Use constant LR for extended training:"
                echo "    LR_SCHEDULER_TYPE=constant LEARNING_RATE=1e-5 RESUME_FROM=... bash ..."
                echo ""
                read -p "    Continue with cosine anyway? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Aborted. Set LR_SCHEDULER_TYPE=constant and try again."
                    exit 1
                fi
            fi
        elif [ "$MAX_STEPS" -eq "$ORIGINAL_MAX_STEPS" ]; then
            echo "    LR schedule will continue correctly (same max_steps)"
        fi
    else
        echo ">>> RESUME MODE (trainer_state.json not found)"
        echo "    Checkpoint: $RESUME_FROM"
    fi

    echo "    Output dir:   $OUTPUT_DIR"
    echo "    LR Scheduler: $LR_SCHEDULER_TYPE"
    echo "    Learning rate: $LEARNING_RATE"
else
    echo ">>> FRESH START"
    echo "    Base model:    $BASE_MODEL"
    echo "    Output dir:    $OUTPUT_DIR"
    echo "    Max steps:     $MAX_STEPS"
    echo "    LR Scheduler:  $LR_SCHEDULER_TYPE"
    echo "    Learning rate: $LEARNING_RATE"
fi

# ===========================================================================
# PRINT FULL CONFIGURATION
# ===========================================================================

echo "=================================================================="
echo "GR00T 1.6 Finetuning - AUGMENTED Dataset + Regularization"
echo "=================================================================="
echo "Model:              $BASE_MODEL"
echo "Dataset:            $DATASET_PATH"
echo "Output:             $OUTPUT_DIR"
echo "Max steps:          $MAX_STEPS"
echo "Batch size:         $GLOBAL_BATCH_SIZE"
echo "Learning rate:      $LEARNING_RATE"
echo "LR scheduler:       $LR_SCHEDULER_TYPE"
echo "Warmup ratio:       $WARMUP_RATIO"
echo "Weight decay:       $WEIGHT_DECAY"
echo "------ Regularization ------"
echo "State dropout:      $STATE_DROPOUT_PROB"
echo "Random rotation:    $RANDOM_ROTATION_ANGLE deg"
echo "Color jitter:       $COLOR_JITTER"
echo "Save steps:         $SAVE_STEPS"
echo "Num GPUs:           $NUM_GPUS"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume from:        $RESUME_FROM"
fi
echo "=================================================================="

# ===========================================================================
# VALIDATION
# ===========================================================================

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run augment_dataset_horizontal.py first"
    exit 1
fi

if [ ! -f "$MODALITY_CONFIG" ]; then
    echo "ERROR: Modality config not found at $MODALITY_CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ===========================================================================
# LOGGING SETUP
# ===========================================================================

LOG_FILE="$OUTPUT_DIR/training.log"
echo "Logging to: $LOG_FILE"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save training configuration to log
{
    echo "=================================================================="
    echo "Training Started: $(date)"
    echo "=================================================================="
    echo "Mode:               $([ -n "$RESUME_FROM" ] && echo "RESUME" || echo "FRESH")"
    echo "Model:              $BASE_MODEL"
    echo "Dataset:            $DATASET_PATH"
    echo "Output:             $OUTPUT_DIR"
    echo "Max steps:          $MAX_STEPS"
    echo "Batch size:         $GLOBAL_BATCH_SIZE"
    echo "Learning rate:      $LEARNING_RATE"
    echo "LR scheduler:       $LR_SCHEDULER_TYPE"
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

# ===========================================================================
# LAUNCH TRAINING
# ===========================================================================

export NUM_GPUS=$NUM_GPUS

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
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --state_dropout_prob $STATE_DROPOUT_PROB \
    --random_rotation_angle $RANDOM_ROTATION_ANGLE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params $COLOR_JITTER \
    --dataloader_num_workers $DATALOADER_WORKERS \
    $RESUME_CHECKPOINT_FLAG \
    2>&1 | tee -a "$LOG_FILE"

# ===========================================================================
# COMPLETION
# ===========================================================================

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
    echo "  3. Resume training (same max_steps - LR continues correctly):"
    echo "     RESUME_FROM=$OUTPUT_DIR/checkpoint-XXXXX MAX_STEPS=$MAX_STEPS bash custom/scripts/ver1_6/train_groot_so101_augmented.sh"
    echo ""
    echo "  4. Extend training (beyond current max_steps - use constant LR):"
    echo "     RESUME_FROM=$OUTPUT_DIR/checkpoint-$MAX_STEPS MAX_STEPS=<higher> LR_SCHEDULER_TYPE=constant LEARNING_RATE=1e-5 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh"
    echo "=================================================================="
} | tee -a "$LOG_FILE"
