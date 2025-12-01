#!/bin/bash
################################################################################
# GR00T 5K LoRA Training Script for SO-101
# Purpose: Full LoRA finetuning run with monitoring and evaluation
# Duration: ~50-60 minutes for 5000 steps
# Goal: Train a usable model and evaluate performance
#
# STATUS: READY FOR 5K RUN (Updated 2025-12-01)
# - Includes pre-training verification
# - Post-training evaluation and diagnosis
# - Automatic best checkpoint selection
################################################################################

set -e  # Exit on error

# Suppress Python warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TORCHVISION_NO_DEPRECATION_WARNING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

# Training configuration (ADJUST THESE)
MAX_STEPS=5000       # 5K steps (~1.5 hours)
SAVE_STEPS=500       # Save checkpoint every 500 steps
BATCH_SIZE=4         # Batch size
LEARNING_RATE=1e-4   # Learning rate (4x higher than default)
LORA_RANK=16         # LoRA rank

echo "========================================================================"
echo "GR00T 5K LoRA Training - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Training Configuration:"
echo "  - Steps: $MAX_STEPS"
echo "  - Save every: $SAVE_STEPS steps"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - LoRA Rank: $LORA_RANK"
echo ""
echo "Duration: ~50-60 minutes"
echo "Expected VRAM: 18-20GB"
echo ""
echo "Features:"
echo "  ✓ Pre-training dataset verification"
echo "  ✓ TensorBoard logging"
echo "  ✓ Post-training MAE evaluation"
echo "  ✓ Inference diagnosis tests"
echo "  ✓ Best checkpoint selection"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Paths
DATASET_PATH_GROOT="/home/jrobot/project/XLeRobot/datasets_groot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S%3N)
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_5k_lora_${TIMESTAMP}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"
CUSTOM_SCRIPTS="$ISAAC_GROOT_ROOT/custom/scripts"

echo ""
echo "========================================================================"
echo "Step 1/8: Environment Check"
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
echo "Step 2/8: Pre-Training Verification"
echo "========================================================================"

# Run pre-training verification script
if [ -f "$CUSTOM_SCRIPTS/verify_groot_training_setup.py" ]; then
    echo "Running pre-training verification..."
    python "$CUSTOM_SCRIPTS/verify_groot_training_setup.py" \
        --dataset "$DATASET_PATH_GROOT" \
        --skip-videos

    VERIFY_EXIT_CODE=$?
    if [ $VERIFY_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Pre-training verification failed!"
        echo "Fix the issues above before training."
        exit 1
    fi
else
    echo "Warning: verify_groot_training_setup.py not found, skipping verification"
fi

# Check GR00T dataset exists
if [ ! -d "$DATASET_PATH_GROOT/meta" ]; then
    echo "ERROR: GR00T dataset not found at $DATASET_PATH_GROOT"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH_GROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_PATH_GROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo "0")
echo ""
echo "Dataset: $DATASET_PATH_GROOT"
echo "  Episodes: $TOTAL_EPISODES"
echo "  Frames: $TOTAL_FRAMES"

DATASET_PATH=$DATASET_PATH_GROOT

echo ""
echo "========================================================================"
echo "Step 3/8: Create modality.json for GR00T"
echo "========================================================================"

# Create modality.json if it doesn't exist (GR00T format)
if [ ! -f "$DATASET_PATH/meta/modality.json" ]; then
    echo "Creating modality.json for SO-101 (GR00T format)..."

    cat > "$DATASET_PATH/meta/modality.json" << 'EOF'
{
    "state": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "video": {
        "front": {
            "original_key": "observation.images.head"
        },
        "wrist": {
            "original_key": "observation.images.left_wrist"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
EOF

    echo "✓ Created modality.json (GR00T format)"
else
    echo "✓ modality.json already exists"
fi

# Verify modality.json (GR00T format)
echo "Validating modality.json..."
python -c "
import json
with open('$DATASET_PATH/meta/modality.json') as f:
    modality = json.load(f)
    # GR00T format validation
    print(f'  Video keys: {list(modality.get(\"video\", {}).keys())}')
    state_dim = sum(v['end'] - v['start'] for v in modality.get('state', {}).values())
    action_dim = sum(v['end'] - v['start'] for v in modality.get('action', {}).values())
    print(f'  State dimension: {state_dim}')
    print(f'  Action dimension: {action_dim}')
"

echo ""
echo "========================================================================"
echo "Step 4/8: Pre-Training Checks"
echo "========================================================================"

# Check CUDA availability
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits | head -1

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Step 5/8: Run 5K LoRA Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Steps: $MAX_STEPS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - LoRA Rank: $LORA_RANK"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Save every: $SAVE_STEPS steps"
echo "  - Expected Duration: ~50-60 minutes"
echo "  - Expected VRAM: ~18-20GB"
echo ""
echo "Monitor training:"
echo "  - TensorBoard: tensorboard --logdir $OUTPUT_DIR/runs"
echo "  - GPU: watch -n 5 nvidia-smi"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Change to Isaac-GR00T directory
cd $ISAAC_GROOT_ROOT

# Run mini training test
echo ""
echo "▶ Training started at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use unbuffer (from expect package) or script to preserve TTY for progress bar
if command -v unbuffer &> /dev/null; then
    unbuffer python -W ignore scripts/gr00t_finetune.py \
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
else
    # Direct execution with script for TTY preservation
    script -q -c "python -W ignore scripts/gr00t_finetune.py \
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
        --report-to tensorboard" $OUTPUT_DIR/training.log
fi

TRAINING_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶ Training finished at $(date)"
echo ""

echo ""
echo "========================================================================"
echo "Step 6/8: Post-Training Checkpoint Evaluation"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $OUTPUT_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi

# Run checkpoint evaluation
if [ -f "$CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py" ]; then
    echo ""
    echo "Evaluating checkpoints on training data..."
    python "$CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py" \
        --training-dir "$OUTPUT_DIR" \
        --dataset "$DATASET_PATH" \
        --num-samples 300 \
        --output "$OUTPUT_DIR/evaluation_results.json" \
        2>&1 | tee "$OUTPUT_DIR/evaluation.log"

    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR/evaluation_results.json"
else
    echo "Warning: evaluate_groot_checkpoint.py not found, skipping evaluation"
fi

echo ""
echo "========================================================================"
echo "Step 7/8: Inference Diagnosis"
echo "========================================================================"

# Find best checkpoint (latest or specifically marked)
BEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR/checkpoint-$MAX_STEPS" ]; then
    BEST_CHECKPOINT="$OUTPUT_DIR/checkpoint-$MAX_STEPS"
elif [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
    BEST_CHECKPOINT="$OUTPUT_DIR"
fi

if [ -n "$BEST_CHECKPOINT" ] && [ -f "$CUSTOM_SCRIPTS/diagnose_groot_inference.py" ]; then
    echo ""
    echo "Running inference diagnosis on: $BEST_CHECKPOINT"
    python "$CUSTOM_SCRIPTS/diagnose_groot_inference.py" \
        --checkpoint "$BEST_CHECKPOINT" \
        --dataset "$DATASET_PATH" \
        --num-samples 5 \
        --output "$OUTPUT_DIR/diagnosis_results.json" \
        2>&1 | tee "$OUTPUT_DIR/diagnosis.log"

    echo ""
    echo "Diagnosis results saved to: $OUTPUT_DIR/diagnosis_results.json"
else
    echo "Warning: diagnose_groot_inference.py not found or no checkpoint, skipping diagnosis"
fi

echo ""
echo "========================================================================"
echo "Step 8/8: Training Summary"
echo "========================================================================"

# Generate training report
REPORT_FILE="$OUTPUT_DIR/training_report.txt"

cat > $REPORT_FILE << EOF
================================================================================
GR00T 5K LoRA Training Report
Generated: $(date)
================================================================================

TRAINING CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: $DATASET_PATH
Episodes: $TOTAL_EPISODES
Frames: $TOTAL_FRAMES

Steps: $MAX_STEPS
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE
LoRA Rank: $LORA_RANK
Save Every: $SAVE_STEPS steps

TRAINING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exit Code: $TRAINING_EXIT_CODE
Training Log: $OUTPUT_DIR/training.log

Loss Progression:
$(grep -E "(step|loss)" $OUTPUT_DIR/training.log 2>/dev/null | tail -10 || echo "  (Check log file)")

CHECKPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | while read d; do echo "  - $(basename $d)"; done || echo "  (none)")
Final checkpoint: $OUTPUT_DIR/

EVALUATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$(cat $OUTPUT_DIR/evaluation_results.json 2>/dev/null | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for ckpt, metrics in data.items():
        if 'error' not in metrics:
            print(f\"  {ckpt.split('/')[-1]}:\")
            print(f\"    MAE: {metrics['overall_mae']:.2f}°\")
            print(f\"    Acc@5°: {metrics['accuracy']['acc@5']:.1f}%\")
            print(f\"    Acc@10°: {metrics['accuracy']['acc@10']:.1f}%\")
except: pass
" 2>/dev/null || echo "  (Run evaluation to see results)")

INFERENCE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run inference with finetuned model:
python $CUSTOM_SCRIPTS/infer_groot_so101.py \\
    --model-path $OUTPUT_DIR \\
    --task "pick red_cube from center"

# Re-run evaluation:
python $CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py \\
    --checkpoint $OUTPUT_DIR \\
    --dataset $DATASET_PATH

# Run diagnosis:
python $CUSTOM_SCRIPTS/diagnose_groot_inference.py \\
    --checkpoint $OUTPUT_DIR \\
    --dataset $DATASET_PATH

================================================================================
EOF

echo ""
cat $REPORT_FILE

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - Training log: $OUTPUT_DIR/training.log"
echo "  - Evaluation: $OUTPUT_DIR/evaluation_results.json"
echo "  - Diagnosis: $OUTPUT_DIR/diagnosis_results.json"
echo "  - Report: $REPORT_FILE"
echo ""
echo "Next steps:"
echo "  1. Check evaluation MAE - target < 10°"
echo "  2. Check diagnosis - all tests should pass"
echo "  3. Run real robot inference with infer_groot_so101.py"
echo ""
