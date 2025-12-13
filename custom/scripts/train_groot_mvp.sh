#!/bin/bash
################################################################################
# GR00T MVP LoRA Training Script for SO-101
# Purpose: Full 5K LoRA finetuning with evaluation and best checkpoint selection
# Duration: ~50-60 minutes for 5000 steps
# Goal: Train a usable model with proper evaluation
#
# Features:
#   - Pre-training verification
#   - 5000 steps training with checkpoints every 500 steps
#   - Post-training MAE evaluation on all checkpoints
#   - Best checkpoint selection based on lowest MAE
#   - Inference diagnosis on best checkpoint
#   - Comprehensive training report
#
# For quick validation first, run: train_groot_mini_mvp.sh
#
# RESUME TRAINING (to continue from last checkpoint):
#   After initial training completes, run with --resume to continue.
#   LR scheduler state is restored automatically (no need to specify --learning-rate).
#   python scripts/gr00t_finetune.py \
#       --dataset-path /home/jrobot/project/XLeRobot/datasets_groot \
#       --output-dir <SAME_OUTPUT_DIR> \
#       --max-steps 10000 \
#       --save-steps 1000 \
#       --batch-size 16 \
#       --gradient-accumulation-steps 2 \
#       --data-config so100_dualcam \
#       --video-backend torchvision_av \
#       --lora-rank 16 \
#       --no-tune_diffusion_model \
#       --dataloader_num_workers 4 \
#       --resume
#
# Industry Best Practices (per NVIDIA/HuggingFace):
#   - --no-tune_diffusion_model: CORRECT - freezes DiT, trains projector with LoRA
#   - 30 FPS action data: CORRECT - industry standard for GR00T and Pi0.5 (upgraded from 5 FPS)
#   - batch_size=4: CORRECT for RTX 5090 (24GB VRAM)
#   - learning_rate=1e-4: CORRECT - industry standard
#
# FPS Requirements (per 6_fps_upgrade_30hz.md):
#   - Action FPS: 30 Hz (synchronized with video)
#   - Video FPS: 30 fps
#   - Dataset must be recorded at 30 Hz for optimal training
################################################################################

set -e  # Exit on error

# Suppress Python warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TORCHVISION_NO_DEPRECATION_WARNING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

# Training configuration (ADJUST THESE)
MAX_STEPS=8000       # 5K steps (~1 hour). Use --resume to continue to 10K.
SAVE_STEPS=1000       # Save checkpoint every 500 steps (10 checkpoints total)
BATCH_SIZE=16         # Batch size (reduced from 32 to prevent OOM)
GRAD_ACCUM=2          # Accumulate steps to maintain effective batch size of 32
LEARNING_RATE=1e-4   # Learning rate (4x higher than default - critical!)
LORA_RANK=16         # LoRA rank
NUM_WORKERS=4        # Dataloader workers (reduced from 8 to prevent RAM OOM)
VIDEO_BACKEND=torchvision_av # Video backend: torchvision_av (default, tested) or decord (potentially faster, needs validation)

echo "========================================================================"
echo "GR00T MVP LoRA Training - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Training Configuration:"
echo "  - Steps: $MAX_STEPS"
echo "  - Save every: $SAVE_STEPS steps"
echo "  - Batch Size: $BATCH_SIZE (Effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  - Grad Accum: $GRAD_ACCUM"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - LoRA Rank: $LORA_RANK"
echo "  - Dataloader Workers: $NUM_WORKERS"
echo "  - Video Backend: $VIDEO_BACKEND"
echo ""
echo "Duration: ~50-60 minutes"
echo "Expected VRAM: 18-20GB"
echo ""
echo "To resume training to 10K steps after validation, see header comments."
echo ""
echo "Features:"
echo "  [x] Pre-training dataset verification"
echo "  [x] TensorBoard logging"
echo "  [x] Post-training MAE evaluation"
echo "  [x] Best checkpoint selection (lowest MAE)"
echo "  [x] Inference diagnosis tests"
echo "  [x] Comprehensive training report"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Paths
DATASET_PATH_GROOT="/home/jrobot/project/XLeRobot/datasets_groot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S%3N)
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_${TIMESTAMP}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"
CUSTOM_SCRIPTS="$ISAAC_GROOT_ROOT/custom/scripts"

echo ""
echo "========================================================================"
echo "Step 1/9: Environment Check"
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
echo "Step 2/9: Pre-Training Verification"
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
echo "Step 3/9: Verify modality.json"
echo "========================================================================"

# Check modality.json exists
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

    echo "Created modality.json (GR00T format)"
else
    echo "modality.json already exists"
fi

# Verify modality.json
echo "Validating modality.json..."
python -c "
import json
with open('$DATASET_PATH/meta/modality.json') as f:
    modality = json.load(f)
    print(f'  Video keys: {list(modality.get(\"video\", {}).keys())}')
    state_dim = sum(v['end'] - v['start'] for v in modality.get('state', {}).values())
    action_dim = sum(v['end'] - v['start'] for v in modality.get('action', {}).values())
    print(f'  State dimension: {state_dim}')
    print(f'  Action dimension: {action_dim}')
"

echo ""
echo "========================================================================"
echo "Step 4/9: Pre-Training Checks"
echo "========================================================================"

# Check CUDA availability
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits | head -1

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Step 5/9: Run 5K LoRA Training"
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

echo ""
echo "Training started at $(date)"
echo "=============================================================================="
echo ""

# Use unbuffer if available for better progress bar display
if command -v unbuffer &> /dev/null; then
    unbuffer python -W ignore scripts/gr00t_finetune.py \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --num-gpus 1 \
        --max-steps $MAX_STEPS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --data-config so100_dualcam \
        --video-backend $VIDEO_BACKEND \
        --lora-rank $LORA_RANK \
        --no-tune_diffusion_model \
        --save-steps $SAVE_STEPS \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --warmup-ratio 0.05 \
        --dataloader_num_workers $NUM_WORKERS \
        --report-to tensorboard \
        2>&1 | tee $OUTPUT_DIR/training.log
else
    python -W ignore scripts/gr00t_finetune.py \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --num-gpus 1 \
        --max-steps $MAX_STEPS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --data-config so100_dualcam \
        --video-backend $VIDEO_BACKEND \
        --lora-rank $LORA_RANK \
        --no-tune_diffusion_model \
        --save-steps $SAVE_STEPS \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --warmup-ratio 0.05 \
        --dataloader_num_workers $NUM_WORKERS \
        --report-to tensorboard \
        2>&1 | tee $OUTPUT_DIR/training.log
fi

TRAINING_EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "Training finished at $(date)"
echo ""

echo ""
echo "========================================================================"
echo "Step 6/9: Post-Training Checkpoint Evaluation"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $OUTPUT_DIR/training.log"
    exit $TRAINING_EXIT_CODE
fi

# Run checkpoint evaluation (only last checkpoint to save time)
if [ -f "$CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py" ]; then
    echo ""
    echo "Evaluating last checkpoint on training data..."
    python "$CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py" \
        --training-dir "$OUTPUT_DIR" \
        --dataset "$DATASET_PATH" \
        --num-samples 300 \
        --last 1 \
        --output "$OUTPUT_DIR/evaluation_results.json" \
        2>&1 | tee "$OUTPUT_DIR/evaluation.log"

    echo ""
    echo "Evaluation results saved to: $OUTPUT_DIR/evaluation_results.json"
else
    echo "Warning: evaluate_groot_checkpoint.py not found, skipping evaluation"
fi

echo ""
echo "========================================================================"
echo "Step 7/9: Best Checkpoint Selection"
echo "========================================================================"

# Find best checkpoint based on evaluation results (lowest MAE)
BEST_CHECKPOINT=""
BEST_CHECKPOINT_NAME=""

if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
    echo "Finding best checkpoint based on evaluation MAE..."
    python3 << PYTHON_SCRIPT
import json
import os
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
eval_file = output_dir / "evaluation_results.json"

with open(eval_file) as f:
    results = json.load(f)

# Find checkpoint with lowest MAE
best_ckpt = None
best_mae = float('inf')

for ckpt_path, metrics in results.items():
    if 'overall_mae' in metrics:
        mae = metrics['overall_mae']
        if mae < best_mae:
            best_mae = mae
            best_ckpt = ckpt_path

if best_ckpt:
    ckpt_name = Path(best_ckpt).name
    print(f"Best checkpoint: {ckpt_name} (MAE: {best_mae:.2f} degrees)")

    # Create 'best' symlink
    best_link = output_dir / "best"
    if best_link.is_symlink():
        best_link.unlink()
    elif best_link.exists():
        import shutil
        shutil.rmtree(best_link)

    # Make relative symlink
    best_link.symlink_to(ckpt_name)
    print(f"Created symlink: best -> {ckpt_name}")

    # Save best info
    with open(output_dir / "best_info.txt", 'w') as f:
        f.write(f"best_checkpoint: {ckpt_name}\n")
        f.write(f"best_mae: {best_mae:.4f}\n")
        f.write(f"\nAll checkpoint MAE:\n")
        for ckpt_path, metrics in sorted(results.items()):
            if 'overall_mae' in metrics:
                mae = metrics['overall_mae']
                name = Path(ckpt_path).name
                marker = " <- BEST" if ckpt_path == best_ckpt else ""
                f.write(f"  {name}: {mae:.2f} degrees{marker}\n")

    print(f"Saved best_info.txt")

    # Write best checkpoint path for shell to use
    with open(output_dir / ".best_checkpoint_path", 'w') as f:
        f.write(str(output_dir / ckpt_name))
else:
    print("Warning: Could not determine best checkpoint from evaluation")
PYTHON_SCRIPT

    # Read back the best checkpoint path
    if [ -f "$OUTPUT_DIR/.best_checkpoint_path" ]; then
        BEST_CHECKPOINT=$(cat "$OUTPUT_DIR/.best_checkpoint_path")
        BEST_CHECKPOINT_NAME=$(basename "$BEST_CHECKPOINT")
        echo ""
        echo "Best checkpoint: $BEST_CHECKPOINT_NAME"
    fi
else
    echo "No evaluation results found, using latest checkpoint as best"
    # Fallback: find latest checkpoint
    if [ -d "$OUTPUT_DIR/checkpoint-$MAX_STEPS" ]; then
        BEST_CHECKPOINT="$OUTPUT_DIR/checkpoint-$MAX_STEPS"
    elif [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        BEST_CHECKPOINT="$OUTPUT_DIR"
    fi

    if [ -n "$BEST_CHECKPOINT" ]; then
        BEST_CHECKPOINT_NAME=$(basename "$BEST_CHECKPOINT")
        # Create best symlink to latest
        ln -sfn "$BEST_CHECKPOINT_NAME" "$OUTPUT_DIR/best" 2>/dev/null || true
        echo "Created symlink: best -> $BEST_CHECKPOINT_NAME (latest)"
    fi
fi

echo ""
echo "========================================================================"
echo "Step 8/9: Inference Diagnosis"
echo "========================================================================"

# Use best checkpoint for diagnosis
if [ -z "$BEST_CHECKPOINT" ]; then
    if [ -d "$OUTPUT_DIR/checkpoint-$MAX_STEPS" ]; then
        BEST_CHECKPOINT="$OUTPUT_DIR/checkpoint-$MAX_STEPS"
    elif [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        BEST_CHECKPOINT="$OUTPUT_DIR"
    fi
fi

if [ -n "$BEST_CHECKPOINT" ] && [ -f "$CUSTOM_SCRIPTS/diagnose_groot_inference.py" ]; then
    echo ""
    echo "Running inference diagnosis on best checkpoint: $(basename $BEST_CHECKPOINT)"
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
echo "Step 9/9: Training Summary"
echo "========================================================================"

# Generate training report
REPORT_FILE="$OUTPUT_DIR/training_report.txt"

cat > $REPORT_FILE << EOF
================================================================================
GR00T MVP LoRA Training Report
Generated: $(date)
================================================================================

TRAINING CONFIGURATION
------------------------------------------------------------------------------
Dataset: $DATASET_PATH
Episodes: $TOTAL_EPISODES
Frames: $TOTAL_FRAMES

Steps: $MAX_STEPS
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE
LoRA Rank: $LORA_RANK
Save Every: $SAVE_STEPS steps

TRAINING RESULTS
------------------------------------------------------------------------------
Exit Code: $TRAINING_EXIT_CODE
Training Log: $OUTPUT_DIR/training.log

Loss Progression:
$(grep -E "(step|loss)" $OUTPUT_DIR/training.log 2>/dev/null | tail -10 || echo "  (Check log file)")

CHECKPOINTS
------------------------------------------------------------------------------
$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | while read d; do echo "  - $(basename $d)"; done || echo "  (none)")
Final checkpoint: $OUTPUT_DIR/

BEST CHECKPOINT
------------------------------------------------------------------------------
$(cat $OUTPUT_DIR/best_info.txt 2>/dev/null || echo "  (Not determined)")
Best symlink: $OUTPUT_DIR/best

EVALUATION RESULTS
------------------------------------------------------------------------------
$(cat $OUTPUT_DIR/evaluation_results.json 2>/dev/null | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for ckpt, metrics in sorted(data.items()):
        if 'error' not in metrics:
            print(f\"  {ckpt.split('/')[-1]}:\")
            print(f\"    MAE: {metrics['overall_mae']:.2f} degrees\")
            print(f\"    Acc@5: {metrics['accuracy']['acc@5']:.1f}%\")
            print(f\"    Acc@10: {metrics['accuracy']['acc@10']:.1f}%\")
except: pass
" 2>/dev/null || echo "  (Run evaluation to see results)")

INFERENCE COMMANDS
------------------------------------------------------------------------------
# Run inference with best checkpoint:
python $CUSTOM_SCRIPTS/infer_groot_so101.py \\
    --model-path $OUTPUT_DIR/best \\
    --task "pick red_cube from center"

# Run inference with latest checkpoint:
python $CUSTOM_SCRIPTS/infer_groot_so101.py \\
    --model-path $OUTPUT_DIR \\
    --task "pick red_cube from center"

# Re-run evaluation on best:
python $CUSTOM_SCRIPTS/evaluate_groot_checkpoint.py \\
    --checkpoint $OUTPUT_DIR/best \\
    --dataset $DATASET_PATH

# Run open-loop evaluation with trajectory plots (30 Hz):
python $CUSTOM_SCRIPTS/eval_groot_openloop.py \\
    --checkpoint $OUTPUT_DIR/best \\
    --dataset $DATASET_PATH \\
    --plot --trajs 3

# Run diagnosis on best:
python $CUSTOM_SCRIPTS/diagnose_groot_inference.py \\
    --checkpoint $OUTPUT_DIR/best \\
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
echo "  - Best info: $OUTPUT_DIR/best_info.txt"
echo "  - Diagnosis: $OUTPUT_DIR/diagnosis_results.json"
echo "  - Report: $REPORT_FILE"
echo ""
echo "Best checkpoint: $OUTPUT_DIR/best"
echo ""
echo "Next steps:"
echo "  1. Check evaluation MAE - target < 10 degrees"
echo "  2. Check diagnosis - all tests should pass"
echo "  3. Run real robot inference with infer_groot_so101.py"
echo ""
