#!/bin/bash
################################################################################
# GR00T Mini-MVP Test Script for SO-101
# Purpose: Ultra-quick pipeline validation with existing 10 episodes
# Duration: ~5-10 minutes for 100 steps
# Goal: Verify data format, data loading, and generate logs for inspection
################################################################################

set -e  # Exit on error

# Suppress torchvision video deprecation warnings (harmless, just clutters logs)
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"

echo "========================================================================"
echo "GR00T Mini-MVP Pipeline Test - SO-101 Left Arm"
echo "========================================================================"
echo ""
echo "Purpose: Quick validation of training pipeline (100 steps)"
echo "Dataset: Current 10 episodes"
echo "Duration: ~5-10 minutes"
echo ""
echo "What this tests:"
echo "  ✓ Dataset v3 → v2 conversion (if needed)"
echo "  ✓ modality.json creation and validation"
echo "  ✓ Data loading works correctly"
echo "  ✓ Training loop executes without errors"
echo "  ✓ Generates logs for inspection"
echo ""
echo "What this DOES NOT test:"
echo "  ✗ Model performance (need more data + training)"
echo "  ✗ Loss convergence (100 steps too short)"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read

# Configuration
DATASET_PATH_V3="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets"
DATASET_PATH_GROOT="/home/jrobot/project/XLeRobot/jdocs/top_level/datasets_groot"
OUTPUT_DIR="/home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-$HOME/project/Isaac-GR00T}"

echo ""
echo "========================================================================"
echo "Step 1/6: Environment Check"
echo "========================================================================"

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

echo ""
echo "========================================================================"
echo "Step 2/6: Dataset Validation"
echo "========================================================================"

# Check GR00T dataset exists
if [ ! -d "$DATASET_PATH_GROOT/meta" ]; then
    echo "❌ ERROR: GR00T dataset not found at $DATASET_PATH_GROOT"
    echo "Run conversion script first to create GR00T-compatible dataset"
    exit 1
fi

TOTAL_EPISODES=$(python -c "import json; print(json.load(open('$DATASET_PATH_GROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")
echo "Found GR00T dataset: $TOTAL_EPISODES episodes at $DATASET_PATH_GROOT"
echo "(Original v3 dataset for Pi0.5 preserved at: $DATASET_PATH_V3)"

DATASET_PATH=$DATASET_PATH_GROOT

echo ""
echo "========================================================================"
echo "Step 3/6: Create modality.json for GR00T"
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
echo "Step 4/6: Pre-Training Checks"
echo "========================================================================"

# Check CUDA availability
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits | head -1

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Step 5/6: Run Mini-MVP Training (100 steps)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Steps: 100 (ultra-short test)"
echo "  - Batch Size: 4 (conservative for 10 episodes)"
echo "  - LoRA Rank: 16"
echo "  - Learning Rate: 1e-4"
echo "  - Expected Duration: 5-10 minutes"
echo "  - Expected VRAM: ~18-20GB"
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

python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 100 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --no-tune_diffusion_model \
    --save-steps 100 \
    --gradient-accumulation-steps 1 \
    --warmup-ratio 0.05 \
    --report-to tensorboard \
    2>&1 | tee $OUTPUT_DIR/mini_mvp_training.log

TRAINING_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "▶ Training finished at $(date)"
echo ""

echo ""
echo "========================================================================"
echo "Step 6/6: Generate Inspection Report"
echo "========================================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs at: $OUTPUT_DIR/mini_mvp_training.log"
    exit $TRAINING_EXIT_CODE
fi

# Generate inspection report
REPORT_FILE="$OUTPUT_DIR/mini_mvp_inspection_report.txt"

cat > $REPORT_FILE << EOF
================================================================================
GR00T Mini-MVP Inspection Report
Generated: $(date)
================================================================================

DATASET INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset Path: $DATASET_PATH
Total Episodes: $TOTAL_EPISODES
Total Frames: $(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_frames'])" 2>/dev/null)
FPS: $(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['fps'])" 2>/dev/null)

Camera Keys:
$(python -c "import json; modality=json.load(open('$DATASET_PATH/meta/modality.json')); print('\n'.join([f'  - {k}' for k in modality.keys() if 'images' in k]))" 2>/dev/null)

Action Shape:
$(python -c "import json; modality=json.load(open('$DATASET_PATH/meta/modality.json')); print(f'  {modality[\"action\"][\"shape\"]}')" 2>/dev/null)

State Shape:
$(python -c "import json; modality=json.load(open('$DATASET_PATH/meta/modality.json')); print(f'  {modality[\"observation.state\"][\"shape\"]}')" 2>/dev/null)

TRAINING CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Steps: 100
Batch Size: 4
LoRA Rank: 16
Learning Rate: 1e-4
Memory Optimization: --no-tune_diffusion_model

TRAINING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exit Code: $TRAINING_EXIT_CODE
Training Log: $OUTPUT_DIR/mini_mvp_training.log

Loss Progression (from log):
$(grep -E "(Step|Loss)" $OUTPUT_DIR/mini_mvp_training.log | tail -20 || echo "  (Check log file for details)")

GPU Memory Usage:
$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "  Used: %s MB / %s MB\n", $1, $2}')

FILES GENERATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output Directory: $OUTPUT_DIR
$(ls -lh $OUTPUT_DIR/)

NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If training completed successfully:
  1. Review training log for any warnings/errors
  2. Check that loss values are reasonable (not NaN)
  3. Verify GPU memory stayed < 22GB
  4. Proceed to collect 40 more episodes for full MVP (50 total)

If training failed:
  1. Check error messages in training log
  2. Common issues:
     - modality.json mismatch with dataset
     - CUDA OOM (reduce batch size to 2)
     - Missing dependencies
  3. See troubleshooting guide in COMPLETE_LORA_GUIDE.md

================================================================================
EOF

echo ""
echo "✅ Inspection report generated: $REPORT_FILE"
echo ""
cat $REPORT_FILE

echo ""
echo "========================================================================"
echo "Mini-MVP Test Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review inspection report: $REPORT_FILE"
echo "  2. Check training log: $OUTPUT_DIR/mini_mvp_training.log"
echo "  3. If successful: Collect 40 more episodes (reach 50 for full MVP)"
echo "  4. Then run: bash scripts/train_groot_so101_mvp.sh"
echo ""
