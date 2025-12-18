#!/bin/bash
################################################################################
# GR00T SO101 Inference Wrapper Script
# Purpose: Easy execution of finetuned GR00T model on SO101 arm
#
# FPS Configuration (per 6_fps_upgrade_30hz.md):
#   - Action frequency: 30 Hz (synchronized with video)
#   - Camera FPS: 30 fps
#   - Action interval: 0.033s (1/30 Hz)
#
# Usage:
#   ./infer_groot_so101.sh                           # Run with defaults (no display)
#   ./infer_groot_so101.sh --task "grasp object"    # Specify task
#   ./infer_groot_so101.sh --actions-to-execute 50  # Limit action chunks
#   ./infer_groot_so101.sh --display                # Enable camera display
#   ./infer_groot_so101.sh --help                   # Show all options
################################################################################

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_GROOT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "========================================================================"
echo "GR00T SO101 Inference - All-in-one"
echo "========================================================================"
echo ""

# Default configuration
# MODEL_PATH must be set by user or via environment variable
MODEL_PATH="${MODEL_PATH:-}"
DATA_CONFIG="${DATA_CONFIG:-so100_dualcam}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
# Task must match training task descriptions
TASK="${TASK:-pick the red cube from the table}"
PORT="${PORT:-/dev/ttyACM2}"  # Left arm port
HEAD_CAM_IDX="${HEAD_CAM_IDX:-4}"
WRIST_CAM_IDX="${WRIST_CAM_IDX:-6}"
ACTION_HORIZON="${ACTION_HORIZON:-12}"
ACTIONS_TO_EXECUTE="${ACTIONS_TO_EXECUTE:-100}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"

# Activate groot environment
echo "Activating groot conda environment..."
eval "$(conda shell.bash hook)"
conda activate groot

# Verify environment
echo "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check if lerobot is available
if ! python -c "import lerobot" 2>/dev/null; then
    echo ""
    echo "ERROR: LeRobot not found in groot environment!"
    echo ""
    echo "Please install LeRobot first:"
    echo "  conda activate groot"
    echo "  conda install -c conda-forge evdev  # Install evdev first (kernel 6.14 fix)"
    echo "  pip install -e /home/jrobot/project/lerobot"
    echo ""
    exit 1
fi

echo "LeRobot: OK"

# Check model path is provided and exists
if [ -z "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: MODEL_PATH not set!"
    echo ""
    echo "Usage:"
    echo "  MODEL_PATH=/path/to/checkpoint ./infer_groot_so101.sh"
    echo ""
    echo "Example:"
    echo "  MODEL_PATH=/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_XXX/best ./infer_groot_so101.sh"
    echo ""
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: Model checkpoint not found at: $MODEL_PATH"
    echo "Please verify the model path."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model path:        $MODEL_PATH"
echo "  Data config:       $DATA_CONFIG"
echo "  Embodiment tag:    $EMBODIMENT_TAG"
echo "  Task:              $TASK"
echo "  Robot port:        $PORT"
echo "  Head camera idx:   $HEAD_CAM_IDX"
echo "  Wrist camera idx:  $WRIST_CAM_IDX"
echo "  Action horizon:    $ACTION_HORIZON"
echo "  Actions to exec:   $ACTIONS_TO_EXECUTE"
echo "  Denoising steps:   $DENOISING_STEPS"
echo ""

# Change to Isaac-GR00T directory
cd "$ISAAC_GROOT_ROOT"

# Run inference
echo "Starting inference..."
echo "========================================================================"
echo ""

python custom/scripts/infer_groot_so101.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --task "$TASK" \
    --port "$PORT" \
    --head-cam-idx "$HEAD_CAM_IDX" \
    --wrist-cam-idx "$WRIST_CAM_IDX" \
    --action-horizon "$ACTION_HORIZON" \
    --actions-to-execute "$ACTIONS_TO_EXECUTE" \
    --denoising-steps "$DENOISING_STEPS" \
    "$@"
