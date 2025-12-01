#!/bin/bash
################################################################################
# GR00T SO101 Inference Wrapper Script
# Purpose: Easy execution of finetuned GR00T model on SO101 arm
#
# Usage:
#   ./infer_groot_so101.sh                           # Run with defaults
#   ./infer_groot_so101.sh --task "grasp object"    # Specify task
#   ./infer_groot_so101.sh --actions-to-execute 50  # Limit action chunks
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
MODEL_PATH="${MODEL_PATH:-/home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test}"
DATA_CONFIG="${DATA_CONFIG:-so100_dualcam}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
TASK="${TASK:-grasp object}"
PORT="${PORT:-/dev/ttyACM0}"
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

# Check model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: Model checkpoint not found at: $MODEL_PATH"
    echo "Please verify the model path or set MODEL_PATH environment variable."
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
