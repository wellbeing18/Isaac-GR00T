#!/bin/bash
# ===========================================================================
# GROOT 1.6 Bimanual Cloud Setup Script
# ===========================================================================
#
# This script handles the complete GROOT training workflow on Vast.ai:
#   1. Clone the forked Isaac-GR00T repo (with bimanual support)
#   2. Download dataset from HuggingFace Hub
#   3. Convert LeRobot v3.0 -> GROOT v2.1 format
#   4. (Optional) Start training
#
# Prerequisites:
#   - Vast.ai instance with PyTorch 2.x template
#   - At least 100GB disk space
#   - A100 40GB+ recommended for Vision+Action mode
#
# Usage:
#   # Full setup (download + convert + train)
#   bash setup_groot_bimanual.sh
#
#   # Setup only (no training)
#   SKIP_TRAINING=true bash setup_groot_bimanual.sh
#
#   # Custom dataset
#   DATASET_REPO_ID=your-username/your-dataset bash setup_groot_bimanual.sh
#
#   # Custom training params
#   MAX_STEPS=5000 TUNE_VISUAL=true bash setup_groot_bimanual.sh
#
# ===========================================================================

set -e

# ===========================================================================
# Configuration
# ===========================================================================

# --- Repository ---
GROOT_REPO="${GROOT_REPO:-https://github.com/wellbeing18/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-lora/reusable-groot-workflow-rtx5090-fixes}"
GROOT_DIR="${GROOT_DIR:-/workspace/Isaac-GR00T}"

# --- Dataset ---
DATASET_REPO_ID="${DATASET_REPO_ID:-jasmine314342/picknplace-bimanual-464}"
DATASET_DOWNLOAD_DIR="${DATASET_DOWNLOAD_DIR:-/workspace/datasets_lerobot}"
DATASET_GROOT_DIR="${DATASET_GROOT_DIR:-/workspace/Isaac-GR00T/datasets/bimanual_groot}"

# --- Training ---
SKIP_TRAINING="${SKIP_TRAINING:-false}"
MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
TUNE_VISUAL="${TUNE_VISUAL:-false}"
TUNE_LLM="${TUNE_LLM:-false}"

# ===========================================================================
# Helper Functions
# ===========================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ -z "$available_gb" ]; then
        available_gb=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        log "ERROR: Need ${required_gb}GB disk space, only ${available_gb}GB available"
        exit 1
    fi
    log "Disk space OK: ${available_gb}GB available (need ${required_gb}GB)"
}

# ===========================================================================
# Step 0: Check Prerequisites
# ===========================================================================

echo "=================================================================="
echo "GROOT 1.6 Bimanual Cloud Setup"
echo "=================================================================="
echo "Repository:     $GROOT_REPO"
echo "Branch:         $GROOT_BRANCH"
echo "Dataset:        $DATASET_REPO_ID"
echo "Output:         $DATASET_GROOT_DIR"
echo "Skip Training:  $SKIP_TRAINING"
echo "=================================================================="

# Check disk space (need ~80GB: 30GB dataset + 20GB converted + 30GB models)
check_disk_space 80

# Check Python/pip
if ! command -v python &> /dev/null; then
    log "ERROR: Python not found. Use a PyTorch template on Vast.ai."
    exit 1
fi

# ===========================================================================
# Step 1: Clone Isaac-GR00T Fork
# ===========================================================================

log "Step 1: Cloning Isaac-GR00T repository..."

if [ -d "$GROOT_DIR" ]; then
    log "Isaac-GR00T already exists at $GROOT_DIR"
    cd "$GROOT_DIR"
    git fetch origin
    git checkout "$GROOT_BRANCH"
    git pull origin "$GROOT_BRANCH" || true
else
    git clone --branch "$GROOT_BRANCH" "$GROOT_REPO" "$GROOT_DIR"
    cd "$GROOT_DIR"
fi

log "Installing Isaac-GR00T dependencies..."
pip install -e . --quiet
pip install jsonlines pyav --quiet

log "Isaac-GR00T setup complete!"

# ===========================================================================
# Step 2: Download Dataset from HuggingFace
# ===========================================================================

log "Step 2: Downloading dataset from HuggingFace..."

mkdir -p "$DATASET_DOWNLOAD_DIR"

# Extract dataset name from repo ID (e.g., "jasmine314342/picknplace-bimanual-464" -> "picknplace-bimanual-464")
DATASET_NAME=$(echo "$DATASET_REPO_ID" | cut -d'/' -f2)
DATASET_LOCAL_PATH="$DATASET_DOWNLOAD_DIR/$DATASET_NAME"

if [ -d "$DATASET_LOCAL_PATH/meta" ]; then
    log "Dataset already exists at $DATASET_LOCAL_PATH"
else
    log "Downloading $DATASET_REPO_ID to $DATASET_LOCAL_PATH..."

    # Use huggingface-cli to download
    pip install huggingface_hub --quiet

    python -c "
from huggingface_hub import snapshot_download
import os

repo_id = '$DATASET_REPO_ID'
local_dir = '$DATASET_LOCAL_PATH'

print(f'Downloading {repo_id} to {local_dir}...')
snapshot_download(
    repo_id=repo_id,
    repo_type='dataset',
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print('Download complete!')
"
fi

# Verify download
if [ ! -f "$DATASET_LOCAL_PATH/meta/info.json" ]; then
    log "ERROR: Dataset download failed - meta/info.json not found"
    exit 1
fi

log "Dataset downloaded successfully!"
log "  Episodes: $(cat $DATASET_LOCAL_PATH/meta/info.json | grep total_episodes | head -1)"

# ===========================================================================
# Step 3: Convert to GROOT Format
# ===========================================================================

log "Step 3: Converting dataset to GROOT format..."

cd "$GROOT_DIR"

if [ -d "$DATASET_GROOT_DIR/meta" ]; then
    log "Converted dataset already exists at $DATASET_GROOT_DIR"
else
    python custom/scripts/cloud/convert_bimanual_to_groot.py \
        --input "$DATASET_LOCAL_PATH" \
        --output "$DATASET_GROOT_DIR" \
        --force
fi

# Verify conversion
if [ ! -f "$DATASET_GROOT_DIR/meta/modality.json" ]; then
    log "ERROR: Conversion failed - modality.json not found"
    exit 1
fi

log "Dataset conversion complete!"
log "  GROOT dataset: $DATASET_GROOT_DIR"

# ===========================================================================
# Step 4: Training (Optional)
# ===========================================================================

if [ "$SKIP_TRAINING" = "true" ]; then
    log "Skipping training (SKIP_TRAINING=true)"
    echo ""
    echo "=================================================================="
    echo "Setup Complete!"
    echo "=================================================================="
    echo ""
    echo "To start training manually:"
    echo "  cd $GROOT_DIR"
    echo "  DATASET_PATH=$DATASET_GROOT_DIR bash custom/scripts/cloud/train_groot_bimanual.sh"
    echo ""
    echo "Training options:"
    echo "  # Default mode (24GB GPU)"
    echo "  DATASET_PATH=$DATASET_GROOT_DIR \\"
    echo "      bash custom/scripts/cloud/train_groot_bimanual.sh"
    echo ""
    echo "  # Vision + Action mode (40GB+ GPU, recommended for bimanual)"
    echo "  DATASET_PATH=$DATASET_GROOT_DIR \\"
    echo "      TUNE_VISUAL=true GLOBAL_BATCH_SIZE=16 \\"
    echo "      bash custom/scripts/cloud/train_groot_bimanual.sh"
    echo ""
    exit 0
fi

log "Step 4: Starting training..."

# Build training command
export DATASET_PATH="$DATASET_GROOT_DIR"
export MAX_STEPS="$MAX_STEPS"
export GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
export TUNE_VISUAL="$TUNE_VISUAL"
export TUNE_LLM="$TUNE_LLM"

cd "$GROOT_DIR"
bash custom/scripts/cloud/train_groot_bimanual.sh

echo ""
echo "=================================================================="
echo "Training Complete!"
echo "=================================================================="
echo ""
echo "Checkpoints saved to: $GROOT_DIR/outputs/groot16_bimanual_*"
echo ""
echo "To download checkpoints to local machine:"
echo "  # From local machine:"
echo "  scp -r root@<instance-ip>:$GROOT_DIR/outputs/groot16_bimanual_*/checkpoint-* ./"
echo ""
