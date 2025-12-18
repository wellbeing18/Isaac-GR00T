#!/bin/bash
#===============================================================================
# Convert Multiple LeRobot v3 Datasets to GR00T Format
#
# This script runs convert_lerobot_v3_to_groot.py on each task dataset.
#
# Usage:
#   bash custom/scripts/convert_multitask_to_groot.sh
#   bash custom/scripts/convert_multitask_to_groot.sh --validate-only
#
# Prerequisites:
#   - Activate groot conda environment before running
#   - Datasets collected with LeRobot v3 format
#
# Author: Claude
# Date: 2025-12-04
#===============================================================================

set -e  # Exit on error

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_GROOT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
# IMPORTANT: Use the working copy, not the original dataset
# Original: /home/jrobot/project/XLeRobot/datasets (DO NOT MODIFY)
# Working copy: /home/jrobot/project/XLeRobot/datasets_copy (MODIFY THIS)
DATASETS_BASE="/home/jrobot/project/XLeRobot/datasets_copy"

# Conversion script
CONVERT_SCRIPT="${ISAAC_GROOT_ROOT}/custom/scripts/convert_lerobot_v3_to_groot.py"

# Robot configuration
ROBOT_TYPE="so101"
ARM="left"

# Tasks to convert (in order)
TASKS=("pick" "place" "push" "reach" "grasp" "release")

# Pass through arguments (e.g., --validate-only)
EXTRA_ARGS="$@"

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

print_step() {
    echo ""
    echo "------------------------------------------------------------------------"
    echo " Step $1: $2"
    echo "------------------------------------------------------------------------"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

print_header "Multi-Task LeRobot v3 to GR00T Conversion"

echo "Configuration:"
echo "  Datasets base: ${DATASETS_BASE}"
echo "  Arm: ${ARM}"
echo "  Robot type: ${ROBOT_TYPE}"
echo "  Tasks: ${TASKS[*]}"
echo "  Extra args: ${EXTRA_ARGS:-none}"

# Check conversion script exists
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "ERROR: Conversion script not found: $CONVERT_SCRIPT"
    exit 1
fi

# Track results
SUCCESSFUL=()
FAILED=()
SKIPPED=()

# Convert each task
STEP=1
TOTAL=${#TASKS[@]}

for TASK in "${TASKS[@]}"; do
    DATASET_PATH="${DATASETS_BASE}/${ARM}/${TASK}"

    print_step "${STEP}/${TOTAL}" "Converting '${TASK}' dataset"

    # Check if dataset exists
    if [[ ! -d "$DATASET_PATH" ]]; then
        echo "  SKIPPED: Dataset not found at ${DATASET_PATH}"
        SKIPPED+=("$TASK")
        ((STEP++))
        continue
    fi

    # Check if meta/info.json exists (valid LeRobot dataset)
    if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
        echo "  SKIPPED: No meta/info.json found (not a valid dataset)"
        SKIPPED+=("$TASK")
        ((STEP++))
        continue
    fi

    echo "  Dataset: ${DATASET_PATH}"

    # Run conversion
    if python "$CONVERT_SCRIPT" \
        --dataset-path "$DATASET_PATH" \
        --robot-type "$ROBOT_TYPE" \
        --dual-camera \
        $EXTRA_ARGS; then
        SUCCESSFUL+=("$TASK")
        echo "  SUCCESS: ${TASK} converted"
    else
        FAILED+=("$TASK")
        echo "  FAILED: ${TASK} conversion failed"
    fi

    ((STEP++))
done

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------

print_header "Conversion Summary"

echo "Successful (${#SUCCESSFUL[@]}/${TOTAL}):"
for task in "${SUCCESSFUL[@]}"; do
    echo "  - $task"
done

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo "Skipped (${#SKIPPED[@]}):"
    for task in "${SKIPPED[@]}"; do
        echo "  - $task"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Failed (${#FAILED[@]}):"
    for task in "${FAILED[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Review the errors above and re-run for failed tasks."
    exit 1
fi

echo ""
echo "All datasets converted successfully!"
echo ""
echo "Next steps:"
echo "  1. Combine datasets for multi-task training:"
echo "     python custom/scripts/combine_groot_datasets.py \\"
echo "         --input-dirs ${DATASETS_BASE}/${ARM}/* \\"
echo "         --output-dir ${DATASETS_BASE}/${ARM}_combined"
echo ""
echo "  2. Or train on individual tasks:"
echo "     bash custom/scripts/train_groot_mvp.sh"
echo ""
