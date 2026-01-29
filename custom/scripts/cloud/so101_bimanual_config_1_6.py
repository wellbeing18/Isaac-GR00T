#!/usr/bin/env python3
"""
Bimanual SO-101 Modality Configuration for GR00T 1.6.

This file defines how GR00T processes bimanual SO-101 robot data during
training and inference. It specifies which modalities to load, their
temporal sampling, and action representations.

Bimanual Configuration:
    - Video: head, left_wrist, right_wrist (3 cameras)
    - State: left_arm (5D), left_gripper (1D), right_arm (5D), right_gripper (1D) = 12D total
    - Action: same as state, 12D total

CRITICAL: modality_keys must exactly match keys in meta/modality.json:
    - video: "head", "left_wrist", "right_wrist"
    - state: "left_arm", "left_gripper", "right_arm", "right_gripper"
    - action: "left_arm", "left_gripper", "right_arm", "right_gripper"

Reference: custom/scripts/ver1_6/so101_config_1_6.py (single arm version)
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Action prediction horizon - number of future steps to predict
# 16 is NVIDIA's default for smooth trajectory prediction
ACTION_HORIZON = 16

# State/action dimensions for bimanual SO-101
LEFT_ARM_DIM = 5      # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
LEFT_GRIPPER_DIM = 1  # gripper position
RIGHT_ARM_DIM = 5     # same as left
RIGHT_GRIPPER_DIM = 1 # gripper position
TOTAL_DIM = 12        # 6 per arm * 2 arms
# ============================================================================

# Define the modality configuration for bimanual SO-101
so101_bimanual_config = {
    # Video modalities - 3 cameras for bimanual
    "video": ModalityConfig(
        # delta_indices=[0] means use only current frame (no temporal stacking)
        delta_indices=[0],
        # Keys must match modality.json video section
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),

    # State modalities - robot proprioception for both arms
    "state": ModalityConfig(
        # delta_indices=[0] means use only current state
        delta_indices=[0],
        # Keys must match modality.json state section
        # Order matters for concatenation!
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
    ),

    # Action modalities - what to predict for both arms
    "action": ModalityConfig(
        # Predict ACTION_HORIZON steps into the future
        # Each step at 30Hz = 16 steps = ~0.53 seconds lookahead
        delta_indices=list(range(ACTION_HORIZON)),
        # Keys must match modality.json action section
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
        # Action configuration for each modality key (must match order above)
        action_configs=[
            # Left arm: relative actions (delta from current state)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # Joint space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # Left gripper: absolute position target
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Right arm: relative actions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Right gripper: absolute position target
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Language modalities - task description
    "language": ModalityConfig(
        delta_indices=[0],
        # Key format: annotation.<source>.<type>.<name>
        # This maps to task_index in the dataset via modality.json
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Register this configuration for NEW_EMBODIMENT tag
# This allows training scripts to use --embodiment_tag NEW_EMBODIMENT
# and automatically get this configuration
register_modality_config(so101_bimanual_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# Print confirmation when this module is imported
print(f"[so101_bimanual_config_1_6] Registered bimanual SO-101 config for NEW_EMBODIMENT")
print(f"  - Video keys: {so101_bimanual_config['video'].modality_keys}")
print(f"  - State keys: {so101_bimanual_config['state'].modality_keys}")
print(f"  - Action keys: {so101_bimanual_config['action'].modality_keys}")
print(f"  - Action horizon: {ACTION_HORIZON} steps")
print(f"  - Total state/action dim: {TOTAL_DIM}")
