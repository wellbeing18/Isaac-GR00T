#!/usr/bin/env python3
"""
SO-101 Modality Configuration for GR00T 1.6.

This file defines how GR00T processes SO-101 robot data during training and inference.
It specifies which modalities to load, their temporal sampling, and action representations.

CRITICAL: modality_keys must exactly match keys in meta/modality.json:
  - video: "head", "wrist" (maps to observation.images.head, observation.images.left_wrist)
  - state: "single_arm", "gripper" (indices 0-5, 5-6 in concatenated state array)
  - action: "single_arm", "gripper" (indices 0-5, 5-6 in concatenated action array)

Decision Notes:
- delta_indices=[0] for video/state: Use current frame only (no temporal context)
- delta_indices=list(range(16)) for action: 16-step prediction horizon
- ActionRepresentation.RELATIVE for arm: Predicts delta from current state
  (Better generalization, but requires reference state for denormalization)
- ActionRepresentation.ABSOLUTE for gripper: Direct position target
  (Gripper is binary-ish, absolute works well)
- ActionType.NON_EEF: Joint space control (not end-effector Cartesian control)
- ActionFormat.DEFAULT: No special rotation encoding

Reference: examples/SO100/so100_config.py
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
# Higher values = smoother but slower adaptation to changes
# Lower values = more reactive but potentially jerky
ACTION_HORIZON = 16

# State/action dimensions for SO-101
# Must match modality.json: single_arm (0-5), gripper (5-6)
ARM_DIM = 5   # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DIM = 1  # gripper position
# ============================================================================

# Define the modality configuration for SO-101
so101_config = {
    # Video modalities - which cameras to use
    "video": ModalityConfig(
        # delta_indices=[0] means use only current frame (no temporal stacking)
        # Could use [-1, 0] for current + previous frame if needed
        delta_indices=[0],
        # Keys must match modality.json video section
        modality_keys=["head", "wrist"],
    ),

    # State modalities - robot proprioception
    "state": ModalityConfig(
        # delta_indices=[0] means use only current state
        delta_indices=[0],
        # Keys must match modality.json state section
        modality_keys=["single_arm", "gripper"],
        # Optional: sin_cos_embedding_keys for periodic state (angles)
        # This doubles the dimension but can help with angle wraparound
        # sin_cos_embedding_keys=["single_arm"],  # Uncomment if needed
    ),

    # Action modalities - what to predict
    "action": ModalityConfig(
        # Predict ACTION_HORIZON steps into the future
        # Each step at 30Hz = 16 steps = ~0.53 seconds lookahead
        delta_indices=list(range(ACTION_HORIZON)),
        # Keys must match modality.json action section
        modality_keys=["single_arm", "gripper"],
        # Action configuration for each modality key (must match order above)
        action_configs=[
            # Arm: relative actions (delta from current state)
            # Relative actions are more generalizable but require:
            #   1. Reference state during inference for denormalization
            #   2. relative_stats.json in dataset meta folder
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # Joint space, not end-effector
                format=ActionFormat.DEFAULT,  # No special rotation encoding
            ),
            # Gripper: absolute position target
            # Gripper is typically binary (open/close) so absolute works well
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
register_modality_config(so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# Print confirmation when this module is imported
print(f"[so101_config_1_6] Registered SO-101 config for NEW_EMBODIMENT")
print(f"  - Video keys: {so101_config['video'].modality_keys}")
print(f"  - State keys: {so101_config['state'].modality_keys}")
print(f"  - Action keys: {so101_config['action'].modality_keys}")
print(f"  - Action horizon: {ACTION_HORIZON} steps")
