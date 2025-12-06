#!/usr/bin/env python3
"""
Quick diagnostic script to test GR00T inference on training data samples.

This helps verify if the inference pipeline is working correctly by:
1. Loading a sample from the training dataset
2. Running inference with the same observation
3. Comparing predicted action vs ground truth action

If this works but real robot doesn't, the issue is likely:
- Camera setup difference
- Robot state calibration difference
- Task description mismatch

Usage:
    python diagnose_inference.py --checkpoint /path/to/checkpoint
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")
warnings.filterwarnings("ignore", message=".*albumentations.*")

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Diagnose GR00T inference")
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--dataset", "-d", type=str,
                       default="/home/jrobot/project/XLeRobot/datasets_groot")
    parser.add_argument("--sample-idx", type=int, default=100,
                       help="Sample index to test")
    parser.add_argument("--data-config", type=str, default="so100_dualcam")
    args = parser.parse_args()

    print("=" * 70)
    print("GR00T Inference Diagnostic")
    print("=" * 70)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config

    data_cfg = load_data_config(args.data_config)
    modality_config = data_cfg.modality_config()

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=modality_config,
        video_backend="torchvision_av",
        transforms=None,
        embodiment_tag="new_embodiment",
    )
    print(f"  Dataset: {len(dataset)} samples, {len(dataset.trajectory_lengths)} trajectories")

    # Load model
    print("\n[2/4] Loading model...")
    sys.path.insert(0, str(Path(__file__).parent))
    from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora

    modality_transform = data_cfg.transform()

    if is_lora_checkpoint(args.checkpoint):
        policy = load_groot_with_lora(
            model_path=args.checkpoint,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
            merge_weights=True,
        )
    else:
        from gr00t.model.policy import Gr00tPolicy
        policy = Gr00tPolicy(
            model_path=args.checkpoint,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
        )

    # Get sample
    print(f"\n[3/4] Loading sample {args.sample_idx}...")
    obs = dataset[args.sample_idx]

    print("\n  Sample contents:")
    for key, val in obs.items():
        if hasattr(val, 'shape'):
            print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    {key}: {type(val).__name__} = {val}")

    # Extract ground truth
    gt_arm = obs["action.single_arm"][0]  # First action in horizon
    gt_gripper = obs["action.gripper"][0]
    if isinstance(gt_arm, torch.Tensor):
        gt_arm = gt_arm.numpy()
    if isinstance(gt_gripper, torch.Tensor):
        gt_gripper = gt_gripper.numpy()
    gt_action = np.concatenate([gt_arm, gt_gripper])

    state_arm = obs["state.single_arm"][0]
    state_gripper = obs["state.gripper"][0]
    if isinstance(state_arm, torch.Tensor):
        state_arm = state_arm.numpy()
    if isinstance(state_gripper, torch.Tensor):
        state_gripper = state_gripper.numpy()
    state = np.concatenate([state_arm, state_gripper])

    task = obs.get("annotation.human.task_description", ["unknown"])
    if isinstance(task, list):
        task = task[0]

    print(f"\n  Task: '{task}'")
    print(f"  State: {np.round(state, 2)}")
    print(f"  GT Action: {np.round(gt_action, 2)}")

    # Run inference
    print("\n[4/4] Running inference...")

    # Build obs dict for policy
    obs_dict = {
        "video.front": obs["video.front"],
        "video.wrist": obs["video.wrist"],
        "state.single_arm": obs["state.single_arm"].astype(np.float64) if isinstance(obs["state.single_arm"], np.ndarray) else obs["state.single_arm"].numpy().astype(np.float64),
        "state.gripper": obs["state.gripper"].astype(np.float64) if isinstance(obs["state.gripper"], np.ndarray) else obs["state.gripper"].numpy().astype(np.float64),
        "annotation.human.task_description": [task],
    }

    # Ensure proper shapes
    for key in ["video.front", "video.wrist"]:
        if isinstance(obs_dict[key], torch.Tensor):
            obs_dict[key] = obs_dict[key].numpy()
    for key in ["state.single_arm", "state.gripper"]:
        if len(obs_dict[key].shape) == 1:
            obs_dict[key] = obs_dict[key][np.newaxis, :]

    with torch.no_grad():
        action_dict = policy.get_action(obs_dict)

    pred_arm = action_dict["action.single_arm"][0]
    pred_gripper = action_dict["action.gripper"][0]
    pred_action = np.concatenate([pred_arm, pred_gripper])

    print(f"  Predicted Action: {np.round(pred_action, 2)}")

    # Compare
    error = np.abs(pred_action - gt_action)
    print(f"\n  Absolute Error: {np.round(error, 2)}")
    print(f"  Mean Absolute Error: {np.mean(error):.2f}°")

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    print("\n  Per-joint comparison:")
    print(f"  {'Joint':<15} {'State':>8} {'GT':>8} {'Pred':>8} {'Error':>8}")
    print("  " + "-" * 55)
    for i, name in enumerate(joint_names):
        print(f"  {name:<15} {state[i]:>8.2f} {gt_action[i]:>8.2f} {pred_action[i]:>8.2f} {error[i]:>8.2f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    mae = np.mean(error)
    if mae < 5:
        print(f"  ✓ MAE {mae:.2f}° is GOOD - inference pipeline working correctly")
        print("  If real robot still fails, check:")
        print("    1. Camera indices (--head-cam-idx, --wrist-cam-idx)")
        print("    2. Robot calibration matches training")
        print("    3. Task description matches exactly")
    elif mae < 15:
        print(f"  ~ MAE {mae:.2f}° is MODERATE - might work but not ideal")
    else:
        print(f"  ✗ MAE {mae:.2f}° is HIGH - inference pipeline may have issues")
        print("  Check:")
        print("    1. Checkpoint loaded correctly")
        print("    2. Normalization metadata exists")
        print("    3. Data config matches training")

    print("=" * 70)


if __name__ == "__main__":
    main()
