#!/usr/bin/env python3
"""
Scientific Experiment: Does the model use head camera for navigation?

Hypothesis: The model exhibits "head camera blindness" - it doesn't use head
camera information to guide the arm back when wrist camera sees empty space.

Experiment Design:
1. Load trace data from step 800 (wrist=empty, head=blocks visible)
2. Run inference with:
   a) Original images (head=blocks, wrist=empty)
   b) Ablated: head=BLACK, wrist=empty
3. Compare predictions:
   - If predictions are SAME: Head camera is NOT being used
   - If predictions are DIFFERENT: Head camera IS being used but model
     can't translate it into correct recovery actions

This will give us facts-based evidence about whether head camera blindness
is a configuration issue or a model behavior issue.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"


def import_modality_config(config_path: str):
    """Import modality config to register NEW_EMBODIMENT."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Modality config not found: {full_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_policy(checkpoint_path: str):
    """Load GR00T policy."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=checkpoint_path,
        device=DEVICE,
    )
    return policy


def format_observation(images: dict, state: np.ndarray, task: str) -> dict:
    """Format observation for GR00T policy input."""
    ARM_DIM = 5
    GRIPPER_DIM = 1

    observation = {"video": {}, "state": {}, "language": {}}

    for name, frame in images.items():
        # Ensure frame is RGB uint8, shape (H, W, 3)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        observation["video"][name] = frame[np.newaxis, np.newaxis, :, :, :]

    arm_state = state[:ARM_DIM].astype(np.float32)
    gripper_state = state[ARM_DIM:ARM_DIM + GRIPPER_DIM].astype(np.float32)
    observation["state"]["single_arm"] = arm_state[np.newaxis, np.newaxis, :]
    observation["state"]["gripper"] = gripper_state[np.newaxis, np.newaxis, :]
    observation["language"]["annotation.human.action.task_description"] = [[task]]

    return observation


def run_inference(policy, images: dict, state: np.ndarray, task: str):
    """Run inference and return predicted actions."""
    observation = format_observation(images, state, task)
    action_dict, info = policy.get_action(observation)

    # Extract actions
    arm_actions = action_dict["single_arm"][0]  # Shape: (16, 5)
    gripper_actions = action_dict["gripper"][0]  # Shape: (16, 1)
    actions = np.concatenate([arm_actions, gripper_actions], axis=1)  # Shape: (16, 6)

    return actions


def main():
    parser = argparse.ArgumentParser(description="Test if model uses head camera")
    parser.add_argument("--trace-dir", type=str, required=True,
                        help="Path to trace directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--step", type=int, default=800,
                        help="Step to analyze (default: 800)")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)

    # Load trace data
    print("=" * 70)
    print("HEAD CAMERA USAGE EXPERIMENT")
    print("=" * 70)

    # Load config
    config_file = trace_dir / "config.json"
    with open(config_file) as f:
        config = json.load(f)
    task = config["task"]
    print(f"Task: {task}")
    print(f"Analyzing step: {args.step}")

    # Load trace entry for the specified step
    trace_file = trace_dir / "trace.jsonl"
    target_entry = None
    with open(trace_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry["step"] == args.step and entry.get("inference_triggered"):
                target_entry = entry
                break

    if target_entry is None:
        # Find nearest inference step
        with open(trace_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("inference_triggered") and entry["step"] <= args.step:
                    target_entry = entry
        print(f"Step {args.step} not an inference step, using nearest: {target_entry['step']}")

    step = target_entry["step"]
    state = np.array(target_entry["joint_states"], dtype=np.float32)

    print(f"\nStep {step} state: {state}")

    # Load images
    head_img_path = trace_dir / "images" / f"step_{step:04d}_head.jpg"
    wrist_img_path = trace_dir / "images" / f"step_{step:04d}_wrist.jpg"

    if not head_img_path.exists() or not wrist_img_path.exists():
        print(f"ERROR: Images not found for step {step}")
        sys.exit(1)

    head_img = cv2.cvtColor(cv2.imread(str(head_img_path)), cv2.COLOR_BGR2RGB)
    wrist_img = cv2.cvtColor(cv2.imread(str(wrist_img_path)), cv2.COLOR_BGR2RGB)

    print(f"\nHead image shape: {head_img.shape}")
    print(f"Wrist image shape: {wrist_img.shape}")

    # Load model
    print("\nLoading model...")
    import_modality_config(MODALITY_CONFIG_PATH)
    policy = load_policy(args.checkpoint)
    print("Model loaded.")

    # Create ablated images
    black_head = np.zeros_like(head_img)  # All black
    black_wrist = np.zeros_like(wrist_img)  # All black

    # Experiment 1: Original (head=blocks, wrist=empty)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Original images (head=blocks, wrist=empty)")
    print("=" * 70)
    images_original = {"head": head_img, "wrist": wrist_img}
    actions_original = run_inference(policy, images_original, state, task)

    print(f"Action[0]: {np.round(actions_original[0], 2)}")
    print(f"Action[3]: {np.round(actions_original[3], 2)}")
    delta_original = actions_original[0] - state
    print(f"Delta[0]:  {np.round(delta_original, 2)}")

    # Experiment 2: Black head (head=BLACK, wrist=empty)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ablated head (head=BLACK, wrist=empty)")
    print("=" * 70)
    images_black_head = {"head": black_head, "wrist": wrist_img}
    actions_black_head = run_inference(policy, images_black_head, state, task)

    print(f"Action[0]: {np.round(actions_black_head[0], 2)}")
    print(f"Action[3]: {np.round(actions_black_head[3], 2)}")
    delta_black_head = actions_black_head[0] - state
    print(f"Delta[0]:  {np.round(delta_black_head, 2)}")

    # Experiment 3: Black wrist (head=blocks, wrist=BLACK)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Ablated wrist (head=blocks, wrist=BLACK)")
    print("=" * 70)
    images_black_wrist = {"head": head_img, "wrist": black_wrist}
    actions_black_wrist = run_inference(policy, images_black_wrist, state, task)

    print(f"Action[0]: {np.round(actions_black_wrist[0], 2)}")
    print(f"Action[3]: {np.round(actions_black_wrist[3], 2)}")
    delta_black_wrist = actions_black_wrist[0] - state
    print(f"Delta[0]:  {np.round(delta_black_wrist, 2)}")

    # Experiment 4: Both black
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Both cameras BLACK")
    print("=" * 70)
    images_both_black = {"head": black_head, "wrist": black_wrist}
    actions_both_black = run_inference(policy, images_both_black, state, task)

    print(f"Action[0]: {np.round(actions_both_black[0], 2)}")
    print(f"Action[3]: {np.round(actions_both_black[3], 2)}")
    delta_both_black = actions_both_black[0] - state
    print(f"Delta[0]:  {np.round(delta_both_black, 2)}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Does the model use head camera?")
    print("=" * 70)

    # Compare original vs black head
    diff_head_ablation = np.abs(actions_original - actions_black_head).mean()
    diff_wrist_ablation = np.abs(actions_original - actions_black_wrist).mean()
    diff_both_ablation = np.abs(actions_original - actions_both_black).mean()

    print(f"\nMean absolute difference from original:")
    print(f"  Head ablated (black):  {diff_head_ablation:.4f}")
    print(f"  Wrist ablated (black): {diff_wrist_ablation:.4f}")
    print(f"  Both ablated (black):  {diff_both_ablation:.4f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    threshold = 0.5  # Degrees

    if diff_head_ablation < threshold:
        print(f"✗ Head camera ablation caused MINIMAL change ({diff_head_ablation:.4f}°)")
        print("  → Model does NOT effectively use head camera for action prediction!")
        head_used = False
    else:
        print(f"✓ Head camera ablation caused SIGNIFICANT change ({diff_head_ablation:.4f}°)")
        print("  → Model DOES use head camera for action prediction")
        head_used = True

    if diff_wrist_ablation < threshold:
        print(f"✗ Wrist camera ablation caused MINIMAL change ({diff_wrist_ablation:.4f}°)")
        print("  → Model does NOT effectively use wrist camera!")
        wrist_used = False
    else:
        print(f"✓ Wrist camera ablation caused SIGNIFICANT change ({diff_wrist_ablation:.4f}°)")
        print("  → Model DOES use wrist camera for action prediction")
        wrist_used = True

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)

    if not head_used and wrist_used:
        print("HEAD CAMERA BLINDNESS CONFIRMED!")
        print("The model relies primarily on wrist camera and ignores head camera.")
        print("This explains why the arm doesn't recover when wrist sees empty space")
        print("even though head camera clearly shows the blocks.")
    elif head_used and wrist_used:
        print("Both cameras are used, but model still fails to recover.")
        print("Issue may be in how the model translates head camera info to actions.")
    elif not head_used and not wrist_used:
        print("Neither camera significantly affects predictions!")
        print("Model may be relying primarily on state/language.")
    else:
        print("Wrist camera not used but head camera is used - unexpected pattern.")

    # ShPan analysis (first joint - most relevant for "swinging" behavior)
    print("\n" + "-" * 70)
    print("SHOULDER_PAN (ShPan) ANALYSIS:")
    print("-" * 70)

    print(f"Current ShPan: {state[0]:.1f}°")
    print(f"Predicted ShPan delta (original):      {delta_original[0]:+.2f}°")
    print(f"Predicted ShPan delta (black head):    {delta_black_head[0]:+.2f}°")
    print(f"Predicted ShPan delta (black wrist):   {delta_black_wrist[0]:+.2f}°")
    print(f"Predicted ShPan delta (both black):    {delta_both_black[0]:+.2f}°")

    # If ShPan delta is negative, model is moving LEFT (away from blocks)
    # If ShPan delta is positive, model is moving RIGHT (toward blocks)
    blocks_direction = "RIGHT (positive ShPan delta)"
    if delta_original[0] < -0.5:
        print(f"\n⚠ Model predicts moving LEFT (ShPan delta: {delta_original[0]:+.2f}°)")
        print(f"  But blocks are visible in head camera - model should move RIGHT!")


if __name__ == "__main__":
    main()
