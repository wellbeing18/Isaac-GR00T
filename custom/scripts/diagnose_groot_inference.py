#!/usr/bin/env python3
"""
GR00T Inference Diagnosis Script

Diagnose GR00T model behavior during inference:
1. Test model responsiveness to input changes
2. Test task conditioning
3. Log detailed input/output for debugging

Similar to Pi0.5's diagnose_model_behavior.py

Usage:
    python diagnose_groot_inference.py --checkpoint /path/to/checkpoint --dataset /path/to/datasets_groot

    # More detailed logging
    python diagnose_groot_inference.py --checkpoint /path/to/checkpoint --dataset /path/to/datasets_groot --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

import numpy as np
import torch
from tqdm import tqdm


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_sample_frames(dataset_path: Path, num_samples: int = 5) -> List[Dict]:
    """Load sample frames from the dataset."""
    import pyarrow.parquet as pq
    import decord
    decord.bridge.set_bridge("native")

    dataset_path = Path(dataset_path)
    samples = []

    # Load episodes
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    # Load tasks
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = {}
    with open(tasks_path, "r") as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks[task["task_index"]] = task["task"]

    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    # Get samples from different episodes
    sample_eps = episodes[:min(num_samples, len(episodes))]

    for ep in sample_eps:
        ep_idx = ep["episode_index"]
        ep_length = ep["length"]
        task_idx = ep.get("task_index", 0)
        task_desc = tasks.get(task_idx, "unknown task")

        # Get a frame from middle of episode
        frame_idx = ep_length // 2

        try:
            # Load parquet
            parquet_file = data_dir / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            row = df.iloc[frame_idx]

            # Load state
            state = np.array(row["observation.state"], dtype=np.float32)

            # Load videos
            front_video_path = videos_dir / f"observation.images.front_episode_{ep_idx:06d}.mp4"
            wrist_video_path = videos_dir / f"observation.images.wrist_episode_{ep_idx:06d}.mp4"

            if not front_video_path.exists():
                front_video_path = videos_dir / f"front_episode_{ep_idx:06d}.mp4"
            if not wrist_video_path.exists():
                wrist_video_path = videos_dir / f"wrist_episode_{ep_idx:06d}.mp4"

            front_vr = decord.VideoReader(str(front_video_path))
            wrist_vr = decord.VideoReader(str(wrist_video_path))

            front_frame = front_vr[frame_idx].asnumpy()
            wrist_frame = wrist_vr[frame_idx].asnumpy()

            samples.append({
                "video.front": front_frame,
                "video.wrist": wrist_frame,
                "state": state,
                "task": task_desc,
                "episode": ep_idx,
                "frame": frame_idx,
            })

        except Exception as e:
            print(f"Warning: Could not load episode {ep_idx}: {e}")
            continue

    return samples


def load_model_for_diagnosis(checkpoint_path: Path, data_config: str = "so100_dualcam"):
    """Load model with detailed logging."""
    # Add custom scripts to path
    custom_scripts_path = Path(__file__).parent
    sys.path.insert(0, str(custom_scripts_path))

    from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'='*70}")
    print(f"Loading model from: {checkpoint_path}")
    print(f"{'='*70}")

    if is_lora_checkpoint(str(checkpoint_path)):
        print("Checkpoint type: LoRA (PEFT adapter)")
        policy = load_groot_with_lora(
            model_path=str(checkpoint_path),
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
            merge_weights=True,
        )
    else:
        print("Checkpoint type: Full checkpoint")
        policy = Gr00tPolicy(
            model_path=str(checkpoint_path),
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
        )

    return policy


def run_inference_with_logging(
    policy,
    front_img: np.ndarray,
    wrist_img: np.ndarray,
    state: np.ndarray,
    task: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Run inference with detailed logging."""
    # Build observation
    obs_dict = {
        "video.front": front_img[np.newaxis, :, :, :],
        "video.wrist": wrist_img[np.newaxis, :, :, :],
        "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
        "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
        "annotation.human.task_description": [task],
    }

    log_info = {
        "input": {
            "front_img_shape": front_img.shape,
            "front_img_range": [float(front_img.min()), float(front_img.max())],
            "wrist_img_shape": wrist_img.shape,
            "wrist_img_range": [float(wrist_img.min()), float(wrist_img.max())],
            "state": state.tolist(),
            "task": task,
        }
    }

    if verbose:
        print(f"\n  Input observation:")
        print(f"    Front image: {front_img.shape}, range [{front_img.min()}, {front_img.max()}]")
        print(f"    Wrist image: {wrist_img.shape}, range [{wrist_img.min()}, {wrist_img.max()}]")
        print(f"    State: {state[:3]}... (first 3 joints)")
        print(f"    Task: '{task}'")

    # Run inference
    with torch.no_grad():
        action = policy.get_action(obs_dict)

    # Extract action
    single_arm = action["action.single_arm"][0]
    gripper = action["action.gripper"][0]
    full_action = np.concatenate([single_arm, gripper])

    log_info["output"] = {
        "action": full_action.tolist(),
        "action_horizon": len(action["action.single_arm"]),
    }

    if verbose:
        print(f"  Output action:")
        for i, (name, val) in enumerate(zip(JOINT_NAMES, full_action)):
            print(f"    {name}: {val:.2f}°")

    return full_action, log_info


def test_input_responsiveness(policy, sample: Dict, verbose: bool = False) -> Dict:
    """
    Test if model responds to different inputs.

    Tests:
    1. Same image, different states → actions should differ
    2. Same state, different images → actions should differ
    3. Same inputs → actions should be consistent
    """
    print(f"\n{'='*70}")
    print(f" TEST: Input Responsiveness")
    print(f"{'='*70}")

    results = {}

    front_img = sample["video.front"]
    wrist_img = sample["video.wrist"]
    state = sample["state"]
    task = sample["task"]

    # Baseline action
    print("\n  1. Baseline action (original inputs):")
    baseline_action, _ = run_inference_with_logging(
        policy, front_img, wrist_img, state, task, verbose=True
    )

    # Test 1: Perturb state
    print("\n  2. Perturbed state (+10° to first joint):")
    perturbed_state = state.copy()
    perturbed_state[0] += 10.0
    perturbed_action, _ = run_inference_with_logging(
        policy, front_img, wrist_img, perturbed_state, task, verbose=True
    )

    state_diff = np.abs(perturbed_action - baseline_action)
    results["state_sensitivity"] = {
        "mean_diff": float(np.mean(state_diff)),
        "max_diff": float(np.max(state_diff)),
        "per_joint_diff": state_diff.tolist(),
    }

    if np.mean(state_diff) < 0.1:
        print(f"\n  ⚠️  WARNING: Model NOT sensitive to state changes!")
        print(f"      Mean action diff: {np.mean(state_diff):.4f}° (should be > 1°)")
        results["state_sensitivity"]["passed"] = False
    else:
        print(f"\n  ✓ Model IS sensitive to state changes")
        print(f"    Mean action diff: {np.mean(state_diff):.2f}°")
        results["state_sensitivity"]["passed"] = True

    # Test 2: Perturb image (add noise)
    print("\n  3. Perturbed image (add noise):")
    noise = np.random.normal(0, 30, front_img.shape).astype(np.uint8)
    noisy_front = np.clip(front_img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    noisy_action, _ = run_inference_with_logging(
        policy, noisy_front, wrist_img, state, task, verbose=True
    )

    image_diff = np.abs(noisy_action - baseline_action)
    results["image_sensitivity"] = {
        "mean_diff": float(np.mean(image_diff)),
        "max_diff": float(np.max(image_diff)),
        "per_joint_diff": image_diff.tolist(),
    }

    if np.mean(image_diff) < 0.1:
        print(f"\n  ⚠️  WARNING: Model NOT sensitive to image changes!")
        print(f"      Mean action diff: {np.mean(image_diff):.4f}° (should be > 0.5°)")
        results["image_sensitivity"]["passed"] = False
    else:
        print(f"\n  ✓ Model IS sensitive to image changes")
        print(f"    Mean action diff: {np.mean(image_diff):.2f}°")
        results["image_sensitivity"]["passed"] = True

    # Test 3: Consistency (same inputs should give same output)
    print("\n  4. Consistency test (same inputs, 3 runs):")
    actions = []
    for i in range(3):
        action, _ = run_inference_with_logging(
            policy, front_img, wrist_img, state, task, verbose=False
        )
        actions.append(action)
        print(f"    Run {i+1}: {action[:3]}...")

    action_std = np.std(actions, axis=0)
    results["consistency"] = {
        "std": action_std.tolist(),
        "mean_std": float(np.mean(action_std)),
    }

    if np.mean(action_std) > 1.0:
        print(f"\n  ⚠️  WARNING: Model outputs inconsistent!")
        print(f"      Mean std: {np.mean(action_std):.2f}° (should be < 0.5°)")
        results["consistency"]["passed"] = False
    else:
        print(f"\n  ✓ Model outputs are consistent")
        print(f"    Mean std: {np.mean(action_std):.4f}°")
        results["consistency"]["passed"] = True

    return results


def test_task_conditioning(policy, sample: Dict, verbose: bool = False) -> Dict:
    """
    Test if model responds to different task descriptions.
    """
    print(f"\n{'='*70}")
    print(f" TEST: Task Conditioning")
    print(f"{'='*70}")

    results = {}

    front_img = sample["video.front"]
    wrist_img = sample["video.wrist"]
    state = sample["state"]

    test_tasks = [
        "pick red_cube from center",
        "place the object",
        "push the cube",
        "reach to the target",
        "grasp and lift",
    ]

    print("\n  Testing different task descriptions:")
    task_actions = {}

    for task in test_tasks:
        action, _ = run_inference_with_logging(
            policy, front_img, wrist_img, state, task, verbose=False
        )
        task_actions[task] = action
        print(f"\n  Task: '{task}'")
        print(f"    Action: [{action[0]:.1f}°, {action[1]:.1f}°, {action[2]:.1f}°, ...]")

    # Compute pairwise differences
    action_list = list(task_actions.values())
    diffs = []
    for i in range(len(action_list)):
        for j in range(i + 1, len(action_list)):
            diff = np.mean(np.abs(action_list[i] - action_list[j]))
            diffs.append(diff)

    mean_diff = np.mean(diffs)
    results["task_sensitivity"] = {
        "mean_pairwise_diff": float(mean_diff),
        "task_actions": {task: action.tolist() for task, action in task_actions.items()},
    }

    print(f"\n  Mean pairwise action difference: {mean_diff:.2f}°")

    if mean_diff < 0.5:
        print(f"\n  ⚠️  WARNING: Model NOT sensitive to task descriptions!")
        print(f"      All tasks produce nearly identical actions.")
        results["task_sensitivity"]["passed"] = False
    else:
        print(f"\n  ✓ Model IS sensitive to task descriptions")
        results["task_sensitivity"]["passed"] = True

    return results


def test_action_range(policy, samples: List[Dict], verbose: bool = False) -> Dict:
    """Test that model produces actions in reasonable ranges."""
    print(f"\n{'='*70}")
    print(f" TEST: Action Range Analysis")
    print(f"{'='*70}")

    all_actions = []

    for sample in samples:
        action, _ = run_inference_with_logging(
            policy,
            sample["video.front"],
            sample["video.wrist"],
            sample["state"],
            sample["task"],
            verbose=False,
        )
        all_actions.append(action)

    all_actions = np.array(all_actions)

    results = {
        "action_stats": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
        }
    }

    print("\n  Per-joint action statistics:")
    print(f"  {'Joint':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 47)

    for i, joint in enumerate(JOINT_NAMES):
        mean = all_actions[:, i].mean()
        std = all_actions[:, i].std()
        min_val = all_actions[:, i].min()
        max_val = all_actions[:, i].max()
        print(f"  {joint:<15} {mean:>8.2f} {std:>8.2f} {min_val:>8.2f} {max_val:>8.2f}")

    # Check for degenerate behavior
    action_std = all_actions.std(axis=0)
    if np.mean(action_std) < 0.5:
        print(f"\n  ⚠️  WARNING: Model produces near-constant actions!")
        print(f"      Mean std: {np.mean(action_std):.4f}° (should vary across samples)")
        results["action_range_ok"] = False
    else:
        print(f"\n  ✓ Model produces varied actions across samples")
        results["action_range_ok"] = True

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose GR00T model behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/jrobot/project/XLeRobot/datasets_groot",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed logging"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save diagnosis results JSON"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1

    # Load samples
    print(f"\nLoading {args.num_samples} samples from dataset...")
    samples = load_sample_frames(dataset_path, args.num_samples)
    if len(samples) == 0:
        print("ERROR: No samples loaded")
        return 1
    print(f"Loaded {len(samples)} samples")

    # Load model
    try:
        policy = load_model_for_diagnosis(checkpoint_path)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        traceback.print_exc()
        return 1

    # Run diagnostic tests
    all_results = {"checkpoint": str(checkpoint_path)}

    # Test 1: Input responsiveness
    input_results = test_input_responsiveness(policy, samples[0], args.verbose)
    all_results["input_responsiveness"] = input_results

    # Test 2: Task conditioning
    task_results = test_task_conditioning(policy, samples[0], args.verbose)
    all_results["task_conditioning"] = task_results

    # Test 3: Action range analysis
    range_results = test_action_range(policy, samples, args.verbose)
    all_results["action_range"] = range_results

    # Summary
    print(f"\n{'='*70}")
    print(f" DIAGNOSIS SUMMARY")
    print(f"{'='*70}")

    tests_passed = 0
    tests_total = 0

    checks = [
        ("State sensitivity", input_results.get("state_sensitivity", {}).get("passed", False)),
        ("Image sensitivity", input_results.get("image_sensitivity", {}).get("passed", False)),
        ("Output consistency", input_results.get("consistency", {}).get("passed", False)),
        ("Task conditioning", task_results.get("task_sensitivity", {}).get("passed", False)),
        ("Action range OK", range_results.get("action_range_ok", False)),
    ]

    for name, passed in checks:
        tests_total += 1
        if passed:
            tests_passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    print(f"\n  {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print("\n  \033[92mAll tests passed! Model behavior looks healthy.\033[0m")
    elif tests_passed >= 3:
        print("\n  \033[93mMost tests passed. Review failed tests.\033[0m")
    else:
        print("\n  \033[91mMultiple tests failed. Model may have issues.\033[0m")

    all_results["summary"] = {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDiagnosis results saved to: {output_path}")

    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())
