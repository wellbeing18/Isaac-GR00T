#!/usr/bin/env python3
"""
GR00T Checkpoint Evaluation Script

Evaluate GR00T checkpoints on training data to compute:
1. Overall MAE (Mean Absolute Error)
2. Per-joint MAE (shoulder_pan, shoulder_lift, elbow_flex, etc.)
3. Accuracy at thresholds (5°, 10°, 15°)
4. Action distribution comparison (predicted vs ground truth)

Similar to Pi0.5's evaluate_checkpoint.py

FPS Configuration (per 6_fps_upgrade_30hz.md):
    - Action FPS: 30 Hz (synchronized with video)
    - Video FPS: 30 fps
    - Dataset should be recorded at 30 Hz for evaluation

For open-loop evaluation with trajectory plots, use eval_groot_openloop.py which
integrates with the official NVIDIA eval_policy.py to provide MSE metrics and
visual trajectory comparisons.

Usage:
    # Evaluate all checkpoints in a training directory
    python evaluate_groot_checkpoint.py --training-dir /path/to/outputs/groot_5k_lora_xxx

    # Evaluate a single checkpoint
    python evaluate_groot_checkpoint.py --checkpoint /path/to/checkpoint --dataset /path/to/datasets_groot

    # Evaluate with more samples
    python evaluate_groot_checkpoint.py --training-dir /path/to/outputs --num-samples 500
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Suppress torchvision video deprecation warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")
warnings.filterwarnings("ignore", message=".*albumentations.*")

import numpy as np
import torch
from tqdm import tqdm


# Joint names for SO-101 arm
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_dataset_samples(
    dataset_path: Path,
    num_samples: int = 300,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
) -> List[Dict]:
    """
    Load samples from a GR00T dataset using LeRobotSingleDataset.

    Returns list of dicts with:
    - video.front: (H, W, 3) numpy array
    - video.wrist: (H, W, 3) numpy array
    - state.single_arm: (5,) numpy array
    - state.gripper: (1,) numpy array
    - action.single_arm: (5,) numpy array
    - action.gripper: (1,) numpy array
    - task: str
    """
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config

    dataset_path = Path(dataset_path)

    # Load data config
    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()

    # Load dataset using GR00T's loader (handles chunked videos, AV1 codec, etc.)
    dataset = LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=modality_config,
        video_backend="torchvision_av",  # Supports AV1 codec
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=embodiment_tag,
    )

    print(f"Dataset loaded: {len(dataset)} samples from {len(dataset.trajectory_lengths)} trajectories")

    # Sample random indices
    total_samples = len(dataset)
    if total_samples > num_samples:
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
    else:
        sample_indices = list(range(total_samples))

    # Load each sample
    print(f"Loading {len(sample_indices)} samples from dataset...")
    samples = []

    for idx in tqdm(sample_indices, desc="Loading samples"):
        try:
            # Get sample from dataset
            obs = dataset[idx]

            # Extract video frames (shape: [1, H, W, 3] -> [H, W, 3])
            front_frame = obs["video.front"][0]
            if isinstance(front_frame, torch.Tensor):
                front_frame = front_frame.numpy()

            wrist_frame = obs["video.wrist"][0]
            if isinstance(wrist_frame, torch.Tensor):
                wrist_frame = wrist_frame.numpy()

            # Extract state (shape: [1, N] -> [N,])
            state_arm = obs["state.single_arm"][0]
            if isinstance(state_arm, torch.Tensor):
                state_arm = state_arm.numpy()

            state_gripper = obs["state.gripper"][0]
            if isinstance(state_gripper, torch.Tensor):
                state_gripper = state_gripper.numpy()

            # Extract action - use first action from horizon (shape: [horizon, N] -> [N,])
            action_arm = obs["action.single_arm"][0]
            if isinstance(action_arm, torch.Tensor):
                action_arm = action_arm.numpy()

            action_gripper = obs["action.gripper"][0]
            if isinstance(action_gripper, torch.Tensor):
                action_gripper = action_gripper.numpy()

            # Get task description
            task = obs.get("annotation.human.task_description", ["unknown task"])
            if isinstance(task, list):
                task = task[0] if task else "unknown task"

            samples.append({
                "video.front": front_frame.astype(np.uint8),
                "video.wrist": wrist_frame.astype(np.uint8),
                "state.single_arm": state_arm.astype(np.float32),
                "state.gripper": state_gripper.astype(np.float32),
                "action.single_arm": action_arm.astype(np.float32),
                "action.gripper": action_gripper.astype(np.float32),
                "task": task,
            })

        except Exception as e:
            # Skip problematic samples silently
            continue

    print(f"Loaded {len(samples)} valid samples")
    return samples


def load_model(checkpoint_path: Path, data_config: str = "so100_dualcam"):
    """Load a GR00T model from checkpoint (handles both full and LoRA checkpoints)."""
    # Import the inference utilities
    try:
        from infer_groot_async import is_lora_checkpoint, load_groot_with_lora
    except ImportError:
        # Fallback: define inline
        def is_lora_checkpoint(model_path):
            model_path = Path(model_path)
            return (model_path / "adapter_config.json").exists() and \
                   (model_path / "adapter_model.safetensors").exists()

    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    # Load data config
    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    checkpoint_path = Path(checkpoint_path)

    if is_lora_checkpoint(str(checkpoint_path)):
        print(f"[EVAL] Loading LoRA checkpoint: {checkpoint_path}")
        # Use the LoRA loading function
        from infer_groot_async import load_groot_with_lora
        policy = load_groot_with_lora(
            model_path=str(checkpoint_path),
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
            merge_weights=True,
        )
    else:
        print(f"[EVAL] Loading full checkpoint: {checkpoint_path}")
        policy = Gr00tPolicy(
            model_path=str(checkpoint_path),
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
        )

    return policy


def run_inference(policy, sample: Dict) -> np.ndarray:
    """Run inference on a single sample, return predicted action (6,)."""
    # Build observation dict
    obs_dict = {
        "video.front": sample["video.front"][np.newaxis, :, :, :],
        "video.wrist": sample["video.wrist"][np.newaxis, :, :, :],
        "state.single_arm": sample["state.single_arm"][np.newaxis, :].astype(np.float64),
        "state.gripper": sample["state.gripper"][np.newaxis, :].astype(np.float64),
        "annotation.human.task_description": [sample["task"]],
    }

    # Get action
    with torch.no_grad():
        action = policy.get_action(obs_dict)

    # Extract first timestep action
    single_arm = action["action.single_arm"][0]  # (5,)
    gripper = action["action.gripper"][0]  # (1,)

    return np.concatenate([single_arm, gripper])


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: (N, 6) predicted actions
        ground_truth: (N, 6) ground truth actions

    Returns:
        Dict with metrics
    """
    # Compute errors
    errors = np.abs(predictions - ground_truth)  # (N, 6)

    # Overall MAE
    overall_mae = np.mean(errors)

    # Per-joint MAE
    per_joint_mae = np.mean(errors, axis=0)

    # Accuracy at thresholds
    thresholds = [5.0, 10.0, 15.0]
    accuracy_at_threshold = {}
    for thresh in thresholds:
        correct = errors < thresh
        accuracy_at_threshold[f"acc@{int(thresh)}"] = np.mean(correct) * 100

    # Per-joint accuracy at 10°
    per_joint_acc_10 = (errors < 10.0).mean(axis=0) * 100

    # Distribution stats
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)
    gt_mean = np.mean(ground_truth, axis=0)
    gt_std = np.std(ground_truth, axis=0)

    return {
        "overall_mae": float(overall_mae),
        "per_joint_mae": {name: float(mae) for name, mae in zip(JOINT_NAMES, per_joint_mae)},
        "accuracy": accuracy_at_threshold,
        "per_joint_acc_10": {name: float(acc) for name, acc in zip(JOINT_NAMES, per_joint_acc_10)},
        "prediction_stats": {
            "mean": pred_mean.tolist(),
            "std": pred_std.tolist(),
        },
        "ground_truth_stats": {
            "mean": gt_mean.tolist(),
            "std": gt_std.tolist(),
        },
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    num_samples: int = 300,
    data_config: str = "so100_dualcam",
) -> Dict:
    """Evaluate a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'='*70}")

    # Load samples
    samples = load_dataset_samples(dataset_path, num_samples, data_config)
    if len(samples) == 0:
        print("ERROR: No valid samples loaded")
        return {"error": "No samples loaded"}

    # Load model
    try:
        policy = load_model(checkpoint_path, data_config)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    # Run inference on all samples
    print(f"\nRunning inference on {len(samples)} samples...")
    predictions = []
    baseline_state_copy = []
    ground_truths = []

    for sample in tqdm(samples, desc="Inference"):
        try:
            pred = run_inference(policy, sample)
            gt = np.concatenate([sample["action.single_arm"], sample["action.gripper"]])
            baseline = np.concatenate([sample["state.single_arm"], sample["state.gripper"]])

            predictions.append(pred)
            baseline_state_copy.append(baseline)
            ground_truths.append(gt)
        except Exception as e:
            continue

    predictions = np.array(predictions)
    baseline_state_copy = np.array(baseline_state_copy)
    ground_truths = np.array(ground_truths)

    print(f"Evaluated {len(predictions)} samples successfully")

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    metrics["baseline_state_copy"] = compute_metrics(baseline_state_copy, ground_truths)
    metrics["num_samples"] = len(predictions)
    metrics["checkpoint"] = str(checkpoint_path)

    return metrics


def print_metrics(metrics: Dict, checkpoint_name: str = ""):
    """Print metrics in a formatted table."""
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"\n  Checkpoint: {checkpoint_name}")
    print(f"  Samples: {metrics['num_samples']}")
    print()

    # Overall MAE
    print(f"  Overall MAE: {metrics['overall_mae']:.2f}°")
    if "baseline_state_copy" in metrics and "overall_mae" in metrics["baseline_state_copy"]:
        b = metrics["baseline_state_copy"]["overall_mae"]
        delta = metrics["overall_mae"] - b
        print(f"  Baseline (predict action=state) MAE: {b:.2f}° (Δ={delta:+.2f}°)")
    print()

    # Accuracy at thresholds
    print("  Accuracy at thresholds:")
    for thresh_name, acc in metrics["accuracy"].items():
        print(f"    {thresh_name}: {acc:.1f}%")
    print()

    # Per-joint MAE
    print("  Per-joint MAE:")
    for joint, mae in metrics["per_joint_mae"].items():
        acc = metrics["per_joint_acc_10"].get(joint, 0)
        status = "✓" if mae < 10 else "✗"
        print(f"    {joint:15s}: {mae:6.2f}° (acc@10°: {acc:5.1f}%) {status}")


def find_checkpoints(training_dir: Path) -> List[Path]:
    """Find all checkpoints in a training directory."""
    checkpoints = []

    # Check for checkpoint-XXX directories
    checkpoint_dirs = sorted(training_dir.glob("checkpoint-*"))
    for ckpt_dir in checkpoint_dirs:
        if (ckpt_dir / "adapter_config.json").exists() or \
           (ckpt_dir / "config.json").exists():
            checkpoints.append(ckpt_dir)

    # Check root directory (final checkpoint)
    if (training_dir / "adapter_config.json").exists() or \
       (training_dir / "config.json").exists():
        checkpoints.append(training_dir)

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T checkpoints on training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--training-dir", "-t",
        type=str,
        help="Path to training output directory (evaluates all checkpoints)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="Path to a single checkpoint to evaluate"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/jrobot/project/XLeRobot/datasets_groot",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=300,
        help="Number of samples to evaluate (default: 300)"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="so100_dualcam",
        help="Data configuration name"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--last", "-l",
        type=int,
        default=0,
        help="Only evaluate the last N checkpoints (default: 0 = all). Use --last 1 for only the latest, --last 2 for last two, etc."
    )

    args = parser.parse_args()

    if not args.training_dir and not args.checkpoint:
        parser.error("Either --training-dir or --checkpoint must be provided")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1

    all_results = {}

    if args.checkpoint:
        # Evaluate single checkpoint
        checkpoint_path = Path(args.checkpoint)
        metrics = evaluate_checkpoint(
            checkpoint_path, dataset_path, args.num_samples, args.data_config
        )
        print_metrics(metrics, checkpoint_path.name)
        all_results[str(checkpoint_path)] = metrics

    elif args.training_dir:
        # Find and evaluate all checkpoints
        training_dir = Path(args.training_dir)
        checkpoints = find_checkpoints(training_dir)

        if len(checkpoints) == 0:
            print(f"ERROR: No checkpoints found in {training_dir}")
            return 1

        # Filter to last N checkpoints if specified
        if args.last > 0 and len(checkpoints) > args.last:
            print(f"Found {len(checkpoints)} checkpoint(s), evaluating last {args.last}")
            checkpoints = checkpoints[-args.last:]
        else:
            print(f"Found {len(checkpoints)} checkpoint(s)")

        for ckpt in checkpoints:
            metrics = evaluate_checkpoint(
                ckpt, dataset_path, args.num_samples, args.data_config
            )
            print_metrics(metrics, ckpt.name)
            all_results[str(ckpt)] = metrics

        # Summary comparison
        print("\n" + "="*70)
        print(" COMPARISON SUMMARY")
        print("="*70)
        print(f"\n  {'Checkpoint':<20} {'MAE':>8} {'Acc@5°':>8} {'Acc@10°':>8} {'Acc@15°':>8}")
        print("  " + "-"*56)

        best_mae = float("inf")
        best_ckpt = None

        for ckpt_path, metrics in all_results.items():
            if "error" in metrics:
                continue

            ckpt_name = Path(ckpt_path).name[:20]
            mae = metrics["overall_mae"]
            acc5 = metrics["accuracy"]["acc@5"]
            acc10 = metrics["accuracy"]["acc@10"]
            acc15 = metrics["accuracy"]["acc@15"]

            if mae < best_mae:
                best_mae = mae
                best_ckpt = ckpt_path

            print(f"  {ckpt_name:<20} {mae:>7.2f}° {acc5:>7.1f}% {acc10:>7.1f}% {acc15:>7.1f}%")

        if best_ckpt:
            print()
            print(f"  Best checkpoint: {Path(best_ckpt).name}")
            print(f"  Best MAE: {best_mae:.2f}°")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    elif args.training_dir:
        # Auto-save to training dir
        output_path = Path(args.training_dir) / "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
