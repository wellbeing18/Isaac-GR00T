#!/usr/bin/env python3
"""
GR00T Open-Loop Evaluation Script with Trajectory Plots

This script performs open-loop evaluation of GR00T checkpoints by comparing
predicted action trajectories against ground truth. It integrates the official
NVIDIA eval_policy.py approach with LoRA checkpoint support.

FPS Configuration (per 6_fps_upgrade_30hz.md):
    - Action FPS: 30 Hz (synchronized with video)
    - Video FPS: 30 fps
    - Dataset should be recorded at 30 Hz for evaluation

Outputs:
1. MSE (Mean Squared Error) for action predictions
2. Trajectory comparison plots (predicted vs ground truth)
3. Per-joint visualization over time

Usage:
    # Evaluate with trajectory plots
    python eval_groot_openloop.py --checkpoint /path/to/checkpoint --plot

    # Evaluate multiple trajectories
    python eval_groot_openloop.py --checkpoint /path/to/checkpoint --trajs 5 --plot

    # Save plots to file
    python eval_groot_openloop.py --checkpoint /path/to/checkpoint --plot --save-plot eval.png
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# Suppress torchvision video deprecation warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")
warnings.filterwarnings("ignore", message=".*albumentations.*")

import numpy as np
import torch
from tqdm import tqdm

# Add parent path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


# Joint names for SO-101 arm
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def is_lora_checkpoint(model_path: str) -> bool:
    """Detect if the given path is a LoRA (PEFT) checkpoint."""
    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"
    adapter_model = model_path / "adapter_model.safetensors"
    return adapter_config.exists() and adapter_model.exists()


def load_policy_with_lora_support(
    model_path: str,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
    denoising_steps: int = 4,
):
    """
    Load GR00T policy with automatic LoRA detection and loading.

    Args:
        model_path: Path to checkpoint (full or LoRA)
        data_config: Data configuration name
        embodiment_tag: Embodiment tag for the model
        denoising_steps: Number of denoising steps

    Returns:
        Loaded Gr00tPolicy ready for inference
    """
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    # Load data config
    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    model_path = Path(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_lora_checkpoint(str(model_path)):
        print(f"[EVAL] Detected LoRA checkpoint - loading with PEFT adapter merge")

        # Use the LoRA loading function from infer_groot_so101.py
        try:
            from infer_groot_so101 import load_groot_with_lora
            policy = load_groot_with_lora(
                model_path=str(model_path),
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
                merge_weights=True,
            )
        except ImportError:
            # Inline LoRA loading if import fails
            from peft import PeftModel
            from gr00t.model.gr00t_n1 import GR00T_N1_5

            # Load adapter config
            with open(model_path / "adapter_config.json") as f:
                lora_config = json.load(f)

            base_model_path = lora_config.get("base_model_name_or_path")
            print(f"[EVAL] Base model: {base_model_path}")
            print(f"[EVAL] LoRA rank: {lora_config.get('r')}")

            # Load base model
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
            base_model.eval()

            # Load and merge LoRA adapter
            peft_model = PeftModel.from_pretrained(base_model, str(model_path))
            merged_model = peft_model.merge_and_unload()
            merged_model.to(device=device)

            # Create policy wrapper
            policy = Gr00tPolicy.__new__(Gr00tPolicy)
            policy.device = device
            policy._modality_config = modality_config
            policy._modality_transform = modality_transform
            policy.model = merged_model
            policy.model.action_head.num_inference_timesteps = denoising_steps

            # Load horizons
            policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
            policy._video_horizon = len(policy._video_delta_indices)
            if "state" in modality_config:
                policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
                policy._state_horizon = len(policy._state_delta_indices)
            policy._action_delta_indices = np.array(modality_config["action"].delta_indices)
            policy._action_horizon = len(policy._action_delta_indices)

            # Load metadata
            exp_cfg_dir = model_path / "experiment_cfg"
            if exp_cfg_dir.exists():
                metadata_path = exp_cfg_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadatas = json.load(f)

                    from gr00t.data.dataset import DatasetMetadata
                    from gr00t.model.policy import EmbodimentTag

                    embodiment_tag_enum = EmbodimentTag(embodiment_tag)
                    policy.embodiment_tag = embodiment_tag_enum

                    metadata_dict = metadatas.get(embodiment_tag_enum.value)
                    if metadata_dict:
                        metadata = DatasetMetadata.model_validate(metadata_dict)
                        policy._modality_transform.set_metadata(metadata)
                        policy.metadata = metadata

            print(f"[EVAL] LoRA model loaded and merged successfully")
    else:
        print(f"[EVAL] Loading full checkpoint")
        policy = Gr00tPolicy(
            model_path=str(model_path),
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )

    return policy


def calc_mse_for_trajectory(
    policy,
    dataset,
    traj_id: int,
    modality_keys: list,
    steps: int = 150,
    action_horizon: int = 16,
    plot: bool = False,
    save_plot_path: Optional[str] = None,
) -> float:
    """
    Calculate MSE for a single trajectory using open-loop evaluation.

    Runs the policy in open-loop mode (no feedback from robot) and compares
    predicted action sequences against ground truth.

    Args:
        policy: Loaded Gr00tPolicy
        dataset: LeRobotSingleDataset
        traj_id: Trajectory/episode index
        modality_keys: Action modality keys (e.g., ["single_arm", "gripper"])
        steps: Number of steps to evaluate
        action_horizon: Number of actions to predict per inference
        plot: Whether to generate trajectory plots
        save_plot_path: Path to save the plot

    Returns:
        MSE value for the trajectory
    """
    traj_length = dataset.trajectory_lengths[traj_id]
    steps = min(steps, traj_length - action_horizon)

    all_gt_actions = []
    all_pred_actions = []

    print(f"  Evaluating trajectory {traj_id} ({steps} steps)...")

    for step in tqdm(range(0, steps, action_horizon), desc=f"Traj {traj_id}", leave=False):
        # Get observation from dataset
        obs = dataset.get_step_data(traj_id, step)

        # Run inference
        with torch.no_grad():
            action_dict = policy.get_action(obs)

        # Collect predictions and ground truth for each action in the horizon
        for h in range(action_horizon):
            if step + h >= traj_length:
                break

            # Get ground truth action from the observation at this step
            # action.single_arm has shape (horizon, 5), action.gripper has shape (horizon, 1)
            # We want the first action (index 0) as the ground truth for this step
            gt_obs = dataset.get_step_data(traj_id, step + h)
            gt_action = np.concatenate([
                gt_obs[f"action.{key}"][0].flatten() for key in modality_keys
            ], axis=0)

            # Get predicted action at horizon index h
            pred_action = np.concatenate([
                np.atleast_1d(action_dict[f"action.{key}"][h]).flatten() for key in modality_keys
            ], axis=0)

            all_gt_actions.append(gt_action)
            all_pred_actions.append(pred_action)

    all_gt_actions = np.array(all_gt_actions)
    all_pred_actions = np.array(all_pred_actions)

    # Calculate MSE
    mse = np.mean((all_pred_actions - all_gt_actions) ** 2)

    # Generate plots if requested
    if plot or save_plot_path:
        import matplotlib.pyplot as plt

        num_joints = all_gt_actions.shape[1]
        fig, axes = plt.subplots(num_joints, 1, figsize=(14, 3 * num_joints), sharex=True)

        if num_joints == 1:
            axes = [axes]

        time_steps = np.arange(len(all_gt_actions))

        for i, (ax, joint_name) in enumerate(zip(axes, JOINT_NAMES[:num_joints])):
            ax.plot(time_steps, all_gt_actions[:, i], 'b-', label='Ground Truth', linewidth=1.5)
            ax.plot(time_steps, all_pred_actions[:, i], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)

            # Calculate per-joint MSE
            joint_mse = np.mean((all_pred_actions[:, i] - all_gt_actions[:, i]) ** 2)

            ax.set_ylabel(f'{joint_name}\n(deg)', fontsize=10)
            ax.set_title(f'{joint_name} - MSE: {joint_mse:.2f}', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time Step (30 Hz)', fontsize=11)
        plt.suptitle(f'Open-Loop Evaluation - Trajectory {traj_id}\nOverall MSE: {mse:.2f}', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_plot_path:
            plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to: {save_plot_path}")

        if plot:
            plt.show()
        else:
            plt.close()

    return mse


def main():
    parser = argparse.ArgumentParser(
        description="GR00T Open-Loop Evaluation with Trajectory Plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a LoRA checkpoint with trajectory plot
    python eval_groot_openloop.py \\
        --checkpoint /path/to/groot_5k_lora/checkpoint-5000 \\
        --dataset /path/to/datasets_groot \\
        --plot

    # Evaluate multiple trajectories and save plots
    python eval_groot_openloop.py \\
        --checkpoint /path/to/checkpoint \\
        --trajs 5 \\
        --save-plot /path/to/output_plot.png

    # Use official eval_policy.py approach (requires torchcodec)
    python scripts/eval_policy.py \\
        --plot \\
        --embodiment_tag new_embodiment \\
        --model_path /path/to/checkpoint \\
        --data_config so100_dualcam \\
        --dataset_path /path/to/dataset \\
        --modality_keys single_arm gripper
        """
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to GR00T checkpoint (full or LoRA)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/jrobot/project/XLeRobot/datasets_groot",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="so100_dualcam",
        help="Data configuration name (default: so100_dualcam)"
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag for the model"
    )
    parser.add_argument(
        "--trajs",
        type=int,
        default=1,
        help="Number of trajectories to evaluate (default: 1)"
    )
    parser.add_argument(
        "--start-traj",
        type=int,
        default=0,
        help="Starting trajectory index (default: 0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of steps to evaluate per trajectory (default: 150)"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Action horizon for inference (default: 16)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display trajectory plots"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        help="Save trajectory plot to this path"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchvision_av",
        choices=["decord", "torchvision_av", "torchcodec"],
        help="Video backend (default: torchvision_av)"
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Number of denoising steps (default: 4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save MSE results to JSON file"
    )

    args = parser.parse_args()

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1

    print("="*70)
    print("GR00T Open-Loop Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Data config: {args.data_config}")
    print(f"Trajectories: {args.start_traj} to {args.start_traj + args.trajs - 1}")
    print(f"Steps per trajectory: {args.steps}")
    print(f"Action horizon: {args.action_horizon}")
    print("="*70)

    # Load policy with LoRA support
    print("\n[1/3] Loading model...")
    policy = load_policy_with_lora_support(
        model_path=args.checkpoint,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # Load dataset
    print("\n[2/3] Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    import decord
    decord.bridge.set_bridge("native")

    data_cfg = load_data_config(args.data_config)
    modality_config = data_cfg.modality_config()

    dataset = LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )

    print(f"  Loaded {len(dataset)} samples from {len(dataset.trajectory_lengths)} trajectories")

    # Run evaluation
    print("\n[3/3] Running open-loop evaluation...")
    modality_keys = ["single_arm", "gripper"]

    all_mse = []
    results = {}

    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        if traj_id >= len(dataset.trajectory_lengths):
            print(f"  Warning: Trajectory {traj_id} not found, stopping")
            break

        # Determine save path for this trajectory
        save_path = None
        if args.save_plot:
            if args.trajs == 1:
                save_path = args.save_plot
            else:
                base, ext = os.path.splitext(args.save_plot)
                save_path = f"{base}_traj{traj_id}{ext}"

        mse = calc_mse_for_trajectory(
            policy=policy,
            dataset=dataset,
            traj_id=traj_id,
            modality_keys=modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            save_plot_path=save_path,
        )

        all_mse.append(mse)
        results[f"trajectory_{traj_id}"] = {"mse": float(mse)}
        print(f"  Trajectory {traj_id} MSE: {mse:.4f}")

    # Summary
    avg_mse = np.mean(all_mse)

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"  Trajectories evaluated: {len(all_mse)}")
    print(f"  Average MSE: {avg_mse:.4f}")
    print()
    print("  Per-trajectory MSE:")
    for i, mse in enumerate(all_mse):
        traj_id = args.start_traj + i
        print(f"    Trajectory {traj_id}: {mse:.4f}")

    # Interpretation
    print()
    print("  Interpretation:")
    if avg_mse < 100:
        print("    ✓ MSE < 100: Model is learning trajectory dynamics well")
    elif avg_mse < 500:
        print("    ~ MSE 100-500: Model is learning but may need more training")
    else:
        print("    ✗ MSE > 500: Model may be outputting mean actions (poor learning)")
        print("      Possible causes: insufficient training, wrong FPS, data quality")
    print("="*70)

    # Save results
    if args.output:
        results["summary"] = {
            "average_mse": float(avg_mse),
            "num_trajectories": len(all_mse),
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
