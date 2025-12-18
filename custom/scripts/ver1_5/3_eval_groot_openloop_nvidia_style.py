#!/usr/bin/env python3
"""
GR00T Open-Loop Evaluation Script - NVIDIA Style Visualization

Reproduces the exact visualization style from NVIDIA's official eval_policy.py:
- Shows state joints (blue), gt action (orange), pred action (green)
- Shows inference points as red dots every action_horizon steps
- Includes LoRA checkpoint support

Reference: scripts/eval_policy.py + gr00t/utils/eval.py

Usage:
    # Basic evaluation with NVIDIA-style plot
    python eval_groot_openloop_nvidia_style.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/dataset \
        --plot

    # Save plot to file
    python eval_groot_openloop_nvidia_style.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/dataset \
        --save-plot output.png

    # Evaluate multiple trajectories
    python eval_groot_openloop_nvidia_style.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/dataset \
        --trajs 3 --plot
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

warnings.filterwarnings("ignore", message=".*video decoding.*")
warnings.filterwarnings("ignore", message=".*albumentations.*")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# Joint names for SO-101 arm (5 arm joints + 1 gripper)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def is_lora_checkpoint(model_path: str) -> bool:
    """Detect if the given path is a LoRA (PEFT) checkpoint."""
    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"
    adapter_model = model_path / "adapter_model.safetensors"
    return adapter_config.exists() and adapter_model.exists()


def load_policy(
    model_path: str,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
    denoising_steps: int = 4,
):
    """
    Load GR00T policy with automatic LoRA detection and loading.
    """
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    model_path = Path(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_lora_checkpoint(str(model_path)):
        print(f"[INFO] Detected LoRA checkpoint")
        from peft import PeftModel
        from gr00t.model.gr00t_n1 import GR00T_N1_5

        with open(model_path / "adapter_config.json") as f:
            lora_config = json.load(f)

        base_model_path = lora_config.get("base_model_name_or_path")
        print(f"[INFO] Base model: {base_model_path}")
        print(f"[INFO] LoRA rank: {lora_config.get('r')}")

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
        base_model.eval()

        peft_model = PeftModel.from_pretrained(base_model, str(model_path))
        merged_model = peft_model.merge_and_unload()
        merged_model.to(device=device)

        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy.device = device
        policy._modality_config = modality_config
        policy._modality_transform = modality_transform
        policy.model = merged_model
        policy.model.action_head.num_inference_timesteps = denoising_steps

        policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
        policy._video_horizon = len(policy._video_delta_indices)
        if "state" in modality_config:
            policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
            policy._state_horizon = len(policy._state_delta_indices)
        policy._action_delta_indices = np.array(modality_config["action"].delta_indices)
        policy._action_horizon = len(policy._action_delta_indices)

        # Load metadata for normalization
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

        print(f"[INFO] LoRA model loaded and merged")
    else:
        print(f"[INFO] Loading full checkpoint")
        policy = Gr00tPolicy(
            model_path=str(model_path),
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )

    return policy


def calc_mse_for_single_trajectory_nvidia_style(
    policy,
    dataset,
    traj_id: int,
    modality_keys: List[str],
    steps: int = 300,
    action_horizon: int = 16,
    plot: bool = False,
    plot_state: bool = True,
    save_plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate MSE for a single trajectory - NVIDIA style.

    This function reproduces the exact behavior of gr00t/utils/eval.py
    calc_mse_for_single_trajectory() with LoRA support.

    Returns dict with:
        - mse: Overall MSE
        - mae: Overall MAE
        - per_joint_mae: Per-joint MAE
        - state_across_time: State trajectory
        - gt_action_across_time: Ground truth actions
        - pred_action_across_time: Predicted actions
    """
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []

    traj_length = dataset.trajectory_lengths[traj_id]
    # If steps not specified, use entire trajectory (NVIDIA style)
    if steps is None:
        steps = traj_length - action_horizon
    else:
        steps = min(steps, traj_length - action_horizon)

    print(f"  Evaluating trajectory {traj_id}: {steps} steps, action_horizon={action_horizon}")

    for step_count in range(steps):
        data_point = None

        # Always get state for plotting (NVIDIA style)
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        # Run inference at action_horizon intervals (every 16 steps)
        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            # Run model inference
            with torch.no_grad():
                action_chunk = policy.get_action(data_point)

            # Collect all actions in this chunk
            for j in range(action_horizon):
                # Predicted action at horizon index j
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                # Ground truth action at horizon index j (from the same observation)
                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

    # Convert to numpy arrays and trim to same length
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]

    assert gt_action_across_time.shape == pred_action_across_time.shape, \
        f"Shape mismatch: gt={gt_action_across_time.shape}, pred={pred_action_across_time.shape}"

    # Check for NaN in predictions
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Predicted actions contain NaN values!")

    # Calculate metrics
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    per_joint_mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time), axis=0)

    print(f"    MSE: {mse:.4f}")
    print(f"    MAE: {mae:.4f}")

    # Generate NVIDIA-style plot
    if plot or save_plot_path is not None:
        plot_trajectory_nvidia_style(
            state_joints_across_time=state_joints_across_time,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            modality_keys=modality_keys,
            traj_id=traj_id,
            mse=mse,
            mae=mae,
            action_horizon=action_horizon,
            steps=steps,
            save_plot_path=save_plot_path,
            show_plot=plot,
        )

    return {
        "mse": float(mse),
        "mae": float(mae),
        "per_joint_mae": per_joint_mae.tolist(),
        "state_across_time": state_joints_across_time,
        "gt_action_across_time": gt_action_across_time,
        "pred_action_across_time": pred_action_across_time,
    }


def plot_trajectory_nvidia_style(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    modality_keys: List[str],
    traj_id: int,
    mse: float,
    mae: float,
    action_horizon: int,
    steps: int,
    save_plot_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Generate NVIDIA-style trajectory visualization.

    Reproduces the exact plot format from gr00t/utils/eval.py plot_trajectory()
    with these elements:
    - Blue line: state joints (current robot position)
    - Orange line: gt action joints (ground truth actions)
    - Green line: pred action joints (model predictions)
    - Red dots: inference points (where model runs)
    """
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = gt_action_across_time.shape[1]

    # Figure sizing to match NVIDIA style
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    if action_dim == 1:
        axes = [axes]

    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    # Build title with modality info (NVIDIA style)
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "

    title_text = (
        f"Trajectory {traj_id} - Modalities: {modality_string[:-2]}\n"
        f"Unnormalized MSE: {mse:.6f} | MAE: {mae:.4f}"
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.98)

    # Plot each action dimension
    for i, ax in enumerate(axes):
        joint_name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"Joint {i}"

        # Plot state joints if available and shapes match
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7, color="tab:blue")

        # Plot ground truth actions (orange in NVIDIA style)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2, color="tab:orange")

        # Plot predicted actions (green in NVIDIA style)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2, color="tab:green")

        # Add red dots at inference points (every action_horizon steps)
        for j in range(0, steps, action_horizon):
            if j < len(gt_action_across_time):
                if j == 0:
                    ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
                else:
                    ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        # Per-joint MAE in title
        joint_mae = np.mean(np.abs(gt_action_across_time[:, i] - pred_action_across_time[:, i]))
        ax.set_title(f"{joint_name} (Joint {i}) - MAE: {joint_mae:.2f}°", fontsize=12, fontweight="bold", pad=10)

        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value (degrees)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot_path:
        print(f"    Saving plot to: {save_plot_path}")
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="GR00T Open-Loop Evaluation - NVIDIA Style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate with NVIDIA-style plot (recommended)
    python eval_groot_openloop_nvidia_style.py \\
        --checkpoint /path/to/checkpoint \\
        --dataset /path/to/dataset \\
        --plot

    # Evaluate multiple trajectories and save plots
    python eval_groot_openloop_nvidia_style.py \\
        --checkpoint /path/to/checkpoint \\
        --dataset /path/to/dataset \\
        --trajs 3 \\
        --save-plot eval_output.png

    # Compare with official NVIDIA script:
    python scripts/eval_policy.py --plot \\
        --embodiment_tag new_embodiment \\
        --model_path /path/to/checkpoint \\
        --data_config so100_dualcam \\
        --dataset_path /path/to/dataset \\
        --modality_keys single_arm gripper
        """
    )

    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to GR00T checkpoint (full or LoRA)")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--data-config", type=str, default="so100_dualcam",
                        help="Data configuration name")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment",
                        help="Embodiment tag for the model")
    parser.add_argument("--modality-keys", nargs="+", default=["single_arm", "gripper"],
                        help="Action modality keys (default: single_arm gripper)")
    parser.add_argument("--trajs", type=int, default=1,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--start-traj", type=int, default=0,
                        help="Starting trajectory index")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of steps to evaluate per trajectory (default: entire trajectory)")
    parser.add_argument("--action-horizon", type=int, default=None,
                        help="Action horizon (default: from data config)")
    parser.add_argument("--plot", action="store_true",
                        help="Display trajectory plots")
    parser.add_argument("--no-state", action="store_true",
                        help="Don't plot state joints (only gt and pred)")
    parser.add_argument("--save-plot", type=str,
                        default="eval_images/openloop_eval.png",
                        help="Save trajectory plot to this path (default: eval_images/openloop_eval.png)")
    parser.add_argument("--video-backend", type=str, default="torchvision_av",
                        choices=["decord", "torchvision_av", "torchcodec"],
                        help="Video backend")
    parser.add_argument("--denoising-steps", type=int, default=4,
                        help="Number of denoising steps")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to JSON file")

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

    # Ensure eval_images directory exists
    if args.save_plot:
        save_plot_path = Path(args.save_plot)
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GR00T Open-Loop Evaluation - NVIDIA Style")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Data config: {args.data_config}")
    print(f"Modality keys: {args.modality_keys}")
    print(f"Trajectories: {args.start_traj} to {args.start_traj + args.trajs - 1}")
    print(f"Steps per trajectory: {args.steps if args.steps else 'entire trajectory'}")
    print("=" * 70)

    # Load policy
    print("\n[1/3] Loading model...")
    policy = load_policy(
        model_path=args.checkpoint,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # Get action horizon from data config if not specified
    if args.action_horizon is None:
        from gr00t.experiment.data_config import load_data_config
        data_cfg = load_data_config(args.data_config)
        args.action_horizon = len(data_cfg.action_indices)
        print(f"[INFO] Using action_horizon={args.action_horizon} from data config")

    # Load dataset
    print("\n[2/3] Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config

    try:
        import decord
        decord.bridge.set_bridge("native")
    except ImportError:
        pass

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
    print(f"  Trajectory lengths: {dataset.trajectory_lengths}")

    # Run evaluation
    print("\n[3/3] Running open-loop evaluation...")

    all_results = []
    all_mse = []
    all_mae = []

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
                base = Path(args.save_plot)
                save_path = str(base.parent / f"{base.stem}_traj{traj_id}{base.suffix}")

        result = calc_mse_for_single_trajectory_nvidia_style(
            policy=policy,
            dataset=dataset,
            traj_id=traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=not args.no_state,
            save_plot_path=save_path,
        )

        all_results.append({"traj_id": traj_id, **{k: v for k, v in result.items()
                                                    if k not in ["state_across_time", "gt_action_across_time", "pred_action_across_time"]}})
        all_mse.append(result["mse"])
        all_mae.append(result["mae"])

    # Summary
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Trajectories evaluated: {len(all_mse)}")
    print(f"  Average MSE: {avg_mse:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}°")
    print()
    print("  Per-trajectory results:")
    for i, (mse, mae) in enumerate(zip(all_mse, all_mae)):
        traj_id = args.start_traj + i
        print(f"    Trajectory {traj_id}: MSE={mse:.4f}, MAE={mae:.4f}°")

    # Interpretation guide
    print()
    print("  Interpretation:")
    print(f"    MSE < 100 (~MAE < 10°): Model tracks trajectory well")
    print(f"    MSE 100-500 (~MAE 10-22°): Model learning, may need more training")
    print(f"    MSE > 500 (~MAE > 22°): Poor learning, check data/training")
    print("=" * 70)

    # Save results
    if args.output:
        output_data = {
            "summary": {
                "average_mse": float(avg_mse),
                "average_mae": float(avg_mae),
                "num_trajectories": len(all_mse),
                "checkpoint": str(args.checkpoint),
                "dataset": str(args.dataset),
            },
            "trajectories": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
