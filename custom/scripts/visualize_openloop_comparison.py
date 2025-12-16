#!/usr/bin/env python3
"""
Visualize Open-Loop MAE Comparison
===================================
Generate plots comparing:
1. Your model predictions vs ground truth on YOUR dataset
2. Pushpakcc model predictions vs ground truth on THEIR dataset
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuration
YOUR_CHECKPOINT = Path("/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914")
PUSHPAKCC_CHECKPOINT = Path("/tmp/pushpakcc_model/checkpoint-1000")
YOUR_DATASET = Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
REF_DATASET = Path("/tmp/so101-table-cleanup")
EMBODIMENT_TAG = "new_embodiment"
NUM_TRAJECTORIES = 3  # Number of trajectories to visualize
OUTPUT_DIR = Path("/home/jrobot/project/Isaac-GR00T/custom/jdocs/lora/new_investigations")

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load_your_model():
    """Load your LoRA fine-tuned model."""
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.dataset import DatasetMetadata
    from gr00t.model.policy import Gr00tPolicy, EmbodimentTag
    from gr00t.experiment.data_config import So100DualCamDataConfig

    print("Loading YOUR model (LoRA)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    adapter_config_path = YOUR_CHECKPOINT / "adapter_config.json"
    with open(adapter_config_path) as f:
        lora_config = json.load(f)
    base_model_path = lora_config.get("base_model_name_or_path")

    base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
    base_model.eval()

    peft_model = PeftModel.from_pretrained(base_model, str(YOUR_CHECKPOINT))
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device=device)

    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.device = device
    policy._modality_config = modality_config
    policy._modality_transform = modality_transform
    policy.model = merged_model
    policy.model.action_head.num_inference_timesteps = 4

    metadata_path = YOUR_CHECKPOINT / "experiment_cfg" / "metadata.json"
    with open(metadata_path) as f:
        metadatas = json.load(f)

    embodiment_tag_enum = EmbodimentTag(EMBODIMENT_TAG)
    policy.embodiment_tag = embodiment_tag_enum

    metadata_dict = metadatas.get(embodiment_tag_enum.value)
    if metadata_dict:
        metadata = DatasetMetadata.model_validate(metadata_dict)
        policy._modality_transform.set_metadata(metadata)
        policy.metadata = metadata

    policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
    policy._video_horizon = len(policy._video_delta_indices)
    if "state" in modality_config:
        policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
        policy._state_horizon = len(policy._state_delta_indices)
    policy._action_delta_indices = np.array(modality_config["action"].delta_indices)
    policy._action_horizon = len(policy._action_delta_indices)

    print(f"  ✓ Your LoRA model loaded")
    return policy


def load_pushpakcc_model():
    """Load Pushpakcc's full fine-tuned model."""
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import So100DualCamDataConfig

    print("Loading PUSHPAKCC model (Full Finetune)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=str(PUSHPAKCC_CHECKPOINT),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    print(f"  ✓ Pushpakcc full-finetune model loaded")
    return policy


def load_dataset(dataset_path):
    """Load a dataset."""
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import So100DualCamDataConfig

    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend="torchvision_av",
    )

    return dataset


def get_trajectory_indices(dataset):
    """Get start indices for each trajectory."""
    # Parse episodes to get trajectory boundaries
    episodes_file = Path(dataset.dataset_path) / "meta" / "episodes.jsonl"

    traj_starts = [0]
    current_idx = 0

    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            current_idx += ep["length"]
            traj_starts.append(current_idx)

    return traj_starts[:-1]  # Remove last one (it's the total length)


def evaluate_trajectory(policy, dataset, start_idx, length=100):
    """Evaluate open-loop on a trajectory segment."""
    gt_actions = []
    pred_actions = []

    end_idx = min(start_idx + length, len(dataset))

    for idx in tqdm(range(start_idx, end_idx), desc="Evaluating trajectory"):
        sample = dataset[idx]

        obs = {
            "video.front": sample["video.front"],
            "video.wrist": sample["video.wrist"],
            "state.single_arm": sample["state.single_arm"],
            "state.gripper": sample["state.gripper"],
            "annotation.human.task_description": [sample.get("annotation.human.task_description", "pick up object")],
        }

        gt_arm = sample["action.single_arm"]
        gt_arm_np = gt_arm.numpy() if hasattr(gt_arm, 'numpy') else np.array(gt_arm)
        gt_actions.append(gt_arm_np[0])  # First action step

        with torch.no_grad():
            pred_action = policy.get_action(obs)

        pred_arm = pred_action["action.single_arm"]
        pred_actions.append(pred_arm[0])

    return np.array(gt_actions), np.array(pred_actions)


def plot_trajectory_comparison(gt_actions, pred_actions, title, output_path, mae):
    """Plot ground truth vs predicted actions for a trajectory."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{title}\nOpen-Loop MAE: {mae:.2f}°", fontsize=14, fontweight='bold')

    timesteps = np.arange(len(gt_actions))

    for j, (ax, joint_name) in enumerate(zip(axes, JOINT_NAMES)):
        gt = gt_actions[:, j]
        pred = pred_actions[:, j]

        # Calculate per-joint MAE
        joint_mae = np.abs(gt - pred).mean()

        ax.plot(timesteps, gt, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(timesteps, pred, 'r--', linewidth=2, label='Predicted', alpha=0.8)

        # Fill between to show error
        ax.fill_between(timesteps, gt, pred, alpha=0.2, color='red')

        ax.set_ylabel(f"{joint_name}\n(degrees)", fontsize=10)
        ax.set_title(f"{joint_name} - MAE: {joint_mae:.2f}°", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("OPEN-LOOP VISUALIZATION: Your Model vs Pushpakcc")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== YOUR MODEL ON YOUR DATASET ==========
    print("\n" + "=" * 70)
    print("[1] YOUR MODEL on YOUR DATASET")
    print("=" * 70)

    your_dataset = load_dataset(YOUR_DATASET)
    print(f"    Dataset: {YOUR_DATASET.name}, {len(your_dataset)} samples")

    your_policy = load_your_model()

    # Get trajectory start indices
    your_traj_starts = get_trajectory_indices(your_dataset)
    print(f"    Found {len(your_traj_starts)} trajectories")

    your_all_mae = []
    for i in range(min(NUM_TRAJECTORIES, len(your_traj_starts))):
        print(f"\n  Trajectory {i}:")
        start_idx = your_traj_starts[i]
        gt, pred = evaluate_trajectory(your_policy, your_dataset, start_idx, length=100)

        mae = np.abs(gt - pred).mean()
        your_all_mae.append(mae)
        print(f"    MAE: {mae:.2f}°")

        output_path = OUTPUT_DIR / f"your_model_openloop_traj{i}.png"
        plot_trajectory_comparison(
            gt, pred,
            f"Your LoRA Model - Trajectory {i}\n(Dataset: pick_and_place)",
            output_path,
            mae
        )

    your_avg_mae = np.mean(your_all_mae)
    print(f"\n  Your Model Average MAE: {your_avg_mae:.2f}°")

    # Free memory
    del your_policy
    torch.cuda.empty_cache()

    # ========== PUSHPAKCC MODEL ON THEIR DATASET ==========
    print("\n" + "=" * 70)
    print("[2] PUSHPAKCC MODEL on THEIR DATASET")
    print("=" * 70)

    ref_dataset = load_dataset(REF_DATASET)
    print(f"    Dataset: {REF_DATASET.name}, {len(ref_dataset)} samples")

    ref_policy = load_pushpakcc_model()

    # Get trajectory start indices
    ref_traj_starts = get_trajectory_indices(ref_dataset)
    print(f"    Found {len(ref_traj_starts)} trajectories")

    ref_all_mae = []
    for i in range(min(NUM_TRAJECTORIES, len(ref_traj_starts))):
        print(f"\n  Trajectory {i}:")
        start_idx = ref_traj_starts[i]
        gt, pred = evaluate_trajectory(ref_policy, ref_dataset, start_idx, length=100)

        mae = np.abs(gt - pred).mean()
        ref_all_mae.append(mae)
        print(f"    MAE: {mae:.2f}°")

        output_path = OUTPUT_DIR / f"pushpakcc_model_openloop_traj{i}.png"
        plot_trajectory_comparison(
            gt, pred,
            f"Pushpakcc Full Finetune Model - Trajectory {i}\n(Dataset: so101-table-cleanup)",
            output_path,
            mae
        )

    ref_avg_mae = np.mean(ref_all_mae)
    print(f"\n  Pushpakcc Model Average MAE: {ref_avg_mae:.2f}°")

    del ref_policy
    torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Your LoRA Model (on pick_and_place):     {your_avg_mae:.2f}°")
    print(f"  Pushpakcc Full Finetune (on so101-table-cleanup): {ref_avg_mae:.2f}°")
    print(f"\n  Output files saved to: {OUTPUT_DIR}")
    print("\n  Generated plots:")
    for i in range(NUM_TRAJECTORIES):
        print(f"    - your_model_openloop_traj{i}.png")
        print(f"    - pushpakcc_model_openloop_traj{i}.png")


if __name__ == "__main__":
    main()
