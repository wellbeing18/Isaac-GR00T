#!/usr/bin/env python3
"""
Fair Open-Loop MAE Comparison
==============================
Compare:
1. Your model on YOUR dataset
2. Pushpakcc model on THEIR dataset (youliangtan/so101-table-cleanup)

This gives a fair comparison of model quality.
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
YOUR_CHECKPOINT = Path("/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914")
PUSHPAKCC_CHECKPOINT = Path("/tmp/pushpakcc_model/checkpoint-1000")
YOUR_DATASET = Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
REF_DATASET = Path("/tmp/so101-table-cleanup")
EMBODIMENT_TAG = "new_embodiment"
NUM_SAMPLES = 100  # Number of samples to evaluate


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


def evaluate_openloop(policy, dataset, num_samples):
    """Evaluate open-loop MAE on a dataset."""
    errors = []
    joint_errors = {i: [] for i in range(5)}  # 5 arm joints

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[int(idx)]

        obs = {
            "video.front": sample["video.front"],
            "video.wrist": sample["video.wrist"],
            "state.single_arm": sample["state.single_arm"],
            "state.gripper": sample["state.gripper"],
            "annotation.human.task_description": [sample.get("annotation.human.task_description", "pick up object")],
        }

        gt_arm = sample["action.single_arm"]
        gt_arm_np = gt_arm.numpy() if hasattr(gt_arm, 'numpy') else gt_arm

        with torch.no_grad():
            pred_action = policy.get_action(obs)

        pred_arm = pred_action["action.single_arm"]

        # MAE for first action step
        error = np.abs(pred_arm[0] - gt_arm_np[0])
        errors.append(error.mean())

        for j in range(5):
            joint_errors[j].append(error[j])

    return {
        "mae": np.mean(errors),
        "std": np.std(errors),
        "joint_mae": {j: np.mean(joint_errors[j]) for j in range(5)},
        "joint_std": {j: np.std(joint_errors[j]) for j in range(5)},
    }


def main():
    print("=" * 70)
    print("FAIR OPEN-LOOP MAE COMPARISON")
    print("=" * 70)

    # Evaluate YOUR model on YOUR dataset
    print("\n" + "=" * 70)
    print("[1] YOUR MODEL on YOUR DATASET")
    print("=" * 70)

    your_dataset = load_dataset(YOUR_DATASET)
    print(f"    Dataset: {YOUR_DATASET.name}")
    print(f"    Samples: {len(your_dataset)}")

    your_policy = load_your_model()
    your_results = evaluate_openloop(your_policy, your_dataset, NUM_SAMPLES)

    print(f"\n    Results:")
    print(f"    MAE: {your_results['mae']:.4f}° ± {your_results['std']:.4f}°")
    print(f"    Per-joint MAE:")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for j, name in enumerate(joint_names):
        print(f"      {name:15s}: {your_results['joint_mae'][j]:.4f}° ± {your_results['joint_std'][j]:.4f}°")

    # Free memory
    del your_policy
    torch.cuda.empty_cache()

    # Evaluate PUSHPAKCC model on THEIR dataset
    print("\n" + "=" * 70)
    print("[2] PUSHPAKCC MODEL on THEIR DATASET")
    print("=" * 70)

    ref_dataset = load_dataset(REF_DATASET)
    print(f"    Dataset: {REF_DATASET.name}")
    print(f"    Samples: {len(ref_dataset)}")

    ref_policy = load_pushpakcc_model()
    ref_results = evaluate_openloop(ref_policy, ref_dataset, NUM_SAMPLES)

    print(f"\n    Results:")
    print(f"    MAE: {ref_results['mae']:.4f}° ± {ref_results['std']:.4f}°")
    print(f"    Per-joint MAE:")
    for j, name in enumerate(joint_names):
        print(f"      {name:15s}: {ref_results['joint_mae'][j]:.4f}° ± {ref_results['joint_std'][j]:.4f}°")

    del ref_policy
    torch.cuda.empty_cache()

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Dataset':<25} {'MAE':<15}")
    print("-" * 70)
    print(f"{'Your LoRA Model':<30} {'pick_and_place':<25} {your_results['mae']:.4f}° ± {your_results['std']:.4f}°")
    print(f"{'Pushpakcc Full Finetune':<30} {'so101-table-cleanup':<25} {ref_results['mae']:.4f}° ± {ref_results['std']:.4f}°")

    print("\n" + "=" * 70)
    print("Per-Joint Comparison:")
    print("=" * 70)
    print(f"\n{'Joint':<15} {'Your Model':<20} {'Pushpakcc':<20}")
    print("-" * 55)
    for j, name in enumerate(joint_names):
        your_j = f"{your_results['joint_mae'][j]:.2f}° ± {your_results['joint_std'][j]:.2f}°"
        ref_j = f"{ref_results['joint_mae'][j]:.2f}° ± {ref_results['joint_std'][j]:.2f}°"
        print(f"{name:<15} {your_j:<20} {ref_j:<20}")

    # Save results
    results = {
        "your_model": {
            "dataset": str(YOUR_DATASET),
            "checkpoint": str(YOUR_CHECKPOINT),
            "mae": float(your_results['mae']),
            "std": float(your_results['std']),
            "joint_mae": {j: float(v) for j, v in your_results['joint_mae'].items()},
        },
        "pushpakcc_model": {
            "dataset": str(REF_DATASET),
            "checkpoint": str(PUSHPAKCC_CHECKPOINT),
            "mae": float(ref_results['mae']),
            "std": float(ref_results['std']),
            "joint_mae": {j: float(v) for j, v in ref_results['joint_mae'].items()},
        },
    }

    output_path = Path("/home/jrobot/project/Isaac-GR00T/custom/jdocs/lora/new_investigations/openloop_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
