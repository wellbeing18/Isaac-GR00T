#!/usr/bin/env python3
"""
Compare Model Outputs: Your LoRA Model vs Pushpakcc Full Finetune
==================================================================
This script compares inference outputs between:
1. Your LoRA fine-tuned model
2. Pushpakcc's full fine-tuned model (reference)

Key differences:
- Your model: LoRA adapters on base model
- Pushpakcc: Full model fine-tuning (no LoRA)
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

import json
import torch
import numpy as np
from pathlib import Path

# Configuration
YOUR_CHECKPOINT = Path("/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914")
PUSHPAKCC_CHECKPOINT = Path("/tmp/pushpakcc_model/checkpoint-1000")
YOUR_DATASET = Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
EMBODIMENT_TAG = "new_embodiment"


def load_your_model():
    """Load your LoRA fine-tuned model using the custom loading logic."""
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.dataset import DatasetMetadata
    from gr00t.model.policy import Gr00tPolicy, EmbodimentTag
    from gr00t.experiment.data_config import So100DualCamDataConfig

    print("Loading YOUR model (LoRA)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Get modality config and transform from data config
    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Load adapter config
    adapter_config_path = YOUR_CHECKPOINT / "adapter_config.json"
    with open(adapter_config_path) as f:
        lora_config = json.load(f)
    base_model_path = lora_config.get("base_model_name_or_path")

    print(f"  Base model: {base_model_path}")
    print(f"  LoRA rank: {lora_config.get('r')}")

    # Load base model
    print(f"  Loading base GR00T model...")
    base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
    base_model.eval()

    # Load PEFT adapter
    print(f"  Loading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(YOUR_CHECKPOINT))

    # Merge weights
    print(f"  Merging LoRA weights...")
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device=device)

    # Create policy wrapper manually
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.device = device
    policy._modality_config = modality_config
    policy._modality_transform = modality_transform
    policy.model = merged_model
    policy.model.action_head.num_inference_timesteps = 4

    # Load metadata
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
        print(f"  ✓ Loaded normalization stats for '{embodiment_tag_enum.value}'")

    # Load horizons from modality config
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
    from gr00t.model.policy import Gr00tPolicy, EmbodimentTag
    from gr00t.experiment.data_config import So100DualCamDataConfig
    from gr00t.data.dataset import DatasetMetadata

    print("Loading PUSHPAKCC model (Full Finetune)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get modality config and transform from data config
    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Load from checkpoint directly (full model weights)
    # Note: Need to use the full checkpoint path for full-finetuned models
    policy = Gr00tPolicy(
        model_path=str(PUSHPAKCC_CHECKPOINT),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    print(f"  ✓ Pushpakcc full-finetune model loaded")
    return policy


def load_sample(sample_idx=0):
    """Load a sample from your dataset."""
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import So100DualCamDataConfig

    data_config = So100DualCamDataConfig()
    modality_config = data_config.modality_config()

    dataset = LeRobotSingleDataset(
        dataset_path=YOUR_DATASET,
        modality_configs=modality_config,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend="torchvision_av",
    )

    return dataset[sample_idx]


def compare_outputs():
    print("=" * 70)
    print("MODEL OUTPUT COMPARISON")
    print("Your LoRA Model vs Pushpakcc Full Finetune")
    print("=" * 70)

    # Load sample
    print("\n[1] Loading test sample from your dataset...")
    sample = load_sample(0)
    print(f"    State: {sample['state.single_arm'][0]}")
    print(f"    Task: {sample.get('annotation.human.task_description', 'N/A')}")

    # Prepare observation
    obs = {
        "video.front": sample["video.front"],
        "video.wrist": sample["video.wrist"],
        "state.single_arm": sample["state.single_arm"],
        "state.gripper": sample["state.gripper"],
        "annotation.human.task_description": [sample.get("annotation.human.task_description", "pick up object")],
    }

    # Ground truth
    gt_arm = sample["action.single_arm"]

    # Load and run your model
    print("\n[2] Running inference with YOUR model...")
    your_policy = load_your_model()

    with torch.no_grad():
        your_action = your_policy.get_action(obs)

    your_arm = your_action["action.single_arm"]
    your_grip = your_action["action.gripper"]
    print(f"    Your action (arm, step 0): {your_arm[0]}")
    print(f"    Your action (gripper, step 0): {your_grip[0]}")

    # Calculate error vs ground truth
    gt_arm_np = gt_arm[0].numpy() if hasattr(gt_arm[0], 'numpy') else gt_arm[0]
    your_error = np.abs(your_arm[0] - gt_arm_np).mean()
    print(f"    Your MAE vs GT: {your_error:.4f} degrees")

    # Free memory
    del your_policy
    torch.cuda.empty_cache()

    # Load and run Pushpakcc model
    print("\n[3] Running inference with PUSHPAKCC model...")
    try:
        ref_policy = load_pushpakcc_model()

        with torch.no_grad():
            ref_action = ref_policy.get_action(obs)

        ref_arm = ref_action["action.single_arm"]
        ref_grip = ref_action["action.gripper"]
        print(f"    Ref action (arm, step 0): {ref_arm[0]}")
        print(f"    Ref action (gripper, step 0): {ref_grip[0]}")

        ref_error = np.abs(ref_arm[0] - gt_arm_np).mean()
        print(f"    Ref MAE vs GT: {ref_error:.4f} degrees")

        # Compare models
        print("\n[4] COMPARISON: Your Model vs Pushpakcc")
        print("-" * 70)

        diff = np.abs(your_arm - ref_arm)
        print(f"    Mean absolute difference (arm): {diff.mean():.4f} degrees")
        print(f"    Max absolute difference (arm): {diff.max():.4f} degrees")

        # Per-joint comparison
        print("\n    Per-joint comparison (step 0):")
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        for i, name in enumerate(joint_names):
            y = your_arm[0, i]
            r = ref_arm[0, i]
            g = gt_arm_np[i].item() if hasattr(gt_arm_np[i], 'item') else gt_arm_np[i]
            print(f"      {name:15s}: Your={y:8.2f}°, Ref={r:8.2f}°, GT={g:8.2f}°")

        del ref_policy

    except Exception as e:
        print(f"    ✗ Failed to load Pushpakcc model: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compare_outputs()
