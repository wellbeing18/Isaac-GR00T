"""
Horizontal Data Augmentation for SO-101 Pick & Place Dataset

This script creates horizontally mirrored versions of episodes to balance
the directional bias in the dataset (76% LEFT -> 50% balanced).

Mirroring rules for SO-101 arm:
- shoulder_pan: NEGATE (rotation around vertical axis)
- wrist_roll: NEGATE (rotation around horizontal axis)
- shoulder_lift, elbow_flex, wrist_flex: KEEP SAME (vertical plane motion)
- gripper: KEEP SAME (open/close is symmetric)
- cameras: flip horizontal axis (width dimension)

Usage:
    # Test on 2 episodes first
    python augment_dataset_horizontal.py \
        --input-dataset /path/to/original \
        --output-dataset /path/to/augmented \
        --max-episodes 2 \
        --verify

    # Full augmentation after verification
    python augment_dataset_horizontal.py \
        --input-dataset /path/to/original \
        --output-dataset /path/to/augmented
"""

import argparse
import json
import copy
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd


# Joint indices for SO-101
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHOULDER_PAN_IDX = 0
WRIST_ROLL_IDX = 4
GRIPPER_IDX = 5

# Joints to negate for horizontal mirroring
JOINTS_TO_NEGATE = [SHOULDER_PAN_IDX, WRIST_ROLL_IDX]


def mirror_joint_data(data: np.ndarray) -> np.ndarray:
    """
    Mirror joint data by negating horizontal-related joints.

    Args:
        data: Array of shape (N, 6) with joint values

    Returns:
        Mirrored array of same shape
    """
    mirrored = data.copy()
    for idx in JOINTS_TO_NEGATE:
        mirrored[:, idx] *= -1
    return mirrored


def flip_video_horizontal(input_path: Path, output_path: Path) -> bool:
    """
    Flip a video horizontally using ffmpeg.

    Args:
        input_path: Path to input video
        output_path: Path to output video

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "hflip",
        "-c:v", "libsvtav1",  # Use AV1 codec to match original
        "-crf", "30",
        "-preset", "6",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"FFmpeg error for {input_path}: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"FFmpeg timeout for {input_path}")
        return False
    except Exception as e:
        print(f"FFmpeg exception for {input_path}: {e}")
        return False


def mirror_episode_parquet(
    input_path: Path,
    output_path: Path,
    new_episode_index: int,
    original_episode_index: int
) -> Dict[str, Any]:
    """
    Mirror a parquet episode file.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        new_episode_index: New episode index for the mirrored episode
        original_episode_index: Original episode index (for metadata)

    Returns:
        Dictionary with verification data
    """
    df = pd.read_parquet(input_path)

    # Store original values for verification
    original_action = np.vstack(df['action'].values)
    original_state = np.vstack(df['observation.state'].values)

    # Mirror action and state
    mirrored_action = mirror_joint_data(original_action)
    mirrored_state = mirror_joint_data(original_state)

    # Create new dataframe
    mirrored_df = df.copy()
    mirrored_df['action'] = [row for row in mirrored_action]
    mirrored_df['observation.state'] = [row for row in mirrored_state]
    mirrored_df['episode_index'] = new_episode_index

    # Update global index
    base_index = new_episode_index * 10000  # Arbitrary large offset
    mirrored_df['index'] = range(base_index, base_index + len(df))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mirrored_df.to_parquet(output_path)

    # Return verification data
    return {
        "original_episode": original_episode_index,
        "mirrored_episode": new_episode_index,
        "num_frames": len(df),
        "original_shoulder_pan_range": [float(original_action[:, 0].min()), float(original_action[:, 0].max())],
        "mirrored_shoulder_pan_range": [float(mirrored_action[:, 0].min()), float(mirrored_action[:, 0].max())],
        "original_wrist_roll_range": [float(original_action[:, 4].min()), float(original_action[:, 4].max())],
        "mirrored_wrist_roll_range": [float(mirrored_action[:, 4].min()), float(mirrored_action[:, 4].max())],
        # These should be unchanged
        "original_shoulder_lift_range": [float(original_action[:, 1].min()), float(original_action[:, 1].max())],
        "mirrored_shoulder_lift_range": [float(mirrored_action[:, 1].min()), float(mirrored_action[:, 1].max())],
    }


def verify_mirroring(verification_data: List[Dict]) -> bool:
    """
    Verify that mirroring was done correctly.

    Returns:
        True if all verifications pass
    """
    all_passed = True

    for v in verification_data:
        ep_orig = v["original_episode"]
        ep_mir = v["mirrored_episode"]

        # Check shoulder_pan was negated
        orig_min, orig_max = v["original_shoulder_pan_range"]
        mir_min, mir_max = v["mirrored_shoulder_pan_range"]

        expected_mir_min = -orig_max
        expected_mir_max = -orig_min

        if not (abs(mir_min - expected_mir_min) < 0.01 and abs(mir_max - expected_mir_max) < 0.01):
            print(f"FAIL: Episode {ep_orig}->{ep_mir} shoulder_pan not correctly negated")
            print(f"  Original: [{orig_min:.2f}, {orig_max:.2f}]")
            print(f"  Mirrored: [{mir_min:.2f}, {mir_max:.2f}]")
            print(f"  Expected: [{expected_mir_min:.2f}, {expected_mir_max:.2f}]")
            all_passed = False
        else:
            print(f"OK: Episode {ep_orig}->{ep_mir} shoulder_pan correctly negated")
            print(f"  [{orig_min:.2f}, {orig_max:.2f}] -> [{mir_min:.2f}, {mir_max:.2f}]")

        # Check wrist_roll was negated
        orig_min, orig_max = v["original_wrist_roll_range"]
        mir_min, mir_max = v["mirrored_wrist_roll_range"]

        expected_mir_min = -orig_max
        expected_mir_max = -orig_min

        if not (abs(mir_min - expected_mir_min) < 0.01 and abs(mir_max - expected_mir_max) < 0.01):
            print(f"FAIL: Episode {ep_orig}->{ep_mir} wrist_roll not correctly negated")
            all_passed = False
        else:
            print(f"OK: Episode {ep_orig}->{ep_mir} wrist_roll correctly negated")

        # Check shoulder_lift was NOT changed
        orig_range = v["original_shoulder_lift_range"]
        mir_range = v["mirrored_shoulder_lift_range"]

        if orig_range != mir_range:
            print(f"FAIL: Episode {ep_orig}->{ep_mir} shoulder_lift was incorrectly modified")
            all_passed = False
        else:
            print(f"OK: Episode {ep_orig}->{ep_mir} shoulder_lift unchanged")

    return all_passed


def update_metadata(
    input_dataset: Path,
    output_dataset: Path,
    num_original_episodes: int,
    num_augmented_episodes: int,
    augmentation_mapping: Dict[int, int]
):
    """
    Update metadata files for the augmented dataset.
    """
    # Copy and update info.json
    info_path = input_dataset / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = num_original_episodes + num_augmented_episodes
    info["total_episodes"] = total_episodes
    info["total_videos"] = total_episodes * 2  # head + wrist
    info["splits"]["train"] = f"0:{total_episodes}"

    output_meta = output_dataset / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    with open(output_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Copy modality.json (unchanged)
    shutil.copy(input_dataset / "meta" / "modality.json", output_meta / "modality.json")

    # Copy tasks.jsonl (unchanged)
    shutil.copy(input_dataset / "meta" / "tasks.jsonl", output_meta / "tasks.jsonl")

    # Copy stats files (will be regenerated later for accuracy, but needed for loader)
    for stats_file in ["stats.json", "episodes_stats.jsonl", "relative_stats.json"]:
        src = input_dataset / "meta" / stats_file
        if src.exists():
            shutil.copy(src, output_meta / stats_file)

    # Update episodes.jsonl
    episodes_path = input_dataset / "meta" / "episodes.jsonl"
    output_episodes = []

    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                output_episodes.append(json.loads(line))

    # Add mirrored episodes
    for orig_idx, mir_idx in augmentation_mapping.items():
        # Find original episode data
        orig_data = None
        for ep in output_episodes:
            if ep.get("episode_index") == orig_idx:
                orig_data = ep
                break

        if orig_data:
            mir_data = copy.deepcopy(orig_data)
            mir_data["episode_index"] = mir_idx
            mir_data["mirrored_from"] = orig_idx  # Track source
            output_episodes.append(mir_data)

    with open(output_meta / "episodes.jsonl", "w") as f:
        for ep in output_episodes:
            f.write(json.dumps(ep) + "\n")

    # Save augmentation metadata
    aug_meta = {
        "augmentation_type": "horizontal_mirror",
        "source_dataset": str(input_dataset),
        "original_episodes": num_original_episodes,
        "augmented_episodes": num_augmented_episodes,
        "total_episodes": total_episodes,
        "joints_negated": [JOINT_NAMES[i] for i in JOINTS_TO_NEGATE],
        "mapping": {str(k): v for k, v in augmentation_mapping.items()}
    }

    with open(output_meta / "augmentation_info.json", "w") as f:
        json.dump(aug_meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Horizontal data augmentation for SO-101")
    parser.add_argument("--input-dataset", type=Path, required=True, help="Path to input dataset")
    parser.add_argument("--output-dataset", type=Path, required=True, help="Path to output dataset")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to process (for testing)")
    parser.add_argument("--verify", action="store_true", help="Run verification after augmentation")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video processing (for testing)")
    args = parser.parse_args()

    input_dataset = args.input_dataset
    output_dataset = args.output_dataset

    if not input_dataset.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dataset}")

    # Load info
    with open(input_dataset / "meta" / "info.json") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    if args.max_episodes:
        total_episodes = min(total_episodes, args.max_episodes)

    print(f"Input dataset: {input_dataset}")
    print(f"Output dataset: {output_dataset}")
    print(f"Processing {total_episodes} episodes")
    print()

    # Create output directory structure
    output_dataset.mkdir(parents=True, exist_ok=True)
    (output_dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dataset / "videos" / "chunk-000" / "observation.images.head").mkdir(parents=True, exist_ok=True)
    (output_dataset / "videos" / "chunk-000" / "observation.images.left_wrist").mkdir(parents=True, exist_ok=True)

    verification_data = []
    augmentation_mapping = {}

    for ep_idx in range(total_episodes):
        print(f"Processing episode {ep_idx}/{total_episodes}...")

        # Input paths
        parquet_in = input_dataset / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        head_video_in = input_dataset / "videos" / "chunk-000" / "observation.images.head" / f"episode_{ep_idx:06d}.mp4"
        wrist_video_in = input_dataset / "videos" / "chunk-000" / "observation.images.left_wrist" / f"episode_{ep_idx:06d}.mp4"

        # First copy original to output
        parquet_out_orig = output_dataset / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        shutil.copy(parquet_in, parquet_out_orig)

        if not args.skip_videos:
            head_video_out_orig = output_dataset / "videos" / "chunk-000" / "observation.images.head" / f"episode_{ep_idx:06d}.mp4"
            wrist_video_out_orig = output_dataset / "videos" / "chunk-000" / "observation.images.left_wrist" / f"episode_{ep_idx:06d}.mp4"
            shutil.copy(head_video_in, head_video_out_orig)
            shutil.copy(wrist_video_in, wrist_video_out_orig)

        # Then create mirrored version
        mir_ep_idx = total_episodes + ep_idx  # Mirrored episodes start after originals

        # Mirror parquet
        parquet_out_mir = output_dataset / "data" / "chunk-000" / f"episode_{mir_ep_idx:06d}.parquet"
        v_data = mirror_episode_parquet(parquet_in, parquet_out_mir, mir_ep_idx, ep_idx)
        verification_data.append(v_data)
        augmentation_mapping[ep_idx] = mir_ep_idx

        # Mirror videos
        if not args.skip_videos:
            head_video_out_mir = output_dataset / "videos" / "chunk-000" / "observation.images.head" / f"episode_{mir_ep_idx:06d}.mp4"
            wrist_video_out_mir = output_dataset / "videos" / "chunk-000" / "observation.images.left_wrist" / f"episode_{mir_ep_idx:06d}.mp4"

            print(f"  Flipping head video...")
            if not flip_video_horizontal(head_video_in, head_video_out_mir):
                print(f"  WARNING: Failed to flip head video for episode {ep_idx}")

            print(f"  Flipping wrist video...")
            if not flip_video_horizontal(wrist_video_in, wrist_video_out_mir):
                print(f"  WARNING: Failed to flip wrist video for episode {ep_idx}")

    # Update metadata
    print("\nUpdating metadata...")
    update_metadata(input_dataset, output_dataset, total_episodes, total_episodes, augmentation_mapping)

    # Verification
    if args.verify:
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)

        if verify_mirroring(verification_data):
            print("\n✓ All verifications PASSED")
        else:
            print("\n✗ Some verifications FAILED - check output above")

    # Save verification data
    with open(output_dataset / "meta" / "verification_data.json", "w") as f:
        json.dump(verification_data, f, indent=2)

    print(f"\nDone! Output dataset: {output_dataset}")
    print(f"  Original episodes: 0-{total_episodes-1}")
    print(f"  Mirrored episodes: {total_episodes}-{total_episodes*2-1}")
    print(f"  Total episodes: {total_episodes*2}")


if __name__ == "__main__":
    main()
