#!/usr/bin/env python3
"""
Trim stationary periods from LeRobot dataset.

This script addresses the data collection issue where episodes start with
a long stationary period (~7+ seconds) before actual movement begins.

The stationary period is caused by the time it takes to position the leader
arm after the recording has already started.

Usage:
    python custom/scripts/trim_stationary_periods.py \
        --input /path/to/original/dataset \
        --output /path/to/trimmed/dataset \
        --threshold 0.5 \
        --start_buffer 10 \
        --end_buffer 30
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def find_movement_boundaries(actions: np.ndarray, threshold: float = 0.5) -> tuple[int, int]:
    """
    Find the first and last frames with significant movement.

    Args:
        actions: Array of shape (num_frames, num_joints)
        threshold: Minimum change in degrees to count as movement

    Returns:
        Tuple of (first_movement_frame, last_movement_frame)
    """
    # Calculate frame-to-frame differences for all joints
    diffs = np.abs(np.diff(actions, axis=0))

    # Find max diff across all joints for each frame transition
    max_diffs = np.max(diffs, axis=1)

    # Find first and last significant movement
    movement_frames = np.where(max_diffs > threshold)[0]

    if len(movement_frames) == 0:
        # No significant movement found
        return 0, len(actions) - 1

    return movement_frames[0], movement_frames[-1] + 1


def trim_episode(
    episode_df: pd.DataFrame,
    threshold: float = 0.5,
    start_buffer: int = 10,
    end_buffer: int = 30,
) -> pd.DataFrame:
    """
    Trim stationary periods from an episode.

    Args:
        episode_df: DataFrame for a single episode
        threshold: Movement threshold in degrees
        start_buffer: Frames to keep before first movement
        end_buffer: Frames to keep after last movement

    Returns:
        Trimmed DataFrame with reindexed frame_index
    """
    actions = np.stack(episode_df["action"].values)

    first_move, last_move = find_movement_boundaries(actions, threshold)

    # Apply buffers
    start_idx = max(0, first_move - start_buffer)
    end_idx = min(len(episode_df), last_move + end_buffer)

    # Slice the episode
    trimmed = episode_df.iloc[start_idx:end_idx].copy()

    # Reindex frame_index to start from 0
    trimmed["frame_index"] = range(len(trimmed))

    return trimmed


def process_dataset(
    input_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    start_buffer: int = 10,
    end_buffer: int = 30,
) -> dict:
    """
    Process entire dataset, trimming stationary periods from each episode.

    Args:
        input_path: Path to input LeRobot dataset
        output_path: Path to output trimmed dataset
        threshold: Movement threshold in degrees
        start_buffer: Frames to keep before first movement
        end_buffer: Frames to keep after last movement

    Returns:
        Statistics dictionary
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    meta_input = input_path / "meta"
    meta_output = output_path / "meta"
    if meta_input.exists():
        shutil.copytree(meta_input, meta_output, dirs_exist_ok=True)

    # Copy videos directory structure (we'll update references later if needed)
    videos_input = input_path / "videos"
    if videos_input.exists():
        print("Note: Videos will need to be re-trimmed manually or re-encoded")
        # For now, just copy the videos - in production you'd re-encode them
        shutil.copytree(videos_input, output_path / "videos", dirs_exist_ok=True)

    # Load all parquet files
    data_input = input_path / "data" / "chunk-000"
    parquet_files = sorted(data_input.glob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet files")

    # Load all data
    dfs = []
    for f in parquet_files:
        table = pq.read_table(f)
        dfs.append(table.to_pandas())

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Get unique episodes
    episodes = sorted(df["episode_index"].unique())
    print(f"Number of episodes: {len(episodes)}")

    # Process each episode
    stats = {
        "original_frames": len(df),
        "trimmed_frames": 0,
        "episodes": [],
    }

    trimmed_dfs = []
    for ep_idx in episodes:
        ep_data = df[df["episode_index"] == ep_idx].sort_values("frame_index")

        original_len = len(ep_data)
        trimmed_ep = trim_episode(ep_data, threshold, start_buffer, end_buffer)
        trimmed_len = len(trimmed_ep)

        trimmed_dfs.append(trimmed_ep)

        ep_stats = {
            "episode": ep_idx,
            "original_frames": original_len,
            "trimmed_frames": trimmed_len,
            "removed_frames": original_len - trimmed_len,
            "removal_percent": (original_len - trimmed_len) / original_len * 100,
        }
        stats["episodes"].append(ep_stats)
        stats["trimmed_frames"] += trimmed_len

        print(
            f"Episode {ep_idx}: {original_len} -> {trimmed_len} frames "
            f"(-{original_len - trimmed_len}, -{ep_stats['removal_percent']:.1f}%)"
        )

    # Combine trimmed data
    trimmed_df = pd.concat(trimmed_dfs, ignore_index=True)
    trimmed_df["index"] = range(len(trimmed_df))

    # Save to output
    data_output = output_path / "data" / "chunk-000"
    data_output.mkdir(parents=True, exist_ok=True)

    # Save as single parquet file (or split as needed)
    output_parquet = data_output / "episode_all.parquet"
    trimmed_df.to_parquet(output_parquet, index=False)

    # Update info.json with new frame counts
    info_path = meta_output / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        info["total_frames"] = stats["trimmed_frames"]
        info["_trimming_applied"] = {
            "threshold": threshold,
            "start_buffer": start_buffer,
            "end_buffer": end_buffer,
            "original_frames": stats["original_frames"],
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # Summary stats
    total_removed = stats["original_frames"] - stats["trimmed_frames"]
    stats["total_removed"] = total_removed
    stats["removal_percent"] = total_removed / stats["original_frames"] * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original total frames: {stats['original_frames']}")
    print(f"Trimmed total frames: {stats['trimmed_frames']}")
    print(f"Removed frames: {total_removed} ({stats['removal_percent']:.1f}%)")
    print(f"Output saved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Trim stationary periods from LeRobot dataset"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input LeRobot dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output trimmed dataset",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Movement threshold in degrees (default: 0.5)",
    )
    parser.add_argument(
        "--start_buffer",
        type=int,
        default=10,
        help="Frames to keep before first movement (default: 10)",
    )
    parser.add_argument(
        "--end_buffer",
        type=int,
        default=30,
        help="Frames to keep after last movement (default: 30)",
    )

    args = parser.parse_args()

    stats = process_dataset(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        start_buffer=args.start_buffer,
        end_buffer=args.end_buffer,
    )

    # Save stats
    stats_path = args.output / "trimming_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
