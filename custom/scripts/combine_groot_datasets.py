#!/usr/bin/env python3
"""
GR00T Dataset Combination Script

Combines multiple GR00T-format datasets into a single unified dataset for multi-task training.
Unlike LeRobot's aggregate_datasets(), this script:
1. Works directly with GR00T v2 format (not LeRobot v3)
2. Preserves modality.json format
3. Properly reindexes episodes and tasks
4. Concatenates videos with correct naming

Usage:
    # Combine all datasets in a directory
    python combine_groot_datasets.py --input-dir /path/to/datasets --output /path/to/combined

    # Combine specific datasets
    python combine_groot_datasets.py --datasets /path/to/pick /path/to/place --output /path/to/combined

    # Dry run
    python combine_groot_datasets.py --input-dir /path/to/datasets --output /path/to/combined --dry-run

Author: GR00T Training Scripts
Date: 2025-12-01
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: Dict, indent: int = 2) -> None:
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, items: List[Dict]) -> None:
    """Save JSONL file."""
    with open(path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')


def get_dataset_info(dataset_path: Path) -> Dict:
    """Get info about a GR00T dataset."""
    meta_dir = dataset_path / "meta"

    if not meta_dir.exists():
        return None

    info_path = meta_dir / "info.json"
    if not info_path.exists():
        return None

    info = load_json(info_path)

    # Load tasks
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = []
    if tasks_path.exists():
        tasks = load_jsonl(tasks_path)

    # Load episodes
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = []
    if episodes_path.exists():
        episodes = load_jsonl(episodes_path)

    return {
        "path": dataset_path,
        "total_episodes": info.get("total_episodes", len(episodes)),
        "total_frames": info.get("total_frames", 0),
        "fps": info.get("fps", 5),
        "tasks": tasks,
        "episodes": episodes,
    }


def validate_datasets(datasets: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate that datasets can be combined."""
    issues = []

    if len(datasets) < 1:
        issues.append("Need at least 1 dataset")
        return False, issues

    # Check fps consistency
    fps_values = set(d["fps"] for d in datasets)
    if len(fps_values) > 1:
        issues.append(f"FPS mismatch: {fps_values}")

    # Check that all datasets have required files
    for ds in datasets:
        path = ds["path"]
        required = ["meta/info.json", "meta/modality.json", "meta/stats.json"]
        for req in required:
            if not (path / req).exists():
                issues.append(f"Missing {req} in {path.name}")

    return len(issues) == 0, issues


def combine_stats(datasets: List[Dict], output_path: Path) -> None:
    """Combine statistics from multiple datasets."""
    print("\n[3/6] Combining statistics...")

    all_stats = []
    for ds in datasets:
        stats_path = ds["path"] / "meta" / "stats.json"
        if stats_path.exists():
            all_stats.append(load_json(stats_path))

    if not all_stats:
        print("  Warning: No stats.json files found")
        return

    # For now, use weighted average based on frame count
    # More sophisticated: recompute from actual data
    total_frames = sum(ds["total_frames"] for ds in datasets)

    combined_stats = {}
    for key in all_stats[0].keys():
        if key not in all_stats[0] or not isinstance(all_stats[0][key], dict):
            continue

        combined_stats[key] = {}
        stat_types = ["min", "max", "mean", "std", "q01", "q99", "count"]

        for stat_type in stat_types:
            if stat_type not in all_stats[0][key]:
                continue

            values = []
            weights = []
            for i, stats in enumerate(all_stats):
                if key in stats and stat_type in stats[key]:
                    val = stats[key][stat_type]
                    if isinstance(val, list):
                        values.append(np.array(val))
                        weights.append(datasets[i]["total_frames"])

            if not values:
                continue

            if stat_type == "min":
                combined_stats[key][stat_type] = np.minimum.reduce(values).tolist()
            elif stat_type == "max":
                combined_stats[key][stat_type] = np.maximum.reduce(values).tolist()
            elif stat_type in ["mean", "q01", "q99"]:
                # Weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                combined_stats[key][stat_type] = (weighted_sum / total_frames).tolist()
            elif stat_type == "std":
                # Approximate combined std (not exact but reasonable)
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                combined_stats[key][stat_type] = (weighted_sum / total_frames).tolist()
            elif stat_type == "count":
                combined_stats[key][stat_type] = sum(values).tolist()

    save_json(output_path / "meta" / "stats.json", combined_stats)
    print(f"  Created: stats.json (combined from {len(all_stats)} datasets)")


def combine_datasets(datasets: List[Dict], output_path: Path, dry_run: bool = False) -> bool:
    """Combine multiple GR00T datasets."""

    print("\n" + "=" * 60)
    print("GR00T Dataset Combination")
    print("=" * 60)

    # Summary
    total_episodes = sum(ds["total_episodes"] for ds in datasets)
    total_frames = sum(ds["total_frames"] for ds in datasets)
    all_tasks = []
    for ds in datasets:
        all_tasks.extend(ds["tasks"])

    print(f"\nInput datasets: {len(datasets)}")
    for ds in datasets:
        task_names = [t.get("task", "?") for t in ds["tasks"]]
        print(f"  - {ds['path'].name}: {ds['total_episodes']} episodes, tasks: {task_names}")

    print(f"\nCombined:")
    print(f"  - Total episodes: {total_episodes}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Total tasks: {len(all_tasks)}")
    print(f"  - Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] No files modified")
        return True

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "videos").mkdir(exist_ok=True)

    # Step 1: Copy modality.json (same for all datasets)
    print("\n[1/6] Copying modality.json...")
    modality_src = datasets[0]["path"] / "meta" / "modality.json"
    shutil.copy2(modality_src, output_path / "meta" / "modality.json")
    print(f"  Copied from: {datasets[0]['path'].name}")

    # Step 2: Combine tasks with new indices
    print("\n[2/6] Combining tasks...")
    combined_tasks = []
    task_mapping = {}  # (dataset_idx, old_task_idx) -> new_task_idx

    for ds_idx, ds in enumerate(datasets):
        for task in ds["tasks"]:
            old_idx = task["task_index"]
            new_idx = len(combined_tasks)
            task_mapping[(ds_idx, old_idx)] = new_idx

            combined_tasks.append({
                "task_index": new_idx,
                "task": task["task"],
            })
            print(f"  Task {new_idx}: '{task['task']}' (from {ds['path'].name})")

    save_jsonl(output_path / "meta" / "tasks.jsonl", combined_tasks)

    # Step 3: Combine statistics
    combine_stats(datasets, output_path)

    # Step 4: Combine episodes with new indices
    print("\n[4/6] Combining episodes...")
    combined_episodes = []
    episode_offset = 0

    for ds_idx, ds in enumerate(datasets):
        for ep in ds["episodes"]:
            old_ep_idx = ep["episode_index"]
            new_ep_idx = episode_offset + old_ep_idx

            old_task_idx = ep.get("task_index", 0)
            new_task_idx = task_mapping.get((ds_idx, old_task_idx), 0)

            combined_episodes.append({
                "episode_index": new_ep_idx,
                "length": ep["length"],
                "task_index": new_task_idx,
            })

        episode_offset += ds["total_episodes"]

    save_jsonl(output_path / "meta" / "episodes.jsonl", combined_episodes)
    print(f"  Created: episodes.jsonl ({len(combined_episodes)} episodes)")

    # Step 5: Copy and rename data files
    print("\n[5/6] Copying data files...")
    episode_offset = 0

    for ds_idx, ds in enumerate(datasets):
        data_dir = ds["path"] / "data"
        if not data_dir.exists():
            print(f"  Warning: No data/ in {ds['path'].name}")
            continue

        # Find parquet files
        parquet_files = sorted(data_dir.glob("**/*.parquet"))

        for pq_file in parquet_files:
            # Parse episode index from filename
            # Format: episode_000000.parquet or chunk-000/episode_000.parquet
            name = pq_file.stem
            if "episode" in name:
                try:
                    old_ep_idx = int(name.split("_")[-1])
                    new_ep_idx = episode_offset + old_ep_idx
                    new_name = f"episode_{new_ep_idx:06d}.parquet"
                    shutil.copy2(pq_file, output_path / "data" / new_name)
                except ValueError:
                    # Just copy with original name if can't parse
                    shutil.copy2(pq_file, output_path / "data" / pq_file.name)

        episode_offset += ds["total_episodes"]

    data_files = list((output_path / "data").glob("*.parquet"))
    print(f"  Copied {len(data_files)} parquet files")

    # Step 6: Copy and rename video files
    print("\n[6/6] Copying video files...")
    episode_offset = 0

    for ds_idx, ds in enumerate(datasets):
        videos_dir = ds["path"] / "videos"
        if not videos_dir.exists():
            print(f"  Warning: No videos/ in {ds['path'].name}")
            continue

        video_files = sorted(videos_dir.glob("*.mp4"))

        for video_file in video_files:
            name = video_file.stem
            # Parse: observation.images.front_episode_000000.mp4
            if "episode_" in name:
                try:
                    parts = name.split("episode_")
                    prefix = parts[0]  # e.g., "observation.images.front_"
                    old_ep_idx = int(parts[1])
                    new_ep_idx = episode_offset + old_ep_idx
                    new_name = f"{prefix}episode_{new_ep_idx:06d}.mp4"
                    shutil.copy2(video_file, output_path / "videos" / new_name)
                except (ValueError, IndexError):
                    shutil.copy2(video_file, output_path / "videos" / video_file.name)

        episode_offset += ds["total_episodes"]

    video_files = list((output_path / "videos").glob("*.mp4"))
    print(f"  Copied {len(video_files)} video files")

    # Step 7: Create info.json
    print("\n[7/6] Creating info.json...")
    info = {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(combined_tasks),
        "fps": datasets[0]["fps"],
        "robot_type": "so101_follower",
        "source_datasets": [str(ds["path"]) for ds in datasets],
    }
    save_json(output_path / "meta" / "info.json", info)

    print("\n" + "=" * 60)
    print("SUCCESS: Datasets combined")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {total_frames}")
    print(f"Tasks: {len(combined_tasks)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple GR00T datasets for multi-task training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=Path,
        help="Specific dataset paths to combine"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        help="Directory containing multiple datasets to combine"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for combined dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be combined without actually combining"
    )

    args = parser.parse_args()

    # Get datasets
    dataset_paths = []

    if args.datasets:
        dataset_paths = args.datasets
    elif args.input_dir:
        # Find all subdirectories with meta/info.json
        for subdir in sorted(args.input_dir.iterdir()):
            if subdir.is_dir() and (subdir / "meta" / "info.json").exists():
                dataset_paths.append(subdir)
    else:
        print("Error: Must specify --datasets or --input-dir")
        sys.exit(1)

    if not dataset_paths:
        print("No datasets found")
        sys.exit(1)

    # Get info for each dataset
    datasets = []
    for path in dataset_paths:
        info = get_dataset_info(path)
        if info:
            datasets.append(info)
        else:
            print(f"Warning: Skipping invalid dataset: {path}")

    # Validate
    valid, issues = validate_datasets(datasets)
    if not valid:
        print("Validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    # Check output
    if args.output.exists() and not args.dry_run:
        response = input(f"Output {args.output} exists. Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborting")
            sys.exit(0)
        shutil.rmtree(args.output)

    # Combine
    success = combine_datasets(datasets, args.output, args.dry_run)

    if success:
        print("\nNext steps:")
        print(f"  1. Verify: python custom/scripts/verify_groot_training_setup.py --dataset {args.output}")
        print(f"  2. Train: bash custom/scripts/train_groot_mini_mvp.sh")
    else:
        print("\nFailed to combine datasets")
        sys.exit(1)


if __name__ == "__main__":
    main()
