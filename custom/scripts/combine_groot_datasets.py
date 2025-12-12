#!/usr/bin/env python3
"""
GR00T Dataset Combination Script

Combines multiple GR00T-format datasets into a single unified dataset for multi-task training.
Unlike LeRobot's aggregate_datasets(), this script:
1. Works directly with GR00T v2 format (not LeRobot v3)
2. Preserves modality.json format
3. Properly reindexes episodes and tasks
4. Copies per-episode video files with correct renaming (no concatenation)

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
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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


def get_dataset_info(dataset_path: Path) -> Optional[Dict]:
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
        "chunk_size": info.get("chunks_size", 1000),
        "tasks": tasks,
        "episodes": episodes,
        "modality": load_json(meta_dir / "modality.json") if (meta_dir / "modality.json").exists() else {}
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
    
    # Calculate offsets based on chunk alignment
    # We need to ensure that if we use chunked videos, the new episodes map to the correct new chunks.
    # Current strategy: We preserve CHUNKS, not just episodes.
    
    # To allow physically merging datasets without re-encoding videos, 
    # we must maintain the alignment: (episode_index // chunk_size) corresponds to a specific video file.
    #
    # Algorithm:
    # 1. Get chunk_size (assume constant across datasets for now, or use max)
    # 2. For each dataset:
    #    - Calculate how many chunks it occupies: num_chunks = ceil(total_episodes / chunk_size)
    #    - The next dataset must start at: current_chunk_offset * chunk_size
    #    - This creates gaps in episode indices, but that is allowed.
    
    # Assume chunk_size from first dataset
    chunk_size = datasets[0]["chunk_size"]
    current_chunk_offset = 0
    
    dataset_offsets = [] # Store (start_episode_index, chunk_offset) for each dataset

    for ds_idx, ds in enumerate(datasets):
        # Ensure consistent chunk size
        if ds["chunk_size"] != chunk_size:
             print(f"  Warning: Dataset {ds['path'].name} has different chunk_size {ds['chunk_size']} vs {chunk_size}. Using {chunk_size}.")
        
        # Calculate start episode index for this dataset
        start_ep_idx = current_chunk_offset * chunk_size
        dataset_offsets.append((start_ep_idx, current_chunk_offset))
        
        print(f"  Dataset {ds['path'].name}: starts at episode {start_ep_idx} (Chunk {current_chunk_offset})")
        
        # Determine how many chunks this dataset uses
        # We need to find the max episode index to know how many chunks are used
        max_ep_idx = 0
        if ds["episodes"]:
            max_ep_idx = max(ep["episode_index"] for ep in ds["episodes"])
        else:
            max_ep_idx = ds["total_episodes"] - 1
            
        used_chunks = (max_ep_idx // chunk_size) + 1
        current_chunk_offset += used_chunks

        for ep in ds["episodes"]:
            old_ep_idx = ep["episode_index"]
            new_ep_idx = start_ep_idx + old_ep_idx

            old_task_idx = ep.get("task_index", 0)
            new_task_idx = task_mapping.get((ds_idx, old_task_idx), 0)

            combined_episodes.append({
                "episode_index": new_ep_idx,
                "length": ep["length"],
                "task_index": new_task_idx,
            })

    save_jsonl(output_path / "meta" / "episodes.jsonl", combined_episodes)
    print(f"  Created: episodes.jsonl ({len(combined_episodes)} episodes)")

    # Step 5: Copy and rename data files
    print("\n[5/6] Copying data files...")
    copied_count = 0

    for ds_idx, ds in enumerate(datasets):
        data_dir = ds["path"] / "data"
        if not data_dir.exists():
            print(f"  Warning: No data/ in {ds['path'].name}")
            continue

        start_ep_idx, chunk_offset = dataset_offsets[ds_idx]

        # Search for per-episode parquet files (GR00T converted format)
        # These are the preferred format: each file contains exactly one episode
        # with frame_index reset to 0
        episode_parquet_files = sorted(data_dir.glob("**/episode_*.parquet"))

        # Search for consolidated parquet files (LeRobot raw format)
        # These contain multiple episodes per file
        file_parquet_files = sorted([
            f for f in data_dir.glob("**/file-*.parquet")
            if not f.name.endswith('.original')  # Skip backup files
        ])

        # Prefer episode_*.parquet if available (already converted)
        # Only use file-*.parquet if no episode files exist
        if episode_parquet_files:
            print(f"  Processing {ds['path'].name}: found {len(episode_parquet_files)} episode parquet files (converted format)")

            for pq_file in episode_parquet_files:
                try:
                    # Extract episode index from filename (e.g., episode_000.parquet -> 0)
                    old_ep_idx = int(pq_file.stem.split('_')[-1])
                    new_ep_idx = start_ep_idx + old_ep_idx

                    # Determine new chunk
                    new_chunk_idx = new_ep_idx // chunk_size

                    # Create chunk directory
                    chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
                    chunk_dir.mkdir(exist_ok=True)

                    target = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"
                    shutil.copy2(pq_file, target)
                    copied_count += 1
                except ValueError:
                    print(f"    Warning: Could not parse episode index from {pq_file.name}")

        elif file_parquet_files:
            # ERROR: Found unconverted file-*.parquet files
            # These are raw LeRobot format where each file contains MULTIPLE episodes
            # and frame_index is NOT reset to 0. This WILL break GR00T training.
            print(f"\n  ❌ ERROR: Dataset '{ds['path'].name}' has not been converted!")
            print(f"     Found {len(file_parquet_files)} file-*.parquet files (raw LeRobot format)")
            print(f"     but no episode_*.parquet files (converted GR00T format).")
            print(f"\n     You MUST run conversion first:")
            print(f"     python custom/scripts/convert_lerobot_v3_to_groot.py \\")
            print(f"         --dataset-path \"{ds['path']}\" \\")
            print(f"         --robot-type so101 \\")
            print(f"         --dual-camera")
            print(f"\n     Then re-run this combine script.")
            sys.exit(1)
        else:
            print(f"\n  ❌ ERROR: No parquet files found in {ds['path'].name}/data/")
            print(f"     Expected either:")
            print(f"       - episode_*.parquet files (converted format)")
            print(f"       - file-*.parquet files (raw LeRobot format)")
            sys.exit(1)

    print(f"  Copied {copied_count} parquet files")

    # Step 6: Copy and rename per-episode video files
    # GR00T expects one video file per episode: episode_{episode_index:06d}.mp4
    print("\n[6/6] Copying per-episode video files...")
    copied_videos = 0
    missing_videos = 0

    # Get video mapping from first dataset's modality.json
    video_keys = {}  # groot_key -> original_key
    if "video" in datasets[0]["modality"]:
        for k, v in datasets[0]["modality"]["video"].items():
            video_keys[k] = v.get("original_key", k)

    print(f"  Video keys to process: {video_keys}")

    for ds_idx, ds in enumerate(datasets):
        videos_root = ds["path"] / "videos"
        if not videos_root.exists():
            print(f"  Warning: No videos/ in {ds['path'].name}")
            continue

        start_ep_idx, chunk_offset = dataset_offsets[ds_idx]

        for groot_key, original_key in video_keys.items():
            # Path to specific camera videos: videos/observation.images.head/
            camera_dir = videos_root / original_key

            if not camera_dir.exists():
                print(f"    Warning: Video dir not found: {camera_dir}")
                continue

            # Look for per-episode video files (GR00T format)
            # Structure: chunk-XXX/episode_YYYYYY.mp4
            episode_videos = sorted(camera_dir.glob("**/episode_*.mp4"))

            if episode_videos:
                # Per-episode format (correct GR00T format)
                print(f"    {ds['path'].name}/{original_key}: {len(episode_videos)} episode videos")

                for v_file in episode_videos:
                    try:
                        # Extract episode index from filename
                        old_ep_idx = int(v_file.stem.split('_')[-1])
                        new_ep_idx = start_ep_idx + old_ep_idx

                        # Determine new chunk
                        new_chunk_idx = new_ep_idx // chunk_size

                        # Create target directory and path
                        target_dir = output_path / "videos" / original_key / f"chunk-{new_chunk_idx:03d}"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_file = target_dir / f"episode_{new_ep_idx:06d}.mp4"

                        shutil.copy2(v_file, target_file)
                        copied_videos += 1
                    except ValueError:
                        print(f"      Warning: Could not parse episode index from {v_file.name}")

            else:
                # Check for old file-based format
                file_videos = sorted(camera_dir.glob("**/file-*.mp4"))
                if file_videos:
                    print(f"\n  ❌ ERROR: Dataset '{ds['path'].name}' has file-based videos!")
                    print(f"     Found {len(file_videos)} file-*.mp4 files (consolidated format)")
                    print(f"     but no episode_*.mp4 files (per-episode format).")
                    print(f"\n     You MUST run conversion first to split videos:")
                    print(f"     python custom/scripts/convert_lerobot_v3_to_groot.py \\")
                    print(f"         --dataset-path \"{ds['path']}\" \\")
                    print(f"         --robot-type so101 \\")
                    print(f"         --dual-camera")
                    print(f"\n     Then re-run this combine script.")
                    sys.exit(1)
                else:
                    print(f"    Warning: No video files found for {ds['path'].name}/{original_key}")
                    missing_videos += 1

    print(f"  Copied {copied_videos} per-episode video files")
    if missing_videos > 0:
        print(f"  ⚠️  Missing videos for {missing_videos} camera(s)")

    # Step 7: Create info.json
    print("\n[7/7] Creating info.json...")
    info = {
        "codebase_version": "v2.0",
        "robot_type": "so101_follower",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(combined_tasks),
        "fps": datasets[0]["fps"],
        "chunks_size": chunk_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        # GR00T per-episode video format (matches official demo_data structure)
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "source_datasets": [str(ds["path"]) for ds in datasets],
    }
    
    # Copy features from first dataset if available (useful for dimension info)
    ds0_info_path = datasets[0]["path"] / "meta" / "info.json"
    if ds0_info_path.exists():
        ds0_info = load_json(ds0_info_path)
        if "features" in ds0_info:
            info["features"] = ds0_info["features"]
            
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
