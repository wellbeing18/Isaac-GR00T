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
import subprocess
import sys
import tempfile
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


def concatenate_video_files(video_files: List[Path], output_path: Path) -> bool:
    """
    Concatenate multiple video files into one using ffmpeg.

    Uses stream copy (no re-encoding) for fast, lossless concatenation.

    Args:
        video_files: List of video file paths to concatenate (in order)
        output_path: Output path for the concatenated video

    Returns:
        True if successful, False otherwise
    """
    if len(video_files) == 0:
        return False

    if len(video_files) == 1:
        # Just copy the single file
        shutil.copy2(video_files[0], output_path)
        return True

    # Create temporary concat list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_list_path = Path(f.name)
        for vf in sorted(video_files):
            # Use absolute paths and escape single quotes
            escaped_path = str(vf.absolute()).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        # Run ffmpeg concat with stream copy (lossless, fast)
        result = subprocess.run(
            [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list_path),
                '-c', 'copy',  # Stream copy, no re-encoding
                str(output_path)
            ],
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Warning: ffmpeg timeout concatenating videos")
        return False
    finally:
        # Cleanup temp file
        concat_list_path.unlink(missing_ok=True)


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
        
        # Case 1: Already converted (flat episode_XXX.parquet)
        # We need to put them into chunk folders to match the new schema
        flat_parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        
        if flat_parquet_files:
             print(f"  Processing {ds['path'].name}: found {len(flat_parquet_files)} flat parquet files")
             for pq_file in flat_parquet_files:
                try:
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
                    pass
        else:
            # Case 2: Chunked (chunk-XXX/episode_XXX.parquet) - from new converter
            # Case 3: Raw LeRobot (chunk-XXX/file-YYY.parquet)
            # We search recursively
            all_parquets = sorted(data_dir.glob("**/*.parquet"))
            print(f"  Processing {ds['path'].name}: found {len(all_parquets)} chunked parquet files")
            
            for pq_file in all_parquets:
                # Try to parse chunk/file structure
                old_ep_idx = -1
                
                # Is it inside a chunk folder?
                parent_name = pq_file.parent.name
                if parent_name.startswith("chunk-"):
                    try:
                        old_chunk_idx = int(parent_name.split("-")[-1])
                        
                        if pq_file.name.startswith("episode_"):
                             # chunk-000/episode_000.parquet (Converted format)
                             # Note: episode index in filename is usually local or global?
                             # In the converter script, we kept global index??
                             # Let's assume the filename contains the global index for that dataset.
                             old_ep_idx = int(pq_file.stem.split("_")[-1])
                        elif pq_file.name.startswith("file-"):
                             # chunk-000/file-000.parquet (LeRobot format)
                             file_idx = int(pq_file.stem.split("-")[-1])
                             old_ep_idx = old_chunk_idx * chunk_size + file_idx
                    except ValueError:
                        pass
                
                if old_ep_idx >= 0:
                    new_ep_idx = start_ep_idx + old_ep_idx
                    new_chunk_idx = new_ep_idx // chunk_size
                    
                    chunk_dir = output_path / "data" / f"chunk-{new_chunk_idx:03d}"
                    chunk_dir.mkdir(exist_ok=True)
                    
                    # We always output as episode_XXXX.parquet
                    target = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"
                    shutil.copy2(pq_file, target)
                    copied_count += 1

    print(f"  Copied {copied_count} parquet files")

    # Step 6: Copy and rename video files
    print("\n[6/6] Copying video files...")
    copied_videos = 0
    concatenated_videos = 0

    # Check if ffmpeg is available (needed for concatenation)
    ffmpeg_available = check_ffmpeg_available()
    if not ffmpeg_available:
        print("  Warning: ffmpeg not found. Videos with multiple files per chunk cannot be concatenated.")
        print("           Install ffmpeg: sudo apt install ffmpeg")

    # Get video mapping from first dataset's modality.json
    video_keys = {} # groot_key -> original_key
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

            # Group video files by their source chunk
            # Structure: chunk-XXX/file-YYY.mp4
            chunk_files: Dict[int, List[Path]] = {}  # old_chunk_idx -> list of video files

            for v_file in sorted(camera_dir.glob("**/*.mp4")):
                if v_file.parent.name.startswith("chunk-"):
                    try:
                        old_chunk_idx = int(v_file.parent.name.split("-")[-1])
                        if old_chunk_idx not in chunk_files:
                            chunk_files[old_chunk_idx] = []
                        chunk_files[old_chunk_idx].append(v_file)
                    except ValueError:
                        pass
                elif "episode_" in v_file.name:
                    print(f"    Warning: Found per-episode video {v_file.name}. Skipping.")

            # Process each chunk
            for old_chunk_idx, files in sorted(chunk_files.items()):
                new_chunk_idx = chunk_offset + old_chunk_idx
                target_dir = output_path / "videos" / original_key / f"chunk-{new_chunk_idx:03d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / "file-000.mp4"

                if len(files) == 1:
                    # Single file - just copy
                    shutil.copy2(files[0], target_file)
                    copied_videos += 1
                elif len(files) > 1:
                    # Multiple files - need to concatenate
                    if ffmpeg_available:
                        sorted_files = sorted(files)  # Ensure correct order (file-000, file-001, ...)
                        print(f"    Concatenating {len(sorted_files)} videos for {original_key}/chunk-{new_chunk_idx:03d}...")
                        if concatenate_video_files(sorted_files, target_file):
                            concatenated_videos += 1
                        else:
                            print(f"    ERROR: Failed to concatenate videos for chunk-{new_chunk_idx:03d}")
                            # Fallback: copy first file only
                            shutil.copy2(sorted_files[0], target_file)
                            print(f"    Fallback: Copied only {sorted_files[0].name}")
                            copied_videos += 1
                    else:
                        # No ffmpeg - copy first file and warn
                        print(f"    Warning: Multiple video files in chunk-{old_chunk_idx:03d} but ffmpeg unavailable")
                        print(f"             Only copying {files[0].name}, other files will be lost!")
                        shutil.copy2(sorted(files)[0], target_file)
                        copied_videos += 1

    print(f"  Copied {copied_videos} video files")
    if concatenated_videos > 0:
        print(f"  Concatenated {concatenated_videos} multi-file chunks")

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
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4", # Fixed: Use literal file-000.mp4
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
