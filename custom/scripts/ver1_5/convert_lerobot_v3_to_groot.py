#!/usr/bin/env python3
"""
Convert LeRobot v3 Dataset to GR00T-compatible Format

This script bridges the gap between LeRobot v3 dataset format and GR00T's
expectations (LeRobot v2 format) by converting metadata files.

FPS Configuration (per 6_fps_upgrade_30hz.md):
    - Action FPS: 30 Hz (synchronized with video)
    - Video FPS: 30 fps
    - Dataset should be recorded at 30 Hz for optimal GR00T training

Usage:
    python convert_lerobot_v3_to_groot.py --dataset-path /path/to/dataset --robot-type so101

Author: Claude
Date: 2025-11-24
Updated: 2025-12-01 (30 FPS documentation)
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# Robot-specific modality configurations
ROBOT_CONFIGS = {
    "so100": {
        "state_action_mapping": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6}
        },
        "video_mapping_single": {
            "webcam": "observation.images.webcam"
        },
        "video_mapping_dual": {
            "front": "observation.images.head",
            "wrist": "observation.images.left_wrist"
        }
    },
    "so101": {
        # SO-101 has same kinematics as SO-100
        "state_action_mapping": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6}
        },
        "video_mapping_single": {
            "webcam": "observation.images.webcam"
        },
        "video_mapping_dual": {
            "front": "observation.images.head",
            "wrist": "observation.images.left_wrist"
        }
    }
}


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def get_video_frame_timestamps(video_path: Path) -> Optional[List[float]]:
    """
    Get frame timestamps from a video file using ffprobe.

    Returns list of frame presentation timestamps in seconds, or None on error.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'frame=pts_time',
             '-of', 'csv=p=0', str(video_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            timestamps = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    timestamps.append(float(line.strip()))
            return timestamps
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def verify_video_timestamps(video_path: Path, expected_duration: float, tolerance: float = 0.5) -> Tuple[bool, str]:
    """
    Verify that a split video has correct timestamp properties.

    Checks:
    1. First frame timestamp ≈ 0 (within tolerance)
    2. Last frame timestamp ≈ expected_duration (within tolerance)
    3. Timestamps are monotonically increasing

    Args:
        video_path: Path to the video file
        expected_duration: Expected video duration in seconds
        tolerance: Acceptable deviation in seconds

    Returns:
        (success, message) tuple
    """
    timestamps = get_video_frame_timestamps(video_path)

    if timestamps is None or len(timestamps) == 0:
        return False, "Could not read frame timestamps"

    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # Check first frame starts near 0
    if first_ts > tolerance:
        return False, f"First frame at {first_ts:.3f}s (expected ~0s)"

    # Check last frame is near expected duration
    if abs(last_ts - expected_duration) > tolerance:
        return False, f"Last frame at {last_ts:.3f}s (expected ~{expected_duration:.3f}s)"

    # Check monotonically increasing
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i-1]:
            return False, f"Non-monotonic timestamps at frame {i}"

    return True, "OK"


def split_video_segment(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    copy_codec: bool = True
) -> bool:
    """
    Extract a segment from a video file using ffmpeg.

    Args:
        input_path: Source video file
        output_path: Destination video file
        start_time: Start time in seconds
        end_time: End time in seconds
        copy_codec: If True, use -c copy for fast lossless extraction
                   If False, re-encode (slower but more accurate cuts)

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    # Using -ss before -i for fast seeking (keyframe-aligned but fast)
    # Using -t for duration (not -to which is absolute time)
    # Adding -avoid_negative_ts make_zero to ensure output starts at 0
    duration = end_time - start_time
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    cmd.extend(['-ss', str(start_time)])
    cmd.extend(['-i', str(input_path)])
    cmd.extend(['-t', str(duration)])  # -t = duration (correct), -to = absolute time (wrong here)
    cmd.extend(['-avoid_negative_ts', 'make_zero'])  # Reset timestamps to start at 0

    if copy_codec:
        cmd.extend(['-c', 'copy'])
    else:
        # Re-encode for accurate cuts (slower)
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '23'])

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Warning: Timeout splitting {input_path.name}")
        return False


def split_videos_to_episodes(dataset_path: Path) -> bool:
    """
    Split consolidated video files into per-episode videos.

    This is CRITICAL for GR00T - it expects one video file per episode.

    Uses from_timestamp/to_timestamp from meta/episodes/chunk-*/file-*.parquet
    to determine cut points.

    Args:
        dataset_path: Path to dataset root

    Returns:
        True if successful, False otherwise
    """
    print("\n[6/7] Splitting videos into per-episode files...")

    # Check ffmpeg
    if not check_ffmpeg_available():
        print("  ❌ ERROR: ffmpeg not found. Required for video splitting.")
        print("     Install with: sudo apt install ffmpeg")
        return False

    try:
        import pandas as pd
    except ImportError:
        print("  ❌ ERROR: pandas not installed. Required for reading episode metadata.")
        return False

    # Get chunks_size from info.json
    info_path = dataset_path / 'meta' / 'info.json'
    chunks_size = 1000  # default
    if info_path.exists():
        info = load_json(info_path)
        chunks_size = info.get('chunks_size', 1000)

    # Find all chunk directories with episode metadata
    episodes_meta_root = dataset_path / 'meta' / 'episodes'
    chunk_dirs = sorted(list(episodes_meta_root.glob('chunk-*')))

    if not chunk_dirs:
        print(f"  ❌ ERROR: No episode metadata directories found in {episodes_meta_root}")
        return False

    print(f"  📂 Found {len(chunk_dirs)} metadata chunk directories")

    # Read all episode metadata from all chunks
    df_list = []
    for chunk_dir in chunk_dirs:
        meta_files = sorted(list(chunk_dir.glob('file-*.parquet')))
        for f in meta_files:
            df_list.append(pd.read_parquet(f))

    if not df_list:
        print(f"  ❌ ERROR: No episode metadata files found")
        return False

    episodes_df = pd.concat(df_list)

    print(f"  📋 Found {len(episodes_df)} episodes to process")

    # Get unique video keys from column names
    video_keys = set()
    for col in episodes_df.columns:
        if col.startswith('videos/') and '/file_index' in col:
            # Extract video key: videos/observation.images.head/file_index -> observation.images.head
            parts = col.split('/')
            if len(parts) >= 3:
                video_key = parts[1]
                video_keys.add(video_key)

    print(f"  🎥 Video keys: {list(video_keys)}")

    total_split = 0
    failed_split = 0

    for video_key in video_keys:
        print(f"\n  Processing {video_key}...")

        file_idx_col = f'videos/{video_key}/file_index'
        from_ts_col = f'videos/{video_key}/from_timestamp'
        to_ts_col = f'videos/{video_key}/to_timestamp'

        if not all(col in episodes_df.columns for col in [file_idx_col, from_ts_col, to_ts_col]):
            print(f"    ⚠️  Missing columns for {video_key}, skipping")
            continue

        # Check if already processed (any chunk)
        existing_episode_videos = list((dataset_path / 'videos' / video_key).glob('**/episode_*.mp4'))
        if existing_episode_videos:
            print(f"    ℹ️  Found {len(existing_episode_videos)} existing episode videos")
            print(f"       Skipping - delete them first to re-process")
            continue

        # Process each episode
        for episode_idx, row in episodes_df.iterrows():
            ep_num = int(row['episode_index'])
            file_idx = int(row[file_idx_col])
            from_ts = float(row[from_ts_col])
            to_ts = float(row[to_ts_col])

            # Determine source chunk from metadata (LeRobot v3 provides explicit chunk_index)
            # This is CRITICAL for multi-chunk datasets where file-000.mp4 exists in multiple chunks
            chunk_idx_col = f'videos/{video_key}/chunk_index'
            if chunk_idx_col in row:
                source_chunk_idx = int(row[chunk_idx_col])
            else:
                # Fallback: derive from episode number (simple single-chunk datasets)
                source_chunk_idx = ep_num // chunks_size

            # Build source video path using the correct chunk
            source_video = dataset_path / 'videos' / video_key / f'chunk-{source_chunk_idx:03d}' / f'file-{file_idx:03d}.mp4'

            if not source_video.exists():
                print(f"    ❌ Source not found: {source_video}")
                failed_split += 1
                continue

            # Determine target chunk based on episode index
            target_chunk = ep_num // chunks_size
            output_video_dir = dataset_path / 'videos' / video_key / f'chunk-{target_chunk:03d}'
            output_video_dir.mkdir(parents=True, exist_ok=True)

            output_video = output_video_dir / f'episode_{ep_num:06d}.mp4'

            # Skip if already exists
            if output_video.exists():
                total_split += 1
                continue

            # Split the video
            success = split_video_segment(source_video, output_video, from_ts, to_ts)

            if success:
                total_split += 1
                if ep_num < 3 or ep_num == len(episodes_df) - 1:
                    print(f"    ✅ Created: {output_video.name} ({from_ts:.2f}s - {to_ts:.2f}s)")
            else:
                failed_split += 1
                print(f"    ❌ Failed: {output_video.name}")

        if total_split > 3:
            print(f"    ... ({total_split} videos created for {video_key})")

    print(f"\n  📊 Summary: {total_split} videos created, {failed_split} failed")

    # Post-split sanity check: verify timestamps for a few episodes
    if total_split > 0:
        print("\n  🔍 Verifying video timestamps (sampling a few episodes)...")
        # Sample from beginning, middle, and end of episode range
        all_ep_nums = sorted(episodes_df['episode_index'].unique())
        sample_episodes = []
        if len(all_ep_nums) >= 3:
            sample_episodes = [all_ep_nums[0], all_ep_nums[len(all_ep_nums)//2], all_ep_nums[-1]]
        else:
            sample_episodes = all_ep_nums
        sample_episodes = list(set(sample_episodes))

        verification_passed = 0
        verification_failed = 0

        for video_key in video_keys:
            for ep_num in sample_episodes:
                # Find the video in the correct chunk
                target_chunk = ep_num // chunks_size
                video_file = dataset_path / 'videos' / video_key / f'chunk-{target_chunk:03d}' / f'episode_{ep_num:06d}.mp4'
                if not video_file.exists():
                    continue

                # Get expected duration from episode metadata
                ep_row = episodes_df[episodes_df['episode_index'] == ep_num]
                if len(ep_row) == 0:
                    continue

                from_ts_col = f'videos/{video_key}/from_timestamp'
                to_ts_col = f'videos/{video_key}/to_timestamp'
                expected_duration = float(ep_row[to_ts_col].iloc[0]) - float(ep_row[from_ts_col].iloc[0])

                success, message = verify_video_timestamps(video_file, expected_duration)
                if success:
                    verification_passed += 1
                else:
                    verification_failed += 1
                    print(f"    ⚠️  {video_key}/episode_{ep_num:06d}.mp4: {message}")

        if verification_failed == 0:
            print(f"    ✅ All {verification_passed} sampled videos have correct timestamps (start ≈ 0)")
        else:
            print(f"    ⚠️  {verification_failed}/{verification_passed + verification_failed} videos have timestamp issues")
            print(f"       This may cause frame sync problems during training.")
            print(f"       Consider re-encoding with --reencode flag if issues persist.")

    return failed_split == 0


def update_info_json_for_groot(dataset_path: Path) -> None:
    """
    Update info.json with GR00T-compatible video_path and data_path patterns.

    GR00T expects:
    - video_path: videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4
    - data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[7/7] Updating info.json for GR00T compatibility...")

    info_path = dataset_path / 'meta' / 'info.json'

    if not info_path.exists():
        print(f"  ❌ ERROR: info.json not found at {info_path}")
        return

    backup_file(info_path)
    info = load_json(info_path)

    # Store original patterns for reference
    original_video_path = info.get('video_path', '')
    original_data_path = info.get('data_path', '')

    # Update to GR00T-compatible patterns
    # Note: GR00T uses {episode_chunk} and {episode_index}, not {chunk_index} and {file_index}
    info['video_path'] = 'videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4'
    info['data_path'] = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'

    save_json(info_path, info, indent=4)

    print(f"  ✅ Updated info.json:")
    print(f"     video_path: {original_video_path}")
    print(f"              → {info['video_path']}")
    print(f"     data_path:  {original_data_path}")
    print(f"              → {info['data_path']}")


def backup_file(file_path: Path) -> None:
    """Create a backup of a file before modifying it."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"  📦 Backed up: {backup_path.name}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_json(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Save JSON file with proper formatting."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def convert_modality_json(dataset_path: Path, robot_type: str, dual_camera: bool = True) -> None:
    """
    Convert modality.json from LeRobot v3 format to GR00T format.

    Args:
        dataset_path: Path to dataset root
        robot_type: Robot type (so100, so101, etc.)
        dual_camera: Whether to use dual-camera setup
    """
    print("\n[1/7] Converting modality.json...")

    modality_path = dataset_path / "meta" / "modality.json"

    if not robot_type in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}. Supported: {list(ROBOT_CONFIGS.keys())}")

    config = ROBOT_CONFIGS[robot_type]

    # Backup if exists and is LeRobot v3 format
    if modality_path.exists():
        backup_file(modality_path)

    # Create GR00T format modality.json
    modality = {
        "state": config["state_action_mapping"],
        "action": config["state_action_mapping"],
        "video": config["video_mapping_dual"] if dual_camera else config["video_mapping_single"],
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }

    # Convert video mapping to proper format
    video_formatted = {}
    for groot_name, dataset_key in modality["video"].items():
        video_formatted[groot_name] = {"original_key": dataset_key}
    modality["video"] = video_formatted

    save_json(modality_path, modality, indent=4)
    print(f"  ✅ Created: {modality_path.name}")
    print(f"     Cameras: {list(modality['video'].keys())}")
    print(f"     State dim: {sum(v['end'] - v['start'] for v in modality['state'].values())}")
    print(f"     Action dim: {sum(v['end'] - v['start'] for v in modality['action'].values())}")


def fix_stats_json(dataset_path: Path) -> None:
    """
    Fix stats.json to have per-dimension count arrays instead of scalar.

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[2/7] Fixing stats.json...")

    stats_path = dataset_path / "meta" / "stats.json"

    if not stats_path.exists():
        print(f"  ⚠️  Warning: {stats_path.name} not found, skipping")
        return

    backup_file(stats_path)
    stats = load_json(stats_path)

    fixed_count = 0
    for key in ["action", "observation.state"]:
        if key not in stats:
            continue

        old_count = stats[key].get("count", [])
        if not old_count:
            continue

        # Get dimension from mean/std
        dimension = len(stats[key].get("mean", []))

        if len(old_count) == 1 and dimension > 1:
            # Fix: replicate scalar across dimensions
            new_count = [old_count[0]] * dimension
            stats[key]["count"] = new_count
            fixed_count += 1
            print(f"  ✅ Fixed {key}: count [1] → [{dimension}]")

    if fixed_count > 0:
        save_json(stats_path, stats)
        print(f"  ✅ Saved: {stats_path.name} ({fixed_count} fields fixed)")
    else:
        print(f"  ℹ️  No fixes needed for {stats_path.name}")


def generate_episodes_jsonl(dataset_path: Path) -> None:
    """
    Generate episodes.jsonl from info.json or episodes parquet files.

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[3/7] Generating episodes.jsonl...")

    info_path = dataset_path / "meta" / "info.json"
    episodes_jsonl_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_meta_root = dataset_path / "meta" / "episodes"
    tasks_parquet_path = dataset_path / "meta" / "tasks.parquet"

    if not info_path.exists():
        raise FileNotFoundError(f"Required file not found: {info_path}")

    # Build task description -> task_index mapping from tasks.parquet
    task_to_index = {}
    if tasks_parquet_path.exists():
        try:
            import pandas as pd
            tasks_df = pd.read_parquet(tasks_parquet_path)
            for task_desc, row in tasks_df.iterrows():
                task_to_index[str(task_desc)] = int(row['task_index'])
            print(f"  📋 Loaded {len(task_to_index)} task mappings from tasks.parquet")
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to read tasks.parquet: {e}")

    # Try to read from meta/episodes/chunk-*/file-*.parquet (LeRobot v3)
    # Support multi-chunk datasets by scanning all chunk directories
    episode_files = []
    if episodes_meta_root.exists():
        for chunk_dir in sorted(episodes_meta_root.glob("chunk-*")):
            episode_files.extend(sorted(chunk_dir.glob("file-*.parquet")))

    if episode_files:
        try:
            import pandas as pd
            df_list = []
            for f in episode_files:
                df_list.append(pd.read_parquet(f))
            df = pd.concat(df_list)

            task_index_stats = {}  # Track task_index distribution

            with open(episodes_jsonl_path, 'w') as f:
                for _, row in df.iterrows():
                    episode_data = {
                        "episode_index": int(row['episode_index']),
                        "length": int(row['length'])
                    }

                    # Determine task_index
                    task_index = 0  # default

                    # Method 1: Direct task_index column (if present)
                    if 'task_index' in row and row['task_index'] is not None:
                        task_index = int(row['task_index'])
                    # Method 2: Look up from tasks field using tasks.parquet mapping
                    elif 'tasks' in row and row['tasks'] is not None and task_to_index:
                        tasks_val = row['tasks']
                        # tasks can be numpy array, list, tuple, or string
                        # Extract first element if it's array-like
                        if hasattr(tasks_val, '__len__') and len(tasks_val) > 0 and not isinstance(tasks_val, str):
                            task_desc = str(tasks_val[0])
                        else:
                            task_desc = str(tasks_val)
                        task_index = task_to_index.get(task_desc, 0)

                    episode_data['task_index'] = task_index
                    task_index_stats[task_index] = task_index_stats.get(task_index, 0) + 1

                    # Add tasks if present (legacy field for reference)
                    if 'tasks' in row and row['tasks'] is not None:
                        tasks_val = row['tasks']
                        if hasattr(tasks_val, 'tolist'):  # numpy array
                            episode_data['tasks'] = tasks_val.tolist()
                        else:
                            episode_data['tasks'] = tasks_val

                    f.write(json.dumps(episode_data) + '\n')

            print(f"  ✅ Created: {episodes_jsonl_path.name} (from episodes parquet)")
            print(f"     Episodes: {len(df)}")
            print(f"     Task distribution: {dict(sorted(task_index_stats.items()))}")
            return

        except Exception as e:
            print(f"  ⚠️  Warning: Failed to read episodes parquet: {e}")
            print(f"     Falling back to info.json")

    # Fallback: Generate from info.json with average length
    info = load_json(info_path)
    total_episodes = info['total_episodes']
    total_frames = info['total_frames']
    total_tasks = info.get('total_tasks', 1)

    # Assume equal distribution for now
    # TODO: Could read from episodes parquet if needed
    frames_per_episode = total_frames // total_episodes

    with open(episodes_jsonl_path, 'w') as f:
        for episode_index in range(total_episodes):
            # Simple heuristic for single-task or implicit task assignment
            # If explicit task mapping is unknown, default to task 0
            task_index = 0 
            
            episode_data = {
                "episode_index": episode_index,
                "length": frames_per_episode,
                "task_index": task_index 
            }
            f.write(json.dumps(episode_data) + '\n')

    print(f"  ✅ Created: {episodes_jsonl_path.name}")
    print(f"     Episodes: {total_episodes}")
    print(f"     Frames per episode: {frames_per_episode}")
    print(f"     Total frames: {total_frames}")


def generate_tasks_jsonl(dataset_path: Path, task_description: str = "pick red_cube from center") -> None:
    """
    Generate tasks.jsonl from tasks.parquet or info.json.

    In LeRobot v3, task descriptions are stored in tasks.parquet where:
    - Row index (DataFrame index) = task description string
    - Column 'task_index' = task index integer

    Args:
        dataset_path: Path to dataset root
        task_description: Fallback description if tasks.parquet is not available
    """
    print("\n[4/7] Generating tasks.jsonl...")

    tasks_parquet_path = dataset_path / "meta" / "tasks.parquet"
    tasks_jsonl_path = dataset_path / "meta" / "tasks.jsonl"
    info_path = dataset_path / "meta" / "info.json"

    # Try to read from tasks.parquet first (LeRobot v3 format)
    if tasks_parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(tasks_parquet_path)

            # In LeRobot v3, the index is the task description and
            # 'task_index' column contains the index
            with open(tasks_jsonl_path, 'w') as f:
                for task_desc, row in df.iterrows():
                    task_data = {
                        "task_index": int(row['task_index']),
                        "task": str(task_desc)
                    }
                    f.write(json.dumps(task_data) + '\n')

            print(f"  ✅ Created: {tasks_jsonl_path.name} (from tasks.parquet)")
            print(f"     Tasks: {len(df)}")
            for task_desc, row in df.iterrows():
                print(f"       [{int(row['task_index'])}] {task_desc}")
            return

        except Exception as e:
            print(f"  ⚠️  Warning: Failed to read tasks.parquet: {e}")
            print(f"     Falling back to info.json")

    # Fallback: Generate from info.json with default/provided task description
    if not info_path.exists():
        raise FileNotFoundError(f"Required file not found: {info_path}")

    info = load_json(info_path)
    total_tasks = info.get('total_tasks', 1)

    with open(tasks_jsonl_path, 'w') as f:
        for task_index in range(total_tasks):
            task_data = {
                "task_index": task_index,
                "task": task_description if total_tasks == 1 else f"task_{task_index}"
            }
            f.write(json.dumps(task_data) + '\n')

    print(f"  ✅ Created: {tasks_jsonl_path.name} (from info.json fallback)")
    print(f"     Tasks: {total_tasks}")
    print(f"     Description: {task_description}")


def split_parquet_files(dataset_path: Path) -> None:
    """
    Split consolidated parquet files into per-episode files with reset indices.

    This is CRITICAL for GR00T - it expects:
    - Per-episode parquet files (episode_XXXXXX.parquet)
    - Each episode with 0-based indices (not global indices)
    - Episodes placed in correct chunk directory based on episode_idx // chunks_size

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[5/7] Splitting parquet files...")

    try:
        import pandas as pd
    except ImportError:
        print("  ⚠️  Warning: pandas not installed, skipping parquet splitting")
        print("     Install with: pip install pandas pyarrow")
        return

    # Get chunks_size from info.json
    info_path = dataset_path / 'meta' / 'info.json'
    chunks_size = 1000  # default
    if info_path.exists():
        info = load_json(info_path)
        chunks_size = info.get('chunks_size', 1000)
    print(f"  📋 Chunk size: {chunks_size}")

    # Find all chunk directories with data
    data_root = dataset_path / 'data'
    chunk_dirs = sorted(list(data_root.glob('chunk-*')))

    if not chunk_dirs:
        print(f"  ⚠️  No chunk directories found in {data_root}")
        return

    print(f"  📂 Found {len(chunk_dirs)} chunk directories")

    total_split = 0
    all_input_files = []

    for chunk_dir in chunk_dirs:
        # Search for all file-*.parquet files in this chunk
        input_files = sorted(list(chunk_dir.glob('file-*.parquet')))
        all_input_files.extend(input_files)

        if not input_files:
            # Check if already split
            episode_files = list(chunk_dir.glob('episode_*.parquet'))
            if episode_files:
                print(f"  ℹ️  {chunk_dir.name}: {len(episode_files)} episode files (already split)")
            continue

        print(f"  📂 Reading {len(input_files)} parquet files from {chunk_dir.name}...")

        # Read all files in this chunk
        df_list = []
        for f in input_files:
            df_list.append(pd.read_parquet(f))

        if not df_list:
            continue

        df = pd.concat(df_list)
        episodes = sorted(df['episode_index'].unique())
        print(f"     Episodes: {len(episodes)}, rows: {len(df)}")

        # Split by episode, placing in correct chunk based on episode_idx // chunks_size
        for episode_idx in episodes:
            episode_df = df[df['episode_index'] == episode_idx].copy()

            # CRITICAL: Reset index to 0-based for each episode
            episode_df = episode_df.reset_index(drop=True)
            # CRITICAL: Ensure frame_index is 0-based per-episode (defensive)
            episode_len = len(episode_df)
            episode_df['frame_index'] = list(range(episode_len))

            # Determine correct chunk for this episode
            target_chunk = episode_idx // chunks_size
            target_chunk_dir = data_root / f'chunk-{target_chunk:03d}'
            target_chunk_dir.mkdir(parents=True, exist_ok=True)

            output_file = target_chunk_dir / f'episode_{episode_idx:06d}.parquet'
            episode_df.to_parquet(output_file)

            total_split += 1
            if total_split <= 3:
                print(f"  ✅ Created: {output_file.relative_to(data_root)} ({len(episode_df)} frames)")

    if total_split > 3:
        print(f"     ... (+ {total_split - 3} more files)")

    print(f"  📊 Total: {total_split} episode parquet files created")

    # Backup the original consolidated files
    if all_input_files:
        for f in all_input_files:
            backup_path = f.with_suffix(f.suffix + '.original')
            if not backup_path.exists():
                shutil.copy2(f, backup_path)

        print(f"  📦 Backed up original files to *.parquet.original")
        print(f"\n  💡 TIP: You can delete file-*.parquet to save space")
        print(f"          The per-episode files contain all the data")


def validate_conversion(dataset_path: Path) -> bool:
    """
    Validate that all required files exist and are in correct format.

    Args:
        dataset_path: Path to dataset root

    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    meta_path = dataset_path / "meta"
    required_files = {
        "info.json": "Original LeRobot metadata",
        "stats.json": "Fixed statistics with per-dimension counts",
        "modality.json": "GR00T format modality mapping",
        "episodes.jsonl": "Episode metadata in JSONL format",
        "tasks.jsonl": "Task metadata in JSONL format"
    }

    all_valid = True
    for filename, description in required_files.items():
        file_path = meta_path / filename
        if file_path.exists():
            print(f"  ✅ {filename:20s} - {description}")
        else:
            print(f"  ❌ {filename:20s} - MISSING!")
            all_valid = False

    if all_valid:
        print("\n  🎉 All required files present!")

        # Additional format checks
        print("\n  Format checks:")

        # Check modality.json format
        modality = load_json(meta_path / "modality.json")
        has_groot_format = all(k in modality for k in ["state", "action", "video", "annotation"])
        print(f"    {'✅' if has_groot_format else '❌'} modality.json has GR00T format")

        # Check stats.json counts
        stats = load_json(meta_path / "stats.json")
        if "action" in stats:
            count_len = len(stats["action"].get("count", []))
            mean_len = len(stats["action"].get("mean", []))
            counts_match = count_len == mean_len
            print(f"    {'✅' if counts_match else '❌'} stats.json counts match dimensions ({count_len} == {mean_len})")

        # Check episodes.jsonl format
        with open(meta_path / "episodes.jsonl") as f:
            first_line = json.loads(f.readline())
            has_episode_fields = "episode_index" in first_line and "length" in first_line
            print(f"    {'✅' if has_episode_fields else '❌'} episodes.jsonl has required fields")

        # Check tasks.jsonl format
        with open(meta_path / "tasks.jsonl") as f:
            first_line = json.loads(f.readline())
            has_task_fields = "task_index" in first_line and "task" in first_line
            print(f"    {'✅' if has_task_fields else '❌'} tasks.jsonl has required fields")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 dataset to GR00T-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert SO-101 dual-camera dataset
  python convert_lerobot_v3_to_groot.py \\
      --dataset-path /path/to/dataset \\
      --robot-type so101 \\
      --dual-camera

  # Convert SO-100 single-camera dataset
  python convert_lerobot_v3_to_groot.py \\
      --dataset-path /path/to/dataset \\
      --robot-type so100 \\
      --task-description "pick and place"
        """
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to dataset root directory (contains meta/ folder)"
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        choices=list(ROBOT_CONFIGS.keys()),
        required=True,
        help="Robot type (determines action/state dimensions)"
    )

    parser.add_argument(
        "--dual-camera",
        action="store_true",
        default=True,
        help="Use dual-camera setup (default: True)"
    )

    parser.add_argument(
        "--task-description",
        type=str,
        default="pick red_cube from center",
        help="Task description for tasks.jsonl (default: 'pick red_cube from center')"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing conversion, don't modify files"
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        return 1

    meta_path = dataset_path / "meta"
    if not meta_path.exists():
        print(f"❌ Error: meta/ directory not found in {dataset_path}")
        return 1

    info_path = meta_path / "info.json"
    if not info_path.exists():
        print(f"❌ Error: meta/info.json not found")
        return 1

    print("="*70)
    print("LeRobot v3 → GR00T Dataset Conversion")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Robot: {args.robot_type}")
    print(f"Cameras: {'Dual' if args.dual_camera else 'Single'}")
    print("="*70)

    if args.validate_only:
        success = validate_conversion(dataset_path)
        return 0 if success else 1

    try:
        # Run all conversions (7 steps total)
        convert_modality_json(dataset_path, args.robot_type, args.dual_camera)
        fix_stats_json(dataset_path)
        generate_episodes_jsonl(dataset_path)
        generate_tasks_jsonl(dataset_path, args.task_description)
        split_parquet_files(dataset_path)  # CRITICAL: Split parquet for GR00T

        # NEW: Split videos into per-episode files (critical for GR00T)
        video_success = split_videos_to_episodes(dataset_path)
        if not video_success:
            print("\n⚠️  Video splitting had issues. Check the output above.")
            print("   You may need to re-run or manually verify video files.")

        # NEW: Update info.json with GR00T-compatible patterns
        update_info_json_for_groot(dataset_path)

        # Validate
        success = validate_conversion(dataset_path)

        if success:
            print("\n" + "="*70)
            print("✅ CONVERSION COMPLETE!")
            print("="*70)
            print("\nYour dataset is now ready for GR00T training.")
            print("\nNext steps:")
            print("  1. Review backup files (*.backup) if needed")
            print("  2. (Optional) Combine with other datasets:")
            print(f"     python custom/scripts/combine_groot_datasets.py ...")
            print("  3. Run training with:")
            print(f"     bash custom/scripts/train_groot_mini_mvp.sh")
            print("="*70)
            return 0
        else:
            print("\n" + "="*70)
            print("⚠️  CONVERSION COMPLETED WITH WARNINGS")
            print("="*70)
            print("Some validation checks failed. Review the output above.")
            return 1

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
