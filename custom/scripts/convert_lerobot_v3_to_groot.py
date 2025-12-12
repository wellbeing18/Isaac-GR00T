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
from pathlib import Path
from typing import Dict, Any


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
    print("\n[1/5] Converting modality.json...")

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
    print("\n[2/5] Fixing stats.json...")

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
    print("\n[3/5] Generating episodes.jsonl...")

    info_path = dataset_path / "meta" / "info.json"
    episodes_jsonl_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_parquet_dir = dataset_path / "meta" / "episodes" / "chunk-000"
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

    # Try to read from meta/episodes/chunk-000/file-*.parquet first (LeRobot v3)
    episode_files = sorted(list(episodes_parquet_dir.glob("file-*.parquet")))

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
    print("\n[4/5] Generating tasks.jsonl...")

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
    - Per-episode parquet files (episode_000.parquet, episode_001.parquet, etc.)
    - Each episode with 0-based indices (not global indices)

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[5/5] Splitting parquet files...")

    try:
        import pandas as pd
    except ImportError:
        print("  ⚠️  Warning: pandas not installed, skipping parquet splitting")
        print("     Install with: pip install pandas pyarrow")
        return

    data_dir = dataset_path / 'data' / 'chunk-000'
    
    # Search for all file-*.parquet files
    input_files = sorted(list(data_dir.glob('file-*.parquet')))

    if not input_files:
        print(f"  ℹ️  No consolidated parquet files found in {data_dir}")
        print(f"     Checking if files are already split...")

        # Check if already split
        episode_files = list(data_dir.glob('episode_*.parquet'))
        if episode_files:
            print(f"  ✅ Found {len(episode_files)} per-episode parquet files")
            print(f"     Files already split - no action needed")
            return
        else:
            print(f"  ⚠️  No parquet files found!")
            return

    print(f"  📂 Reading {len(input_files)} parquet files...")
    
    # Read all chunks and concatenate
    # Note: This assumes dataset fits in memory (LeRobot chunks are usually small enough)
    df_list = []
    for f in input_files:
        df_list.append(pd.read_parquet(f))
    
    if not df_list:
        print("  ⚠️  Empty file list after glob?")
        return

    df = pd.concat(df_list)

    total_rows = len(df)
    episodes = sorted(df['episode_index'].unique())

    print(f"     Total rows: {total_rows}")
    print(f"     Episodes: {len(episodes)}")

    # Split by episode
    split_count = 0
    for episode_idx in episodes:
        episode_df = df[df['episode_index'] == episode_idx].copy()

        # CRITICAL: Reset index to 0-based for each episode
        episode_df = episode_df.reset_index(drop=True)

        output_file = data_dir / f'episode_{episode_idx:03d}.parquet'
        episode_df.to_parquet(output_file)

        split_count += 1
        if split_count <= 3 or split_count == len(episodes):
            print(f"  ✅ Created: {output_file.name} ({len(episode_df)} frames)")

    if split_count > 3:
        print(f"     ... (+ {split_count - 3} more files)")

    # Optionally backup the original consolidated files
    # Only backup if we haven't already (to avoid backing up twice)
    for f in input_files:
        backup_path = f.with_suffix(f.suffix + '.original')
        if not backup_path.exists():
            shutil.copy2(f, backup_path)
            # print(f"  📦 Backed up: {backup_path.name}")
    
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
        # Run all conversions
        convert_modality_json(dataset_path, args.robot_type, args.dual_camera)
        fix_stats_json(dataset_path)
        generate_episodes_jsonl(dataset_path)
        generate_tasks_jsonl(dataset_path, args.task_description)
        split_parquet_files(dataset_path)  # CRITICAL: Split parquet for GR00T

        # Validate
        success = validate_conversion(dataset_path)

        if success:
            print("\n" + "="*70)
            print("✅ CONVERSION COMPLETE!")
            print("="*70)
            print("\nYour dataset is now ready for GR00T training.")
            print("\nNext steps:")
            print("  1. Review backup files (*.backup) if needed")
            print("  2. Run training with:")
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
