#!/usr/bin/env python3
"""
Convert LeRobot v3 dataset to GR00T 1.6 format for SO-101.

This script converts a LeRobot v3.0 dataset to GR00T-compatible format by:
1. Converting v3.0 → v2.1 format (per-episode parquet + video files)
2. Adding GR00T-specific meta/modality.json
3. Generating meta/tasks.jsonl from task metadata

The output dataset can be used directly with GR00T 1.6 training scripts.

Decision Notes:
- Uses official scripts/lerobot_conversion/convert_v3_to_v2.py functions for reliability
- Adds modality.json for GR00T-specific data interpretation
- Dual camera setup: head + wrist for better spatial awareness
- Preserves video timestamps for accurate frame synchronization

Usage:
    python convert_lerobot_v3_to_groot_1_6.py [--input PATH] [--output PATH]

Reference: scripts/lerobot_conversion/convert_v3_to_v2.py
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# ============================================================================
# KEY CONFIGURATION - Modify these for your setup
# ============================================================================
INPUT_DATASET = "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place"
OUTPUT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"

# SO-101 Robot Configuration (must match modality.json and ModalityConfig)
ARM_JOINTS = 5      # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DIMS = 1    # single gripper position
TOTAL_STATE_DIM = 6  # ARM_JOINTS + GRIPPER_DIMS
FPS = 30            # Data collection frequency

# Path to modality.json template
MODALITY_JSON_TEMPLATE = "/home/jrobot/project/Isaac-GR00T/custom/cfgs/so101_modality.json"
# ============================================================================

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lerobot_conversion"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input_dataset(input_path: Path) -> dict:
    """Validate that input dataset is LeRobot v3.0 format."""
    logger.info(f"Validating input dataset: {input_path}")

    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {input_path}")

    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "unknown")
    if version != "v3.0":
        raise ValueError(f"Expected codebase_version 'v3.0', got '{version}'")

    logger.info(f"  - Version: {version}")
    logger.info(f"  - Total episodes: {info.get('total_episodes', 'N/A')}")
    logger.info(f"  - FPS: {info.get('fps', 'N/A')}")

    return info


def convert_v3_to_v2(input_path: Path, output_path: Path) -> None:
    """Convert LeRobot v3.0 to v2.1 format using official conversion functions."""
    logger.info("Converting v3.0 → v2.1 format...")

    try:
        # Import official conversion functions
        from convert_v3_to_v2 import (
            load_episode_records,
            convert_info,
            copy_global_stats,
            convert_tasks,
            convert_data,
            convert_videos,
            convert_episodes_metadata,
            copy_ancillary_directories,
        )
        from lerobot.datasets.utils import load_info, DEFAULT_CHUNK_SIZE
    except ImportError as e:
        logger.error(f"Failed to import conversion functions: {e}")
        logger.error("Make sure lerobot is installed and accessible")
        raise

    # Load metadata
    info = load_info(input_path)
    episode_records = load_episode_records(input_path)
    video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    logger.info(f"  - Found {len(episode_records)} episodes")
    logger.info(f"  - Video keys: {video_keys}")
    logger.info(f"  - Chunk size: {chunks_size}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Run conversion steps
    logger.info("  - Converting info.json...")
    convert_info(input_path, output_path, episode_records, video_keys)

    logger.info("  - Copying stats.json...")
    copy_global_stats(input_path, output_path)

    logger.info("  - Converting tasks...")
    convert_tasks(input_path, output_path)

    logger.info("  - Converting parquet data...")
    convert_data(input_path, output_path, episode_records, chunks_size)

    logger.info("  - Converting videos (this may take a while)...")
    convert_videos(input_path, output_path, episode_records, video_keys, chunks_size)

    logger.info("  - Converting episode metadata...")
    convert_episodes_metadata(output_path, episode_records)

    logger.info("  - Copying ancillary directories...")
    copy_ancillary_directories(input_path, output_path)

    logger.info("v3.0 → v2.1 conversion complete!")


def add_modality_json(output_path: Path, template_path: Path) -> None:
    """Add GR00T-specific modality.json to the dataset."""
    logger.info("Adding GR00T modality.json...")

    if not template_path.exists():
        raise FileNotFoundError(f"Modality template not found: {template_path}")

    # Load template
    with open(template_path) as f:
        modality = json.load(f)

    # Remove comment fields (they start with _)
    def remove_comments(obj):
        if isinstance(obj, dict):
            return {k: remove_comments(v) for k, v in obj.items() if not k.startswith('_')}
        return obj

    modality_clean = remove_comments(modality)

    # Write to output
    modality_path = output_path / "meta" / "modality.json"
    with open(modality_path, 'w') as f:
        json.dump(modality_clean, f, indent=2)

    logger.info(f"  - Written to: {modality_path}")


def verify_output_dataset(output_path: Path) -> bool:
    """Verify the converted dataset has all required files."""
    logger.info("Verifying output dataset...")

    required_files = [
        "meta/info.json",
        "meta/modality.json",
        "meta/stats.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
    ]

    all_ok = True
    for rf in required_files:
        path = output_path / rf
        status = "✓" if path.exists() else "✗"
        logger.info(f"  {status} {rf}")
        if not path.exists():
            all_ok = False

    # Check for data files
    data_dir = output_path / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.glob("**/*.parquet"))
        logger.info(f"  ✓ Found {len(parquet_files)} parquet files")
    else:
        logger.error("  ✗ No data directory found")
        all_ok = False

    # Check for video files
    video_dir = output_path / "videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("**/*.mp4"))
        logger.info(f"  ✓ Found {len(video_files)} video files")
    else:
        logger.warning("  ! No videos directory found (may be image-only dataset)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 dataset to GR00T 1.6 format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=INPUT_DATASET,
        help=f"Input dataset path (default: {INPUT_DATASET})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_DATASET,
        help=f"Output dataset path (default: {OUTPUT_DATASET})"
    )
    parser.add_argument(
        "--modality-template",
        type=str,
        default=MODALITY_JSON_TEMPLATE,
        help=f"Path to modality.json template (default: {MODALITY_JSON_TEMPLATE})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    template_path = Path(args.modality_template)

    print("=" * 70)
    print("LeRobot v3 → GR00T 1.6 Dataset Conversion")
    print("=" * 70)
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Template: {template_path}")
    print("=" * 70)

    # Validate input
    if not input_path.exists():
        logger.error(f"Input dataset not found: {input_path}")
        sys.exit(1)

    # Check output
    if output_path.exists():
        if args.force:
            logger.warning(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            logger.error(f"Output directory already exists: {output_path}")
            logger.error("Use --force to overwrite")
            sys.exit(1)

    try:
        # Step 1: Validate input
        info = validate_input_dataset(input_path)

        # Step 2: Convert v3 → v2.1
        convert_v3_to_v2(input_path, output_path)

        # Step 3: Add modality.json
        add_modality_json(output_path, template_path)

        # Step 4: Verify output
        if verify_output_dataset(output_path):
            print("\n" + "=" * 70)
            print("Conversion successful!")
            print(f"Output dataset: {output_path}")
            print("=" * 70)
            print("\nNext steps:")
            print(f"  1. Verify dataset: python verify_groot_dataset.py --dataset {output_path}")
            print(f"  2. Test zero-shot: python verify_zeroshot_1_6.py --dataset {output_path}")
            print(f"  3. Start training: bash train_groot_so101_1_6.sh")
        else:
            logger.error("Verification failed - some files are missing")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
