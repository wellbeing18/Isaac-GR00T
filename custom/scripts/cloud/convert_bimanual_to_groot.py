#!/usr/bin/env python3
"""
Convert LeRobot v3 bimanual dataset to GR00T 1.6 format.

This script converts a LeRobot v3.0 bimanual dataset to GR00T-compatible format by:
1. Converting v3.0 -> v2.1 format (per-episode parquet + video files)
2. Adding GR00T-specific meta/modality.json for bimanual (12D state/action, 3 cameras)
3. Generating meta/tasks.jsonl from task metadata

Bimanual Configuration:
    - State: 12 DOF (left arm 5 + gripper 1 + right arm 5 + gripper 1)
    - Action: 12 DOF (same layout)
    - Cameras: head, left_wrist, right_wrist

Usage:
    # Basic conversion
    python convert_bimanual_to_groot.py \\
        --input /path/to/lerobot/datasets_bimanuel/picknplace_300_144 \\
        --output /path/to/groot/datasets/picknplace_bimanual_groot

    # With force overwrite
    python convert_bimanual_to_groot.py \\
        --input /path/to/dataset \\
        --output /path/to/output \\
        --force

Reference: custom/scripts/ver1_6/convert_lerobot_v3_to_groot_1_6.py
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Default paths (override with --input, --output)
INPUT_DATASET = "/home/jrobot/project/lerobot/datasets_bimanuel/picknplace_300_144"
OUTPUT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/picknplace_bimanual_groot"

# Bimanual SO-101 Robot Configuration
LEFT_ARM_JOINTS = 5   # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
LEFT_GRIPPER = 1      # gripper
RIGHT_ARM_JOINTS = 5  # same as left
RIGHT_GRIPPER = 1     # gripper
TOTAL_STATE_DIM = 12  # (5+1) * 2 = 12
FPS = 30              # Data collection frequency

# Cameras
CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]

# Path to modality.json template (in same directory as this script)
SCRIPT_DIR = Path(__file__).parent
MODALITY_JSON_TEMPLATE = SCRIPT_DIR / "so101_bimanual_modality.json"
# ============================================================================

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lerobot_conversion"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input_dataset(input_path: Path) -> dict:
    """Validate that input dataset is LeRobot v3.0 bimanual format."""
    logger.info(f"Validating input dataset: {input_path}")

    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {input_path}")

    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "unknown")
    if version != "v3.0":
        logger.warning(f"Expected codebase_version 'v3.0', got '{version}'")

    logger.info(f"  - Version: {version}")
    logger.info(f"  - Total episodes: {info.get('total_episodes', 'N/A')}")
    logger.info(f"  - FPS: {info.get('fps', 'N/A')}")

    # Validate bimanual structure
    features = info.get("features", {})
    state_dim = features.get("observation.state", {}).get("shape", [0])[0]
    action_dim = features.get("action", {}).get("shape", [0])[0]

    if state_dim != TOTAL_STATE_DIM:
        logger.warning(f"  - State dim {state_dim} != expected {TOTAL_STATE_DIM}")
    if action_dim != TOTAL_STATE_DIM:
        logger.warning(f"  - Action dim {action_dim} != expected {TOTAL_STATE_DIM}")

    logger.info(f"  - State dim: {state_dim}")
    logger.info(f"  - Action dim: {action_dim}")

    # Check cameras
    video_keys = [k for k in features.keys() if k.startswith("observation.images.")]
    logger.info(f"  - Video keys: {video_keys}")

    return info


def convert_v3_to_v2(input_path: Path, output_path: Path) -> None:
    """Convert LeRobot v3.0 to v2.1 format using official conversion functions."""
    logger.info("Converting v3.0 -> v2.1 format...")

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
        logger.error("Make sure lerobot is installed and you're running from groot env")
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

    logger.info("v3.0 -> v2.1 conversion complete!")


def add_modality_json(output_path: Path, template_path: Path) -> None:
    """Add GR00T-specific modality.json for bimanual robot."""
    logger.info("Adding bimanual modality.json...")

    if not template_path.exists():
        logger.error(f"Modality template not found: {template_path}")
        logger.info("Creating default bimanual modality config...")
        # Create inline if template not found
        modality = create_default_bimanual_modality()
    else:
        with open(template_path) as f:
            modality = json.load(f)
        logger.info(f"  - Loaded from template: {template_path}")

    # Remove comment fields (they start with _)
    def remove_comments(obj):
        if isinstance(obj, dict):
            return {k: remove_comments(v) for k, v in obj.items() if not k.startswith('_')}
        return obj

    modality_clean = remove_comments(modality)

    # Write to output
    modality_path = output_path / "meta" / "modality.json"
    modality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_path, 'w') as f:
        json.dump(modality_clean, f, indent=2)

    logger.info(f"  - Written to: {modality_path}")


def create_default_bimanual_modality() -> dict:
    """Create default bimanual modality config."""
    return {
        "state": {
            "left_arm": {"start": 0, "end": 5},
            "left_gripper": {"start": 5, "end": 6},
            "right_arm": {"start": 6, "end": 11},
            "right_gripper": {"start": 11, "end": 12}
        },
        "action": {
            "left_arm": {"start": 0, "end": 5},
            "left_gripper": {"start": 5, "end": 6},
            "right_arm": {"start": 6, "end": 11},
            "right_gripper": {"start": 11, "end": 12}
        },
        "video": {
            "head": {"original_key": "observation.images.head"},
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"}
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        }
    }


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
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"  {status}: {rf}")
        if not path.exists():
            all_ok = False

    # Check for data files
    data_dir = output_path / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.glob("**/*.parquet"))
        logger.info(f"  OK: Found {len(parquet_files)} parquet files")
    else:
        logger.error("  MISSING: No data directory found")
        all_ok = False

    # Check for video files
    video_dir = output_path / "videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("**/*.mp4"))
        logger.info(f"  OK: Found {len(video_files)} video files")
    else:
        logger.warning("  WARNING: No videos directory found")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 bimanual dataset to GR00T 1.6 format"
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
        default=str(MODALITY_JSON_TEMPLATE),
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
    print("LeRobot v3 Bimanual -> GR00T 1.6 Dataset Conversion")
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

        # Step 2: Convert v3 -> v2.1
        convert_v3_to_v2(input_path, output_path)

        # Step 3: Add modality.json for bimanual
        add_modality_json(output_path, template_path)

        # Step 4: Verify output
        if verify_output_dataset(output_path):
            print("\n" + "=" * 70)
            print("Conversion successful!")
            print(f"Output dataset: {output_path}")
            print("=" * 70)
            print("\nNext steps:")
            print(f"  1. Start training:")
            print(f"     DATASET_PATH={output_path} bash custom/scripts/cloud/train_groot_bimanual.sh")
            print("")
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
