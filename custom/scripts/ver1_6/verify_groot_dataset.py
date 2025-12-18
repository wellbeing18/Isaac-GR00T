#!/usr/bin/env python3
"""
Verify GR00T 1.6 Dataset Structure and Integrity.

This script validates that a converted dataset is properly formatted for GR00T 1.6
training and inference. It checks:
1. Required metadata files existence and schema validity
2. modality.json structure and key consistency
3. Episode count consistency between metadata and data files
4. State/action dimensions matching modality.json slicing
5. Video files existence and basic format checks
6. Statistics file completeness for normalization
7. Timestamp monotonicity in parquet files
8. Optionally: Try loading with LeRobotEpisodeLoader to validate full compatibility

Usage:
    python verify_groot_dataset.py --dataset /path/to/dataset [--load-test]

Reference: gr00t/data/dataset/lerobot_episode_loader.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Default dataset path (override with --dataset)
DEFAULT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"

# Expected dimensions for SO-101
EXPECTED_STATE_DIM = 6   # 5 arm + 1 gripper
EXPECTED_ACTION_DIM = 6  # Same as state

# Expected video resolution
EXPECTED_VIDEO_HEIGHT = 480
EXPECTED_VIDEO_WIDTH = 640
# ============================================================================

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DatasetVerifier:
    """Verify GR00T dataset structure and integrity."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.meta_dir = dataset_path / "meta"
        self.errors = []
        self.warnings = []

        # Metadata containers
        self.info = None
        self.modality = None
        self.stats = None
        self.episodes = []
        self.tasks = []

    def log_error(self, msg: str):
        """Log and record an error."""
        logger.error(f"  ERROR: {msg}")
        self.errors.append(msg)

    def log_warning(self, msg: str):
        """Log and record a warning."""
        logger.warning(f"  WARNING: {msg}")
        self.warnings.append(msg)

    def log_ok(self, msg: str):
        """Log success."""
        logger.info(f"  OK: {msg}")

    def verify_file_exists(self, filepath: Path, required: bool = True) -> bool:
        """Check if a file exists."""
        exists = filepath.exists()
        if not exists:
            if required:
                self.log_error(f"Required file missing: {filepath.name}")
            else:
                self.log_warning(f"Optional file missing: {filepath.name}")
        return exists

    def verify_metadata_files(self) -> bool:
        """Verify all required metadata files exist and are valid JSON/JSONL."""
        logger.info("\n[1/7] Checking metadata files...")

        required_files = {
            "info.json": True,
            "modality.json": True,
            "stats.json": True,
            "episodes.jsonl": True,
            "tasks.jsonl": True,
        }

        all_ok = True
        for filename, required in required_files.items():
            filepath = self.meta_dir / filename
            if not self.verify_file_exists(filepath, required):
                all_ok = False
                continue

            # Try loading the file
            try:
                with open(filepath, 'r') as f:
                    if filename.endswith('.jsonl'):
                        data = [json.loads(line) for line in f if line.strip()]
                    else:
                        data = json.load(f)

                # Store for later use
                if filename == "info.json":
                    self.info = data
                elif filename == "modality.json":
                    self.modality = data
                elif filename == "stats.json":
                    self.stats = data
                elif filename == "episodes.jsonl":
                    self.episodes = data
                elif filename == "tasks.jsonl":
                    self.tasks = data

                self.log_ok(f"{filename} loaded successfully ({len(data) if isinstance(data, list) else 'dict'})")

            except json.JSONDecodeError as e:
                self.log_error(f"{filename} is not valid JSON: {e}")
                all_ok = False
            except Exception as e:
                self.log_error(f"Failed to load {filename}: {e}")
                all_ok = False

        return all_ok

    def verify_info_json(self) -> bool:
        """Verify info.json has required fields."""
        logger.info("\n[2/7] Checking info.json structure...")

        if not self.info:
            self.log_error("info.json not loaded")
            return False

        required_fields = [
            "codebase_version",
            "total_episodes",
            "total_frames",
            "fps",
            "features",
            "data_path",
            "chunks_size",
        ]

        all_ok = True
        for field in required_fields:
            if field not in self.info:
                self.log_error(f"Missing required field: {field}")
                all_ok = False
            else:
                value = self.info[field]
                # Truncate long values for display
                display_val = str(value)[:50] + "..." if len(str(value)) > 50 else value
                self.log_ok(f"{field}: {display_val}")

        # Check codebase version
        version = self.info.get("codebase_version", "")
        if version not in ["v2.0", "v2.1"]:
            self.log_warning(f"Unexpected codebase_version: {version} (expected v2.0 or v2.1)")

        return all_ok

    def verify_modality_json(self) -> bool:
        """Verify modality.json structure and key consistency."""
        logger.info("\n[3/7] Checking modality.json structure...")

        if not self.modality:
            self.log_error("modality.json not loaded")
            return False

        required_sections = ["state", "action", "video"]
        all_ok = True

        for section in required_sections:
            if section not in self.modality:
                self.log_error(f"Missing required section: {section}")
                all_ok = False
                continue

            self.log_ok(f"Section '{section}' found with keys: {list(self.modality[section].keys())}")

            # Verify state and action have start/end indices
            if section in ["state", "action"]:
                for key, config in self.modality[section].items():
                    # Skip comment fields
                    if key.startswith("_"):
                        continue

                    if "start" not in config or "end" not in config:
                        self.log_error(f"{section}.{key} missing 'start' or 'end' indices")
                        all_ok = False
                    else:
                        start, end = config["start"], config["end"]
                        self.log_ok(f"  {key}: [{start}:{end}] (dim={end-start})")

            # Verify video has original_key mapping
            if section == "video":
                for key, config in self.modality[section].items():
                    if key.startswith("_"):
                        continue

                    if "original_key" not in config:
                        self.log_warning(f"video.{key} missing 'original_key' (will default to observation.images.{key})")
                    else:
                        self.log_ok(f"  {key} -> {config['original_key']}")

        # Check annotation section (optional but recommended)
        if "annotation" in self.modality:
            self.log_ok(f"Annotation section found: {list(self.modality['annotation'].keys())}")
        else:
            self.log_warning("No 'annotation' section in modality.json (language support may be limited)")

        # Verify total dimensions match expected
        state_dim = sum(
            config["end"] - config["start"]
            for key, config in self.modality.get("state", {}).items()
            if not key.startswith("_") and "start" in config
        )
        action_dim = sum(
            config["end"] - config["start"]
            for key, config in self.modality.get("action", {}).items()
            if not key.startswith("_") and "start" in config
        )

        if state_dim != EXPECTED_STATE_DIM:
            self.log_warning(f"Total state dimension ({state_dim}) != expected ({EXPECTED_STATE_DIM})")
        else:
            self.log_ok(f"Total state dimension: {state_dim}")

        if action_dim != EXPECTED_ACTION_DIM:
            self.log_warning(f"Total action dimension ({action_dim}) != expected ({EXPECTED_ACTION_DIM})")
        else:
            self.log_ok(f"Total action dimension: {action_dim}")

        return all_ok

    def verify_episode_consistency(self) -> bool:
        """Verify episode counts match across metadata and data files."""
        logger.info("\n[4/7] Checking episode consistency...")

        if not self.info or not self.episodes:
            self.log_error("info.json or episodes.jsonl not loaded")
            return False

        # Get counts from different sources
        info_episode_count = self.info.get("total_episodes", 0)
        episodes_jsonl_count = len(self.episodes)

        # Count actual parquet files
        data_dir = self.dataset_path / "data"
        parquet_files = list(data_dir.glob("**/*.parquet")) if data_dir.exists() else []
        parquet_count = len(parquet_files)

        self.log_ok(f"info.json total_episodes: {info_episode_count}")
        self.log_ok(f"episodes.jsonl entries: {episodes_jsonl_count}")
        self.log_ok(f"Parquet files found: {parquet_count}")

        all_ok = True

        if info_episode_count != episodes_jsonl_count:
            self.log_error(f"Episode count mismatch: info.json ({info_episode_count}) != episodes.jsonl ({episodes_jsonl_count})")
            all_ok = False

        if info_episode_count != parquet_count:
            self.log_error(f"Episode count mismatch: info.json ({info_episode_count}) != parquet files ({parquet_count})")
            all_ok = False

        # Verify episode indices are sequential
        expected_indices = set(range(info_episode_count))
        actual_indices = set(ep.get("episode_index", -1) for ep in self.episodes)

        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            if missing:
                self.log_error(f"Missing episode indices: {sorted(missing)[:10]}...")
            if extra:
                self.log_error(f"Unexpected episode indices: {sorted(extra)[:10]}...")
            all_ok = False
        else:
            self.log_ok(f"Episode indices are sequential [0, {info_episode_count-1}]")

        return all_ok

    def verify_data_dimensions(self) -> bool:
        """Verify state/action dimensions in parquet files match modality.json."""
        logger.info("\n[5/7] Checking data dimensions...")

        data_dir = self.dataset_path / "data"
        if not data_dir.exists():
            self.log_error("data/ directory not found")
            return False

        # Load a sample parquet file
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            self.log_error("No parquet files found")
            return False

        sample_file = parquet_files[0]
        self.log_ok(f"Sampling from: {sample_file.name}")

        try:
            df = pd.read_parquet(sample_file)
            self.log_ok(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        except Exception as e:
            self.log_error(f"Failed to load parquet file: {e}")
            return False

        all_ok = True

        # Check state dimensions
        state_col = "observation.state"
        if state_col in df.columns:
            sample_state = df[state_col].iloc[0]
            if isinstance(sample_state, np.ndarray):
                state_dim = len(sample_state)
                if state_dim != EXPECTED_STATE_DIM:
                    self.log_warning(f"State dimension ({state_dim}) != expected ({EXPECTED_STATE_DIM})")
                else:
                    self.log_ok(f"State dimension: {state_dim}")

                # Show sample values
                self.log_ok(f"Sample state: {sample_state[:3]}... (showing first 3)")
            else:
                self.log_warning(f"State column not numpy array: {type(sample_state)}")
        else:
            self.log_error(f"Column '{state_col}' not found in parquet")
            all_ok = False

        # Check action dimensions
        action_col = "action"
        if action_col in df.columns:
            sample_action = df[action_col].iloc[0]
            if isinstance(sample_action, np.ndarray):
                action_dim = len(sample_action)
                if action_dim != EXPECTED_ACTION_DIM:
                    self.log_warning(f"Action dimension ({action_dim}) != expected ({EXPECTED_ACTION_DIM})")
                else:
                    self.log_ok(f"Action dimension: {action_dim}")

                # Show sample values
                self.log_ok(f"Sample action: {sample_action[:3]}... (showing first 3)")
            else:
                self.log_warning(f"Action column not numpy array: {type(sample_action)}")
        else:
            self.log_error(f"Column '{action_col}' not found in parquet")
            all_ok = False

        # Check timestamp column
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].values
            if np.all(np.diff(timestamps) >= 0):
                self.log_ok(f"Timestamps are monotonically increasing ({timestamps[0]:.3f} -> {timestamps[-1]:.3f})")
            else:
                self.log_warning("Timestamps are NOT monotonically increasing")
        else:
            self.log_warning("No 'timestamp' column found")

        # Check task_index column
        if "task_index" in df.columns:
            task_indices = df["task_index"].unique()
            self.log_ok(f"Task indices found: {sorted(task_indices)}")
        else:
            self.log_warning("No 'task_index' column found (language conditioning may not work)")

        return all_ok

    def verify_video_files(self) -> bool:
        """Verify video files exist and have expected format."""
        logger.info("\n[6/7] Checking video files...")

        videos_dir = self.dataset_path / "videos"
        if not videos_dir.exists():
            self.log_warning("videos/ directory not found (may be image-only dataset)")
            return True

        video_files = list(videos_dir.glob("**/*.mp4"))
        self.log_ok(f"Found {len(video_files)} video files")

        if not video_files:
            self.log_warning("No .mp4 files found in videos/")
            return True

        # Check video keys from modality.json
        if self.modality and "video" in self.modality:
            expected_keys = [k for k in self.modality["video"].keys() if not k.startswith("_")]
            self.log_ok(f"Expected video keys: {expected_keys}")

            # Check that we have videos for each key
            for video_key in expected_keys:
                original_key = self.modality["video"][video_key].get("original_key", f"observation.images.{video_key}")
                # Video paths typically use the original_key
                matching_videos = [v for v in video_files if original_key in str(v)]
                if matching_videos:
                    self.log_ok(f"  {video_key} ({original_key}): {len(matching_videos)} videos")
                else:
                    self.log_error(f"  {video_key} ({original_key}): No matching videos found")

        # Sample a video file and check format (if ffprobe available)
        try:
            import subprocess
            sample_video = video_files[0]
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(sample_video)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        width = stream.get("width", 0)
                        height = stream.get("height", 0)
                        codec = stream.get("codec_name", "unknown")
                        fps_str = stream.get("avg_frame_rate", "0/1")
                        try:
                            num, den = fps_str.split("/")
                            fps = float(num) / float(den) if float(den) > 0 else 0
                        except:
                            fps = 0

                        self.log_ok(f"Sample video: {width}x{height}, {codec}, {fps:.1f}fps")

                        if height != EXPECTED_VIDEO_HEIGHT or width != EXPECTED_VIDEO_WIDTH:
                            self.log_warning(f"Video resolution ({width}x{height}) != expected ({EXPECTED_VIDEO_WIDTH}x{EXPECTED_VIDEO_HEIGHT})")
        except FileNotFoundError:
            self.log_warning("ffprobe not found, skipping video format check")
        except Exception as e:
            self.log_warning(f"Failed to probe video: {e}")

        return True

    def verify_statistics(self) -> bool:
        """Verify statistics file has required keys for normalization."""
        logger.info("\n[7/7] Checking statistics file...")

        if not self.stats:
            self.log_error("stats.json not loaded")
            return False

        required_stat_types = ["mean", "std", "min", "max"]
        required_keys = ["observation.state", "action"]

        all_ok = True

        for key in required_keys:
            if key not in self.stats:
                self.log_error(f"Missing statistics for: {key}")
                all_ok = False
                continue

            key_stats = self.stats[key]
            for stat_type in required_stat_types:
                if stat_type not in key_stats:
                    self.log_error(f"Missing {stat_type} in stats[{key}]")
                    all_ok = False
                else:
                    values = key_stats[stat_type]
                    dim = len(values) if isinstance(values, list) else 1
                    self.log_ok(f"  {key}.{stat_type}: dim={dim}")

        # Check for relative_stats.json (optional but useful for relative actions)
        relative_stats_path = self.meta_dir / "relative_stats.json"
        if relative_stats_path.exists():
            self.log_ok("relative_stats.json found (relative actions supported)")
        else:
            self.log_warning("relative_stats.json not found (relative actions may not work correctly)")

        return all_ok

    def try_load_with_loader(self) -> bool:
        """Try loading the dataset with LeRobotEpisodeLoader."""
        logger.info("\n[Optional] Testing with LeRobotEpisodeLoader...")

        try:
            from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
            from gr00t.data.types import ModalityConfig

            # Create minimal modality config for testing
            modality_configs = {
                "video": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["head", "wrist"],
                ),
                "state": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["single_arm", "gripper"],
                ),
                "action": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["single_arm", "gripper"],
                ),
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["annotation.human.action.task_description"],
                ),
            }

            loader = LeRobotEpisodeLoader(
                dataset_path=self.dataset_path,
                modality_configs=modality_configs,
                video_backend="torchcodec",
            )

            self.log_ok(f"LeRobotEpisodeLoader initialized successfully")
            self.log_ok(f"  Number of episodes: {len(loader)}")
            self.log_ok(f"  Episode lengths: {loader.episode_lengths[:5]}... (showing first 5)")

            # Try loading first episode
            logger.info("  Loading sample episode (episode 0)...")
            df = loader[0]
            self.log_ok(f"  Episode 0 loaded: {len(df)} frames")
            self.log_ok(f"  Columns: {list(df.columns)}")

            # Get statistics
            stats = loader.get_dataset_statistics()
            self.log_ok(f"  Statistics keys: {list(stats.keys())}")

            return True

        except ImportError as e:
            self.log_warning(f"Could not import LeRobotEpisodeLoader: {e}")
            return True  # Not a failure, just skip
        except Exception as e:
            self.log_error(f"LeRobotEpisodeLoader test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_checks(self, load_test: bool = False) -> bool:
        """Run all verification checks."""
        print("=" * 70)
        print("GR00T 1.6 Dataset Verification")
        print("=" * 70)
        print(f"Dataset: {self.dataset_path}")
        print("=" * 70)

        results = [
            self.verify_metadata_files(),
            self.verify_info_json(),
            self.verify_modality_json(),
            self.verify_episode_consistency(),
            self.verify_data_dimensions(),
            self.verify_video_files(),
            self.verify_statistics(),
        ]

        if load_test:
            results.append(self.try_load_with_loader())

        # Summary
        print("\n" + "=" * 70)
        print("Verification Summary")
        print("=" * 70)

        all_passed = all(results) and len(self.errors) == 0

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")

        if all_passed:
            print("\n  RESULT: ALL CHECKS PASSED")
            print("  Dataset is ready for GR00T 1.6 training!")
        else:
            print(f"\n  RESULT: {len(self.errors)} ERRORS, {len(self.warnings)} WARNINGS")
            print("  Please fix errors before proceeding with training.")

        print("=" * 70)

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify GR00T 1.6 dataset structure and integrity"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to dataset (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Also test loading with LeRobotEpisodeLoader"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    verifier = DatasetVerifier(dataset_path)
    success = verifier.run_all_checks(load_test=args.load_test)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
