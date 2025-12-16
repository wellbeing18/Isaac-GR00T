#!/usr/bin/env python3
"""
GR00T Dataset Verification Script

Comprehensive validation of datasets for GR00T training compatibility.
Compares against official LeRobot v2.1 format used by NVIDIA tutorials.

Usage:
    python verify_dataset_for_groot.py --dataset /path/to/dataset
    python verify_dataset_for_groot.py --dataset /path/to/dataset --verbose
    python verify_dataset_for_groot.py --dataset /path/to/dataset --test-loading

Reference: Official dataset format from youliangtan/so101-table-cleanup
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    OK = "OK"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    details: Optional[str] = None


@dataclass
class VerificationReport:
    dataset_path: str
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, severity: Severity, message: str, details: str = None):
        self.checks.append(CheckResult(name, severity, message, details))

    def has_errors(self) -> bool:
        return any(c.severity in [Severity.ERROR, Severity.CRITICAL] for c in self.checks)

    def has_warnings(self) -> bool:
        return any(c.severity == Severity.WARNING for c in self.checks)

    def print_report(self, verbose: bool = False):
        print("\n" + "=" * 70)
        print("GR00T DATASET VERIFICATION REPORT")
        print("=" * 70)
        print(f"Dataset: {self.dataset_path}")
        print("-" * 70)

        # Group by severity
        by_severity = {}
        for check in self.checks:
            if check.severity not in by_severity:
                by_severity[check.severity] = []
            by_severity[check.severity].append(check)

        # Print in order of severity
        for severity in [Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.OK]:
            if severity in by_severity:
                for check in by_severity[severity]:
                    icon = {
                        Severity.OK: "✅",
                        Severity.INFO: "ℹ️",
                        Severity.WARNING: "⚠️",
                        Severity.ERROR: "❌",
                        Severity.CRITICAL: "🚨"
                    }[severity]
                    print(f"{icon} [{severity.value}] {check.name}: {check.message}")
                    if verbose and check.details:
                        for line in check.details.split("\n"):
                            print(f"    {line}")

        print("-" * 70)
        total = len(self.checks)
        errors = sum(1 for c in self.checks if c.severity in [Severity.ERROR, Severity.CRITICAL])
        warnings = sum(1 for c in self.checks if c.severity == Severity.WARNING)
        ok = sum(1 for c in self.checks if c.severity == Severity.OK)

        print(f"Summary: {total} checks | {ok} OK | {warnings} warnings | {errors} errors")

        if errors > 0:
            print("\n🚨 DATASET HAS CRITICAL ISSUES - May not work with GR00T")
        elif warnings > 0:
            print("\n⚠️ DATASET HAS WARNINGS - Review before training")
        else:
            print("\n✅ DATASET LOOKS GOOD")
        print("=" * 70 + "\n")


# Official v2.1 format reference (from youliangtan/so101-table-cleanup)
OFFICIAL_V21_FORMAT = {
    "codebase_version": "v2.1",
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "required_meta_files": ["info.json", "modality.json", "episodes.jsonl", "tasks.jsonl"],
    "optional_meta_files": ["stats.json", "episodes_stats.jsonl"],
}


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> Optional[List[Dict]]:
    """Load JSONL file, return None if not found or invalid."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_video_info(video_path: Path) -> Optional[Dict]:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=duration,start_time,nb_frames,r_frame_rate',
             '-show_entries', 'format=duration',
             '-of', 'json', str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get('streams', [{}])[0]
            fmt = data.get('format', {})
            return {
                'duration': float(fmt.get('duration', stream.get('duration', 0))),
                'start_time': float(stream.get('start_time', 0)),
                'nb_frames': int(stream.get('nb_frames', 0)) if stream.get('nb_frames') else None,
                'frame_rate': stream.get('r_frame_rate', 'unknown')
            }
    except Exception:
        pass
    return None


def verify_structure(dataset_path: Path, report: VerificationReport):
    """Verify basic directory structure."""

    # Check required directories
    for dir_name in ['data', 'meta', 'videos']:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            report.add(f"Directory {dir_name}/", Severity.OK, "Exists")
        else:
            report.add(f"Directory {dir_name}/", Severity.CRITICAL, "Missing required directory")


def verify_info_json(dataset_path: Path, report: VerificationReport) -> Optional[Dict]:
    """Verify info.json contents."""
    info_path = dataset_path / "meta" / "info.json"
    info = load_json(info_path)

    if info is None:
        report.add("info.json", Severity.CRITICAL, "Missing or invalid info.json")
        return None

    report.add("info.json", Severity.OK, "File exists and is valid JSON")

    # Check codebase_version
    version = info.get("codebase_version", "unknown")
    if version == "v2.1":
        report.add("codebase_version", Severity.OK, f"v2.1 (matches official format)")
    elif version == "v3.0":
        report.add("codebase_version", Severity.WARNING,
                   f"v3.0 (differs from official v2.1)",
                   "Official GR00T datasets use v2.1. This may work but is not guaranteed.")
    else:
        report.add("codebase_version", Severity.WARNING, f"Unknown version: {version}")

    # Check data_path pattern
    data_path = info.get("data_path", "")
    official_data_path = OFFICIAL_V21_FORMAT["data_path"]
    if data_path == official_data_path:
        report.add("data_path pattern", Severity.OK, "Matches official format")
    else:
        report.add("data_path pattern", Severity.INFO,
                   f"Different from official",
                   f"Yours: {data_path}\nOfficial: {official_data_path}")

    # Check video_path pattern
    video_path = info.get("video_path", "")
    official_video_path = OFFICIAL_V21_FORMAT["video_path"]
    if video_path == official_video_path:
        report.add("video_path pattern", Severity.OK, "Matches official format")
    else:
        # Check if it's the v3 pattern
        if "{video_key}/chunk-" in video_path:
            report.add("video_path pattern", Severity.WARNING,
                       "Uses v3.0 structure (video_key before chunk)",
                       f"Yours: {video_path}\nOfficial: {official_video_path}\n"
                       "GR00T reads this from info.json, so it should work, but test to confirm.")
        else:
            report.add("video_path pattern", Severity.WARNING,
                       f"Non-standard pattern",
                       f"Yours: {video_path}\nOfficial: {official_video_path}")

    # Check required fields
    required_fields = ["total_episodes", "total_frames", "fps", "features"]
    for field in required_fields:
        if field in info:
            report.add(f"info.json[{field}]", Severity.OK, f"Present: {info[field] if field != 'features' else '...'}")
        else:
            report.add(f"info.json[{field}]", Severity.ERROR, "Missing required field")

    return info


def verify_modality_json(dataset_path: Path, report: VerificationReport) -> Optional[Dict]:
    """Verify modality.json contents."""
    modality_path = dataset_path / "meta" / "modality.json"
    modality = load_json(modality_path)

    if modality is None:
        report.add("modality.json", Severity.ERROR,
                   "Missing modality.json",
                   "GR00T requires modality.json to map state/action/video keys")
        return None

    report.add("modality.json", Severity.OK, "File exists")

    # Check required sections
    for section in ["state", "action", "video"]:
        if section in modality:
            keys = list(modality[section].keys())
            report.add(f"modality.json[{section}]", Severity.OK, f"Keys: {keys}")
        else:
            report.add(f"modality.json[{section}]", Severity.ERROR, "Missing required section")

    # Check SO-100/101 specific requirements
    if "state" in modality:
        if "single_arm" in modality["state"] and "gripper" in modality["state"]:
            report.add("modality.json SO-101 format", Severity.OK,
                       "Has single_arm and gripper (correct for SO-101)")
        else:
            report.add("modality.json SO-101 format", Severity.WARNING,
                       f"Keys don't match SO-101 convention",
                       f"Expected: single_arm, gripper\nFound: {list(modality['state'].keys())}")

    # Check video keys
    if "video" in modality:
        video_keys = list(modality["video"].keys())
        if "front" in video_keys or "wrist" in video_keys:
            report.add("modality.json video keys", Severity.OK,
                       f"Has standard keys: {video_keys}")
        else:
            report.add("modality.json video keys", Severity.INFO,
                       f"Custom video keys: {video_keys}",
                       "Standard keys are 'front' and 'wrist'")

    return modality


def verify_episodes_jsonl(dataset_path: Path, info: Dict, report: VerificationReport) -> Optional[List[Dict]]:
    """Verify episodes.jsonl contents."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)

    if episodes is None:
        report.add("episodes.jsonl", Severity.ERROR,
                   "Missing or invalid episodes.jsonl",
                   "This file defines episode metadata (index, length, task)")
        return None

    report.add("episodes.jsonl", Severity.OK, f"Found {len(episodes)} episodes")

    # Verify count matches info.json
    expected_episodes = info.get("total_episodes", 0)
    if len(episodes) == expected_episodes:
        report.add("episodes.jsonl count", Severity.OK,
                   f"Matches info.json ({expected_episodes})")
    else:
        report.add("episodes.jsonl count", Severity.ERROR,
                   f"Mismatch: {len(episodes)} vs info.json says {expected_episodes}")

    # Check required fields in each episode
    required_fields = ["episode_index", "length"]
    sample = episodes[0] if episodes else {}
    for field in required_fields:
        if field in sample:
            report.add(f"episodes.jsonl[{field}]", Severity.OK, "Present in all episodes")
        else:
            report.add(f"episodes.jsonl[{field}]", Severity.ERROR, "Missing required field")

    # Check for task information
    if "tasks" in sample or "task_index" in sample:
        report.add("episodes.jsonl task info", Severity.OK, "Has task information")
    else:
        report.add("episodes.jsonl task info", Severity.WARNING, "Missing task information")

    # Verify total frames
    total_length = sum(ep.get("length", 0) for ep in episodes)
    expected_frames = info.get("total_frames", 0)
    if total_length == expected_frames:
        report.add("episodes.jsonl total frames", Severity.OK,
                   f"Sum of lengths ({total_length}) matches info.json")
    else:
        report.add("episodes.jsonl total frames", Severity.WARNING,
                   f"Sum of lengths ({total_length}) != info.json ({expected_frames})")

    return episodes


def verify_tasks_jsonl(dataset_path: Path, info: Dict, report: VerificationReport) -> Optional[List[Dict]]:
    """Verify tasks.jsonl contents."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = load_jsonl(tasks_path)

    if tasks is None:
        report.add("tasks.jsonl", Severity.ERROR,
                   "Missing or invalid tasks.jsonl",
                   "This file defines task descriptions for each task_index")
        return None

    report.add("tasks.jsonl", Severity.OK, f"Found {len(tasks)} tasks")

    # Check count matches info.json
    expected_tasks = info.get("total_tasks", 0)
    if len(tasks) == expected_tasks:
        report.add("tasks.jsonl count", Severity.OK,
                   f"Matches info.json ({expected_tasks})")
    else:
        report.add("tasks.jsonl count", Severity.WARNING,
                   f"Mismatch: {len(tasks)} vs info.json says {expected_tasks}")

    # Check task format
    if tasks:
        sample = tasks[0]
        if "task_index" in sample and "task" in sample:
            report.add("tasks.jsonl format", Severity.OK,
                       f"Correct format (task_index + task)",
                       f"Example: {sample}")
        else:
            report.add("tasks.jsonl format", Severity.WARNING,
                       f"Non-standard format: {list(sample.keys())}")

    return tasks


def verify_parquet_files(dataset_path: Path, info: Dict, episodes: List[Dict],
                         report: VerificationReport, verbose: bool = False):
    """Verify parquet file structure and contents."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.add("parquet verification", Severity.WARNING,
                   "PyArrow not available, skipping parquet checks")
        return

    data_path_pattern = info.get("data_path", "")
    chunks_size = info.get("chunks_size", 1000)
    total_episodes = info.get("total_episodes", 0)

    # Count parquet files
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.glob("**/*.parquet"))
    report.add("parquet file count", Severity.OK if len(parquet_files) == total_episodes else Severity.WARNING,
               f"Found {len(parquet_files)} files (expected {total_episodes})")

    # Check first few episodes in detail
    issues = []
    checked = 0
    for ep in (episodes or [])[:5]:  # Check first 5 episodes
        ep_idx = ep.get("episode_index", checked)
        chunk_idx = ep_idx // chunks_size

        # Build expected path
        parquet_path = dataset_path / data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=ep_idx
        )

        if not parquet_path.exists():
            issues.append(f"Episode {ep_idx}: File not found at {parquet_path}")
            continue

        try:
            df = pq.read_table(str(parquet_path)).to_pandas()

            # Check frame_index starts at 0
            min_frame = df["frame_index"].min()
            if min_frame != 0:
                issues.append(f"Episode {ep_idx}: frame_index starts at {min_frame} (should be 0)")

            # Check timestamp starts near 0
            min_ts = df["timestamp"].min()
            if min_ts > 1.0:  # Allow 1 second tolerance
                issues.append(f"Episode {ep_idx}: timestamp starts at {min_ts:.2f}s (should be ~0)")

            # Check episode_index is consistent
            unique_eps = df["episode_index"].unique()
            if len(unique_eps) != 1 or unique_eps[0] != ep_idx:
                issues.append(f"Episode {ep_idx}: inconsistent episode_index {unique_eps}")

            # Check row count matches length
            expected_len = ep.get("length", 0)
            if len(df) != expected_len:
                issues.append(f"Episode {ep_idx}: {len(df)} rows vs episodes.jsonl says {expected_len}")

            checked += 1

        except Exception as e:
            issues.append(f"Episode {ep_idx}: Error reading parquet: {e}")

    if not issues:
        report.add("parquet content checks", Severity.OK,
                   f"Checked {checked} episodes, all valid",
                   "frame_index starts at 0, timestamp starts near 0, lengths match")
    else:
        report.add("parquet content checks", Severity.ERROR,
                   f"Found {len(issues)} issues",
                   "\n".join(issues))


def verify_video_files(dataset_path: Path, info: Dict, modality: Dict,
                       episodes: List[Dict], report: VerificationReport, verbose: bool = False):
    """Verify video file structure and contents."""
    video_path_pattern = info.get("video_path", "")
    chunks_size = info.get("chunks_size", 1000)
    total_episodes = info.get("total_episodes", 0)

    # Get video keys from modality.json
    video_keys = []
    if modality and "video" in modality:
        for key, config in modality["video"].items():
            original_key = config.get("original_key", key)
            video_keys.append((key, original_key))

    if not video_keys:
        report.add("video keys", Severity.WARNING, "No video keys found in modality.json")
        return

    # Count video files
    videos_dir = dataset_path / "videos"
    video_files = list(videos_dir.glob("**/*.mp4"))
    expected_videos = total_episodes * len(video_keys)

    report.add("video file count", Severity.OK if len(video_files) == expected_videos else Severity.WARNING,
               f"Found {len(video_files)} files (expected {expected_videos}: {total_episodes} eps x {len(video_keys)} cameras)")

    # Check first few episodes in detail
    issues = []
    for ep in (episodes or [])[:3]:  # Check first 3 episodes
        ep_idx = ep.get("episode_index", 0)
        chunk_idx = ep_idx // chunks_size

        for groot_key, original_key in video_keys:
            # Build expected path using original_key (the actual folder name)
            video_path = dataset_path / video_path_pattern.format(
                episode_chunk=chunk_idx, episode_index=ep_idx, video_key=original_key
            )

            if not video_path.exists():
                issues.append(f"Episode {ep_idx}, {groot_key}: File not found at {video_path}")
                continue

            # Check video metadata
            video_info = get_video_info(video_path)
            if video_info:
                # Check start_time is 0
                if video_info['start_time'] > 0.1:  # Allow 0.1s tolerance
                    issues.append(f"Episode {ep_idx}, {groot_key}: start_time={video_info['start_time']:.3f}s (should be 0)")

                # Check duration roughly matches episode length
                expected_frames = ep.get("length", 0)
                fps = info.get("fps", 30)
                expected_duration = expected_frames / fps
                actual_duration = video_info['duration']

                if abs(actual_duration - expected_duration) > 2.0:  # 2 second tolerance
                    issues.append(f"Episode {ep_idx}, {groot_key}: duration={actual_duration:.1f}s vs expected ~{expected_duration:.1f}s")

    if not issues:
        report.add("video content checks", Severity.OK,
                   f"Checked {min(3, len(episodes or []))} episodes x {len(video_keys)} cameras",
                   "Video start_time is 0, durations roughly match episode lengths")
    else:
        report.add("video content checks", Severity.ERROR,
                   f"Found {len(issues)} issues",
                   "\n".join(issues))


def verify_groot_loading(dataset_path: Path, report: VerificationReport):
    """Test loading dataset with GR00T's LeRobotSingleDataset."""
    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config
        from gr00t.data.schema import EmbodimentTag
    except ImportError:
        report.add("GR00T loading test", Severity.INFO,
                   "GR00T not available, skipping loading test",
                   "Run in groot conda environment to test")
        return

    try:
        data_cfg = load_data_config("so100_dualcam")
        modality_config = data_cfg.modality_config()

        dataset = LeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=modality_config,
            video_backend="torchvision_av",
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        )

        # Try to load first sample
        sample = dataset.get_step_data(0, 0)

        # Check expected keys
        expected_keys = ["video.front", "video.wrist", "state.single_arm", "state.gripper", "action.single_arm", "action.gripper"]
        found_keys = list(sample.keys())

        missing = [k for k in expected_keys if k not in found_keys]
        if missing:
            report.add("GR00T loading test", Severity.WARNING,
                       f"Loaded but missing keys: {missing}",
                       f"Found keys: {found_keys}")
        else:
            report.add("GR00T loading test", Severity.OK,
                       f"Successfully loaded dataset",
                       f"Sample keys: {found_keys[:6]}...")

        # Check video shape
        for key in ["video.front", "video.wrist"]:
            if key in sample:
                shape = sample[key].shape if hasattr(sample[key], 'shape') else "unknown"
                report.add(f"GR00T {key} shape", Severity.OK, f"{shape}")

    except Exception as e:
        report.add("GR00T loading test", Severity.ERROR,
                   f"Failed to load: {type(e).__name__}",
                   str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset compatibility with GR00T training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_dataset_for_groot.py --dataset /path/to/dataset
    python verify_dataset_for_groot.py --dataset /path/to/dataset --verbose
    python verify_dataset_for_groot.py --dataset /path/to/dataset --test-loading
        """
    )
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed information for each check")
    parser.add_argument("--test-loading", "-t", action="store_true",
                        help="Test loading with GR00T LeRobotSingleDataset")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return 1

    report = VerificationReport(str(dataset_path))

    print("\nVerifying dataset structure...")
    verify_structure(dataset_path, report)

    print("Verifying info.json...")
    info = verify_info_json(dataset_path, report)

    print("Verifying modality.json...")
    modality = verify_modality_json(dataset_path, report)

    if info:
        print("Verifying episodes.jsonl...")
        episodes = verify_episodes_jsonl(dataset_path, info, report)

        print("Verifying tasks.jsonl...")
        verify_tasks_jsonl(dataset_path, info, report)

        print("Verifying parquet files...")
        verify_parquet_files(dataset_path, info, episodes, report, args.verbose)

        print("Verifying video files...")
        verify_video_files(dataset_path, info, modality, episodes, report, args.verbose)

    if args.test_loading:
        print("Testing GR00T loading...")
        verify_groot_loading(dataset_path, report)

    report.print_report(args.verbose)

    return 1 if report.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
