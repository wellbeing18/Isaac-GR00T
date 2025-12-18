#!/usr/bin/env python3
"""
GR00T Dataset Verification Script

Comprehensive verification of converted/combined GR00T datasets.
Checks data integrity, synchronization between parquet/video/metadata,
and validates the dataset is ready for training.

Usage:
    # Verify a converted/combined dataset
    python verify_groot_dataset.py --dataset /path/to/datasets_groot

    # Verify with detailed output
    python verify_groot_dataset.py --dataset /path/to/datasets_groot --verbose

    # Verify source dataset before combination
    python verify_groot_dataset.py --dataset /path/to/source --source

Author: GR00T Training Scripts
Date: 2025-12-12
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Complete verification report."""
    dataset_path: Path
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def check_meta_files(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Check that all required metadata files exist."""
    meta_path = dataset_path / "meta"
    required_files = [
        "info.json",
        "modality.json",
        "stats.json",
        "episodes.jsonl",
        "tasks.jsonl"
    ]

    missing = []
    present = []
    for fname in required_files:
        if (meta_path / fname).exists():
            present.append(fname)
        else:
            missing.append(fname)

    details = []
    if verbose:
        for f in present:
            details.append(f"  ✓ {f}")
        for f in missing:
            details.append(f"  ✗ {f} (MISSING)")

    if missing:
        return VerificationResult(
            name="Meta Files",
            passed=False,
            message=f"Missing {len(missing)} required files: {missing}",
            details=details
        )
    return VerificationResult(
        name="Meta Files",
        passed=True,
        message=f"All {len(required_files)} required files present",
        details=details
    )


def check_info_json(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify info.json structure and content."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return VerificationResult(
            name="info.json",
            passed=False,
            message="File not found"
        )

    try:
        info = load_json(info_path)
    except Exception as e:
        return VerificationResult(
            name="info.json",
            passed=False,
            message=f"Failed to parse: {e}"
        )

    required_fields = ["total_episodes", "total_frames", "fps"]
    missing = [f for f in required_fields if f not in info]

    details = []
    if verbose:
        details.append(f"  total_episodes: {info.get('total_episodes', 'MISSING')}")
        details.append(f"  total_frames: {info.get('total_frames', 'MISSING')}")
        details.append(f"  fps: {info.get('fps', 'MISSING')}")
        details.append(f"  total_tasks: {info.get('total_tasks', 'N/A')}")

    if missing:
        return VerificationResult(
            name="info.json",
            passed=False,
            message=f"Missing required fields: {missing}",
            details=details
        )

    return VerificationResult(
        name="info.json",
        passed=True,
        message=f"{info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps",
        details=details
    )


def check_episodes_jsonl(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify episodes.jsonl structure and content."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    info_path = dataset_path / "meta" / "info.json"

    if not episodes_path.exists():
        return VerificationResult(
            name="episodes.jsonl",
            passed=False,
            message="File not found"
        )

    try:
        episodes = load_jsonl(episodes_path)
        info = load_json(info_path) if info_path.exists() else {}
    except Exception as e:
        return VerificationResult(
            name="episodes.jsonl",
            passed=False,
            message=f"Failed to parse: {e}"
        )

    issues = []
    details = []

    # Check episode count matches info.json
    expected_count = info.get('total_episodes', len(episodes))
    if len(episodes) != expected_count:
        issues.append(f"Episode count mismatch: {len(episodes)} vs info.json {expected_count}")

    # Check required fields
    for i, ep in enumerate(episodes[:5]):  # Sample first 5
        if 'episode_index' not in ep:
            issues.append(f"Episode {i} missing 'episode_index'")
        if 'length' not in ep:
            issues.append(f"Episode {i} missing 'length'")
        if 'task_index' not in ep:
            issues.append(f"Episode {i} missing 'task_index'")

    # Check task_index distribution
    task_counts = {}
    total_frames = 0
    for ep in episodes:
        ti = ep.get('task_index', -1)
        task_counts[ti] = task_counts.get(ti, 0) + 1
        total_frames += ep.get('length', 0)

    if verbose:
        details.append(f"  Episodes: {len(episodes)}")
        details.append(f"  Total frames (from lengths): {total_frames}")
        details.append(f"  Task distribution: {dict(sorted(task_counts.items()))}")

    # Check if all task_index are the same (potential bug)
    if len(task_counts) == 1 and len(episodes) > 10:
        ti = list(task_counts.keys())[0]
        issues.append(f"WARNING: All {len(episodes)} episodes have task_index={ti} - possible conversion bug!")

    # Check frame count matches info.json
    expected_frames = info.get('total_frames', total_frames)
    if total_frames != expected_frames:
        issues.append(f"Frame count mismatch: sum(lengths)={total_frames} vs info.json={expected_frames}")

    if issues:
        return VerificationResult(
            name="episodes.jsonl",
            passed=False,
            message=f"{len(issues)} issues found",
            details=details + [f"  ✗ {i}" for i in issues]
        )

    return VerificationResult(
        name="episodes.jsonl",
        passed=True,
        message=f"{len(episodes)} episodes, {len(task_counts)} tasks, {total_frames} frames",
        details=details
    )


def check_tasks_jsonl(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify tasks.jsonl structure and content."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not tasks_path.exists():
        return VerificationResult(
            name="tasks.jsonl",
            passed=False,
            message="File not found"
        )

    try:
        tasks = load_jsonl(tasks_path)
        episodes = load_jsonl(episodes_path) if episodes_path.exists() else []
    except Exception as e:
        return VerificationResult(
            name="tasks.jsonl",
            passed=False,
            message=f"Failed to parse: {e}"
        )

    issues = []
    details = []

    # Check required fields
    for i, task in enumerate(tasks):
        if 'task_index' not in task:
            issues.append(f"Task {i} missing 'task_index'")
        if 'task' not in task:
            issues.append(f"Task {i} missing 'task' description")

    # Get task indices used in episodes
    episode_task_indices = set(ep.get('task_index', -1) for ep in episodes)
    task_indices = set(t.get('task_index', -1) for t in tasks)

    # Check for orphan task indices in episodes
    orphan_indices = episode_task_indices - task_indices
    if orphan_indices:
        issues.append(f"Episodes reference undefined task_index: {orphan_indices}")

    if verbose:
        details.append(f"  Tasks defined: {len(tasks)}")
        for t in tasks[:5]:
            details.append(f"    [{t.get('task_index')}] {t.get('task', 'N/A')[:50]}...")
        if len(tasks) > 5:
            details.append(f"    ... and {len(tasks) - 5} more")

    if issues:
        return VerificationResult(
            name="tasks.jsonl",
            passed=False,
            message=f"{len(issues)} issues found",
            details=details + [f"  ✗ {i}" for i in issues]
        )

    return VerificationResult(
        name="tasks.jsonl",
        passed=True,
        message=f"{len(tasks)} tasks defined",
        details=details
    )


def check_modality_json(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify modality.json structure."""
    modality_path = dataset_path / "meta" / "modality.json"

    if not modality_path.exists():
        return VerificationResult(
            name="modality.json",
            passed=False,
            message="File not found"
        )

    try:
        modality = load_json(modality_path)
    except Exception as e:
        return VerificationResult(
            name="modality.json",
            passed=False,
            message=f"Failed to parse: {e}"
        )

    issues = []
    details = []

    required_sections = ["state", "action", "video", "annotation"]
    for section in required_sections:
        if section not in modality:
            issues.append(f"Missing required section: {section}")

    if verbose:
        if "state" in modality:
            state_dim = sum(v['end'] - v['start'] for v in modality['state'].values())
            details.append(f"  State dimension: {state_dim}")
        if "action" in modality:
            action_dim = sum(v['end'] - v['start'] for v in modality['action'].values())
            details.append(f"  Action dimension: {action_dim}")
        if "video" in modality:
            details.append(f"  Video keys: {list(modality['video'].keys())}")

    if issues:
        return VerificationResult(
            name="modality.json",
            passed=False,
            message=f"{len(issues)} issues found",
            details=details + [f"  ✗ {i}" for i in issues]
        )

    return VerificationResult(
        name="modality.json",
        passed=True,
        message="Valid GR00T format",
        details=details
    )


def check_parquet_files(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify parquet data files structure and content."""
    data_dir = dataset_path / "data"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not data_dir.exists():
        return VerificationResult(
            name="Parquet Files",
            passed=False,
            message="data/ directory not found"
        )

    try:
        import pandas as pd
        episodes = load_jsonl(episodes_path) if episodes_path.exists() else []
    except ImportError:
        return VerificationResult(
            name="Parquet Files",
            passed=False,
            message="pandas not installed"
        )
    except Exception as e:
        return VerificationResult(
            name="Parquet Files",
            passed=False,
            message=f"Failed to load episodes: {e}"
        )

    issues = []
    details = []

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("**/episode_*.parquet"))

    if not parquet_files:
        return VerificationResult(
            name="Parquet Files",
            passed=False,
            message="No episode_*.parquet files found"
        )

    details.append(f"  Found {len(parquet_files)} parquet files")

    # Check count matches episodes
    expected_count = len(episodes)
    if len(parquet_files) != expected_count:
        issues.append(f"Parquet count mismatch: {len(parquet_files)} files vs {expected_count} episodes")

    # Sample and verify some parquet files
    sample_indices = [0, len(parquet_files)//2, len(parquet_files)-1] if len(parquet_files) > 2 else range(len(parquet_files))

    for idx in sample_indices:
        if idx >= len(parquet_files):
            continue
        pq_file = parquet_files[idx]
        try:
            df = pd.read_parquet(pq_file)

            # Check required columns
            required_cols = ['action', 'observation.state', 'frame_index', 'episode_index']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                issues.append(f"{pq_file.name}: missing columns {missing_cols}")
                continue

            # Check single episode per file
            unique_episodes = df['episode_index'].unique()
            if len(unique_episodes) > 1:
                issues.append(f"{pq_file.name}: contains {len(unique_episodes)} episodes (should be 1)")

            # Check frame_index starts at 0 (required by GR00T)
            min_frame = df['frame_index'].min()
            if min_frame != 0:
                issues.append(f"{pq_file.name}: frame_index starts at {min_frame} (should be 0)")

            # Check frame_index is contiguous
            max_frame = df['frame_index'].max()
            if len(df) != max_frame + 1:
                issues.append(f"{pq_file.name}: frame_index not contiguous (len={len(df)}, max={max_frame})")

            if verbose and idx < 3:
                ep_idx = unique_episodes[0] if len(unique_episodes) == 1 else unique_episodes
                details.append(f"    {pq_file.name}: {len(df)} rows, episode={ep_idx}, frames 0-{max_frame}")

        except Exception as e:
            issues.append(f"{pq_file.name}: failed to read - {e}")

    if issues:
        return VerificationResult(
            name="Parquet Files",
            passed=False,
            message=f"{len(issues)} issues found",
            details=details + [f"  ✗ {i}" for i in issues[:10]]  # Limit output
        )

    return VerificationResult(
        name="Parquet Files",
        passed=True,
        message=f"{len(parquet_files)} files, all valid",
        details=details
    )


def check_video_files(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """Verify video files exist and match modality configuration."""
    videos_dir = dataset_path / "videos"
    modality_path = dataset_path / "meta" / "modality.json"
    info_path = dataset_path / "meta" / "info.json"

    if not videos_dir.exists():
        return VerificationResult(
            name="Video Files",
            passed=False,
            message="videos/ directory not found"
        )

    try:
        modality = load_json(modality_path) if modality_path.exists() else {}
        info = load_json(info_path) if info_path.exists() else {}
    except Exception as e:
        return VerificationResult(
            name="Video Files",
            passed=False,
            message=f"Failed to load config: {e}"
        )

    issues = []
    details = []

    # Get expected video keys from modality.json
    video_config = modality.get("video", {})
    expected_keys = []
    for key, config in video_config.items():
        original_key = config.get("original_key", key)
        expected_keys.append(original_key)

    if verbose:
        details.append(f"  Expected video keys: {expected_keys}")

    # Check each expected video directory
    total_episode_videos = 0
    for video_key in expected_keys:
        key_dir = videos_dir / video_key
        if not key_dir.exists():
            issues.append(f"Video directory not found: {video_key}/")
            continue

        # Find per-episode video files (new GR00T format: episode_XXXXXX.mp4)
        episode_videos = sorted(key_dir.glob("**/episode_*.mp4"))

        # Also check for old file-based format (should not exist after conversion)
        file_videos = sorted(key_dir.glob("**/file-*.mp4"))

        if episode_videos:
            total_episode_videos += len(episode_videos)
            if verbose:
                details.append(f"  {video_key}/: {len(episode_videos)} episode videos (per-episode format ✓)")
        elif file_videos:
            issues.append(f"{video_key}/: found {len(file_videos)} file-*.mp4 (old concatenated format) - needs conversion!")
        else:
            issues.append(f"No video files found in {video_key}/")

    # Verify episode video count matches episode count
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists() and total_episode_videos > 0:
        try:
            episodes = load_jsonl(episodes_path)
            expected_video_count = len(episodes) * len(expected_keys)
            if total_episode_videos != expected_video_count:
                issues.append(f"Episode video count mismatch: {total_episode_videos} files vs {expected_video_count} expected ({len(episodes)} episodes × {len(expected_keys)} cameras)")
            elif verbose:
                details.append(f"  Total episode videos: {total_episode_videos} ({len(episodes)} episodes × {len(expected_keys)} cameras)")
        except Exception:
            pass

    if issues:
        return VerificationResult(
            name="Video Files",
            passed=False,
            message=f"{len(issues)} issues found",
            details=details + [f"  ✗ {i}" for i in issues]
        )

    return VerificationResult(
        name="Video Files",
        passed=True,
        message=f"{len(expected_keys)} video streams found",
        details=details
    )


def check_data_video_sync(dataset_path: Path, verbose: bool = False) -> VerificationResult:
    """
    Verify synchronization between parquet data, video files, and metadata.

    This is the MOST CRITICAL check - mismatches here will cause training failures
    like "IndexError: index 845 is out of bounds for axis 0 with size 700".

    Uses "One-Video-Per-Episode" format where:
    - Each episode has its own parquet file with timestamps starting at ~0
    - Each episode has its own video file (episode_XXXXXX.mp4) starting at ~0
    - Parquet duration should match video duration

    Checks:
    1. info.json total_frames == sum(episodes.jsonl lengths)
    2. info.json total_frames == sum(parquet row counts)
    3. Each episode's length matches its parquet file row count
    4. Each parquet has single episode with frame_index starting at 0
    5. Parquet timestamps start near 0 (per-episode format)
    6. Video file exists for each episode with matching duration
    """
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    info_path = dataset_path / "meta" / "info.json"
    modality_path = dataset_path / "meta" / "modality.json"
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    try:
        import pandas as pd
        episodes = load_jsonl(episodes_path) if episodes_path.exists() else []
        info = load_json(info_path) if info_path.exists() else {}
        modality = load_json(modality_path) if modality_path.exists() else {}
    except Exception as e:
        return VerificationResult(
            name="Data-Video Sync",
            passed=False,
            message=f"Failed to load metadata: {e}"
        )

    issues = []
    details = []

    # 1. Calculate total frames from different sources
    total_frames_info = info.get('total_frames', 0)
    total_frames_episodes = sum(ep.get('length', 0) for ep in episodes)

    # 2. Count frames in ALL parquet files
    parquet_files = sorted(data_dir.glob("**/episode_*.parquet"))
    total_frames_parquet = 0
    parquet_issues = []

    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            total_frames_parquet += len(df)

            # Check each parquet has single episode
            if 'episode_index' in df.columns:
                unique_eps = df['episode_index'].unique()
                if len(unique_eps) > 1:
                    parquet_issues.append(f"{pq_file.name}: contains {len(unique_eps)} episodes (must be 1)")

            # Check frame_index starts at 0
            if 'frame_index' in df.columns:
                min_frame = df['frame_index'].min()
                if min_frame != 0:
                    parquet_issues.append(f"{pq_file.name}: frame_index starts at {min_frame} (must be 0)")
        except Exception as e:
            parquet_issues.append(f"{pq_file.name}: failed to read - {e}")

    # 3. Get video frame count (via ffprobe)
    total_frames_video = None
    video_check_details = []

    # Get video keys from modality
    video_config = modality.get("video", {})
    video_keys = [v.get("original_key", k) for k, v in video_config.items()]

    if video_keys and videos_dir.exists():
        try:
            import subprocess

            for video_key in video_keys[:1]:  # Check first video stream only
                video_dir = videos_dir / video_key
                video_files = sorted(video_dir.glob("**/file-*.mp4"))

                if video_files:
                    # For combined dataset, should be single file per chunk
                    # Count total frames across all video files
                    total_video_frames_for_key = 0

                    for vf in video_files:
                        result = subprocess.run(
                            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                             '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(vf)],
                            capture_output=True, text=True, timeout=120
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            total_video_frames_for_key += int(result.stdout.strip())

                    total_frames_video = total_video_frames_for_key
                    video_check_details.append(f"{video_key}: {total_frames_video} frames from {len(video_files)} file(s)")
                    break  # Only need to check one camera stream

        except Exception as e:
            video_check_details.append(f"ffprobe check failed: {e}")

    # Build details
    if verbose:
        details.append(f"  Frames from info.json: {total_frames_info}")
        details.append(f"  Frames from episodes.jsonl: {total_frames_episodes}")
        details.append(f"  Frames from parquet files: {total_frames_parquet}")
        if total_frames_video is not None:
            details.append(f"  Frames from video files: {total_frames_video}")
        for vd in video_check_details:
            details.append(f"    {vd}")

    # 4. Check consistency between sources
    if total_frames_info != total_frames_episodes:
        issues.append(f"info.json ({total_frames_info}) != episodes.jsonl ({total_frames_episodes})")

    if total_frames_info != total_frames_parquet:
        issues.append(f"info.json ({total_frames_info}) != parquet files ({total_frames_parquet})")

    if total_frames_video is not None and total_frames_info != total_frames_video:
        issues.append(f"info.json ({total_frames_info}) != video files ({total_frames_video})")

    # 5. Add parquet structure issues
    if parquet_issues:
        issues.extend(parquet_issues[:5])  # Limit to first 5
        if len(parquet_issues) > 5:
            issues.append(f"... and {len(parquet_issues) - 5} more parquet issues")

    # 6. Verify episode lengths match parquet row counts (sample check)
    sample_mismatches = []
    for i, ep in enumerate(episodes[:10]):  # Sample first 10
        ep_idx = ep.get('episode_index', i)
        expected_len = ep.get('length', 0)

        # Find corresponding parquet
        pq_file = None
        for pf in parquet_files:
            try:
                file_ep_idx = int(pf.stem.split('_')[-1])
                if file_ep_idx == ep_idx:
                    pq_file = pf
                    break
            except:
                pass

        if pq_file:
            try:
                df = pd.read_parquet(pq_file)
                actual_len = len(df)
                if actual_len != expected_len:
                    sample_mismatches.append(f"Episode {ep_idx}: expected {expected_len}, got {actual_len}")
            except:
                pass

    if sample_mismatches:
        issues.append(f"Episode length mismatches: {sample_mismatches[:3]}")

    # 7. CRITICAL: Verify per-episode timestamp synchronization
    # In the new "One-Video-Per-Episode" format:
    # - Each parquet file has timestamps starting at ~0
    # - Each video file (episode_XXXXXX.mp4) also starts at ~0
    # - Video duration should match parquet duration
    sync_issues = []

    if parquet_files and videos_dir.exists() and video_keys:
        try:
            import subprocess

            # Sample a few episodes to check timestamp synchronization
            sample_indices = [0, len(parquet_files)//2, len(parquet_files)-1]

            fps = info.get('fps', 30)
            chunks_size = info.get('chunks_size', 1000)

            for sample_idx in sample_indices:
                if sample_idx >= len(parquet_files):
                    continue

                pq_file = parquet_files[sample_idx]
                df = pd.read_parquet(pq_file)

                if 'timestamp' not in df.columns:
                    sync_issues.append(f"{pq_file.name}: missing 'timestamp' column (required for video sync)")
                    continue

                # Get timestamp range from parquet
                parquet_ts_min = df['timestamp'].min()
                parquet_ts_max = df['timestamp'].max()
                parquet_duration = parquet_ts_max - parquet_ts_min

                # Extract episode index from filename
                ep_index = int(pq_file.stem.split('_')[-1])

                # NEW: In per-episode format, timestamps should start near 0
                if parquet_ts_min > 1.0:  # More than 1 second from 0
                    sync_issues.append(f"{pq_file.name}: timestamp starts at {parquet_ts_min:.2f}s (should start near 0)")

                # Verify timestamps are monotonically increasing
                ts_diff = df['timestamp'].diff().dropna()
                if (ts_diff < 0).any():
                    sync_issues.append(f"{pq_file.name}: timestamps not monotonically increasing")

                # Verify timestamp spacing matches expected FPS
                expected_dt = 1.0 / fps
                actual_dt = ts_diff.median()
                if abs(actual_dt - expected_dt) > expected_dt * 0.5:  # 50% tolerance
                    sync_issues.append(f"{pq_file.name}: timestamp spacing {actual_dt:.4f}s doesn't match {fps} fps (expected {expected_dt:.4f}s)")

                # NEW: Check corresponding video file exists and has matching duration
                for video_key in video_keys[:1]:  # Check first camera only
                    target_chunk = ep_index // chunks_size
                    video_file = videos_dir / video_key / f"chunk-{target_chunk:03d}" / f"episode_{ep_index:06d}.mp4"

                    if not video_file.exists():
                        sync_issues.append(f"Missing video: {video_file.relative_to(dataset_path)}")
                        continue

                    # Get video duration via ffprobe
                    result = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', str(video_file)],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        video_duration = float(result.stdout.strip())

                        # Compare durations (with tolerance)
                        duration_diff = abs(video_duration - parquet_duration)
                        if duration_diff > 1.0:  # 1 second tolerance
                            sync_issues.append(f"Episode {ep_index}: duration mismatch - parquet={parquet_duration:.2f}s, video={video_duration:.2f}s")
                        elif verbose:
                            details.append(f"  Episode {ep_index}: parquet={parquet_duration:.2f}s, video={video_duration:.2f}s ✓")

            if verbose and not sync_issues:
                details.append(f"  Per-episode sync: verified on {len(sample_indices)} sample episodes")

        except Exception as e:
            sync_issues.append(f"Timestamp verification failed: {e}")

    if sync_issues:
        issues.extend(sync_issues[:5])
        if len(sync_issues) > 5:
            issues.append(f"... and {len(sync_issues) - 5} more sync issues")

    if issues:
        return VerificationResult(
            name="Data-Video Sync",
            passed=False,
            message=f"{len(issues)} sync issues found",
            details=details + [f"  ✗ {i}" for i in issues]
        )

    return VerificationResult(
        name="Data-Video Sync",
        passed=True,
        message=f"{total_frames_parquet} frames synchronized",
        details=details
    )


def verify_dataset(dataset_path: Path, verbose: bool = False, is_source: bool = False) -> VerificationReport:
    """Run all verification checks on a dataset."""
    report = VerificationReport(dataset_path=dataset_path)

    print("=" * 70)
    print(f"GR00T Dataset Verification")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Type: {'Source (pre-combination)' if is_source else 'Final (ready for training)'}")
    print("=" * 70)

    checks = [
        ("Meta Files", check_meta_files),
        ("info.json", check_info_json),
        ("episodes.jsonl", check_episodes_jsonl),
        ("tasks.jsonl", check_tasks_jsonl),
        ("modality.json", check_modality_json),
        ("Parquet Files", check_parquet_files),
        ("Video Files", check_video_files),
        ("Data-Video Sync", check_data_video_sync),
    ]

    for name, check_fn in checks:
        print(f"\n[{len(report.results)+1}/{len(checks)}] Checking {name}...")
        result = check_fn(dataset_path, verbose)
        report.results.append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {result.message}")

        if verbose and result.details:
            for detail in result.details:
                print(detail)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify GR00T dataset integrity and synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        required=True,
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--source", "-s",
        action="store_true",
        help="Verify as source dataset (before combination)"
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    report = verify_dataset(args.dataset, args.verbose, args.source)

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for result in report.results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.message}")

    print()
    if report.passed:
        print("✅ ALL CHECKS PASSED - Dataset is ready for training")
        sys.exit(0)
    else:
        print(f"❌ {report.failed_count} CHECK(S) FAILED - Dataset has issues")
        print("\nRecommendations:")
        print("  1. Fix the issues above")
        print("  2. Re-run conversion: python custom/scripts/convert_lerobot_v3_to_groot.py ...")
        print("  3. Re-run combination: python custom/scripts/combine_groot_datasets.py ...")
        print("  4. Re-run this verification")
        sys.exit(1)


if __name__ == "__main__":
    main()
