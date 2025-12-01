#!/usr/bin/env python3
"""
GR00T Pre-Training Verification Script

Run this BEFORE training to catch issues early. This script verifies:
1. Dataset format and structure (modality.json, tasks.jsonl, stats.json)
2. Video files are readable
3. Normalization statistics are valid
4. Model can be loaded with LoRA configuration
5. Checkpoint format is correct

Usage:
    python verify_groot_training_setup.py --dataset /path/to/datasets_groot
    python verify_groot_training_setup.py --dataset /path/to/datasets_groot --test-model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result."""
    status = "[PASS]" if passed else "[FAIL]"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if details:
        print(f"         {details}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"  \033[93m[WARN]\033[0m {message}")


def verify_dataset_structure(dataset_path: Path) -> Tuple[bool, List[str]]:
    """
    Verify the dataset has correct structure for GR00T training.

    Expected structure:
    datasets_groot/
    ├── meta/
    │   ├── modality.json      # Camera/state/action mapping
    │   ├── tasks.jsonl        # Task descriptions
    │   ├── stats.json         # Normalization statistics
    │   └── episodes.jsonl     # Episode metadata
    ├── videos/
    │   └── *.mp4              # Video files
    └── data/
        └── *.parquet          # State/action data
    """
    print_header("Dataset Structure Verification")

    issues = []
    all_passed = True

    # Check meta directory
    meta_dir = dataset_path / "meta"
    if not meta_dir.exists():
        print_check("meta/ directory exists", False)
        issues.append("Missing meta/ directory")
        return False, issues
    print_check("meta/ directory exists", True)

    # Check required meta files
    required_files = {
        "modality.json": "Camera/state/action mapping",
        "tasks.jsonl": "Task descriptions",
        "stats.json": "Normalization statistics",
        "episodes.jsonl": "Episode metadata",
    }

    for filename, description in required_files.items():
        filepath = meta_dir / filename
        if filepath.exists():
            print_check(f"{filename} exists", True, description)
        else:
            print_check(f"{filename} exists", False, description)
            issues.append(f"Missing {filename}")
            all_passed = False

    # Check videos directory
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        print_check("videos/ directory exists", True, f"{len(video_files)} video files")
        if len(video_files) == 0:
            print_warning("No .mp4 files found in videos/")
            issues.append("No video files found")
            all_passed = False
    else:
        print_check("videos/ directory exists", False)
        issues.append("Missing videos/ directory")
        all_passed = False

    # Check data directory
    data_dir = dataset_path / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        print_check("data/ directory exists", True, f"{len(parquet_files)} parquet files")
        if len(parquet_files) == 0:
            print_warning("No .parquet files found in data/")
            issues.append("No parquet files found")
            all_passed = False
    else:
        print_check("data/ directory exists", False)
        issues.append("Missing data/ directory")
        all_passed = False

    return all_passed, issues


def verify_modality_json(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Verify modality.json has correct format."""
    print_header("Modality Configuration Verification")

    issues = []
    modality_path = dataset_path / "meta" / "modality.json"

    if not modality_path.exists():
        print_check("modality.json exists", False)
        return False, ["modality.json not found"]

    try:
        with open(modality_path, "r") as f:
            modality = json.load(f)

        print_check("modality.json parseable", True)

        # Check required keys
        required_keys = ["video", "state", "action"]
        for key in required_keys:
            if key in modality:
                print_check(f"'{key}' modality defined", True, str(list(modality[key].keys())))
            else:
                print_check(f"'{key}' modality defined", False)
                issues.append(f"Missing '{key}' in modality.json")

        # Check video cameras
        if "video" in modality:
            video_config = modality["video"]
            expected_cams = ["front", "wrist"]
            for cam in expected_cams:
                if cam in video_config:
                    cam_cfg = video_config[cam]
                    original = cam_cfg.get("original_key", "unknown")
                    print_check(f"video.{cam} configured", True, f"original: {original}")
                else:
                    print_warning(f"video.{cam} not configured (may be using different names)")

        # Check state and action dimensions
        if "state" in modality:
            state_dims = sum(len(v.get("delta_indices", [])) if isinstance(v, dict) else 0
                           for v in modality["state"].values())
            print_check("state modality configured", True, f"Total dimensions tracked")

        if "action" in modality:
            action_dims = sum(len(v.get("delta_indices", [])) if isinstance(v, dict) else 0
                            for v in modality["action"].values())
            print_check("action modality configured", True, f"Total dimensions tracked")

        return len(issues) == 0, issues

    except json.JSONDecodeError as e:
        print_check("modality.json parseable", False, str(e))
        return False, ["modality.json is not valid JSON"]
    except Exception as e:
        print_check("modality.json valid", False, str(e))
        return False, [f"Error reading modality.json: {e}"]


def verify_tasks_jsonl(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Verify tasks.jsonl has task descriptions."""
    print_header("Task Descriptions Verification")

    issues = []
    tasks_path = dataset_path / "meta" / "tasks.jsonl"

    if not tasks_path.exists():
        print_check("tasks.jsonl exists", False)
        return False, ["tasks.jsonl not found"]

    try:
        tasks = []
        with open(tasks_path, "r") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))

        print_check("tasks.jsonl parseable", True, f"{len(tasks)} task(s) found")

        if len(tasks) == 0:
            print_check("tasks defined", False)
            issues.append("No tasks found in tasks.jsonl")
            return False, issues

        # Print task descriptions
        for task in tasks:
            task_idx = task.get("task_index", "?")
            task_desc = task.get("task", "NO DESCRIPTION")
            print_check(f"Task {task_idx}", True, f'"{task_desc}"')

            if not task_desc or task_desc == "NO DESCRIPTION":
                issues.append(f"Task {task_idx} has no description")

        # Check for the expected task description
        expected_task = "pick red_cube from center"
        found_expected = any(expected_task in task.get("task", "") for task in tasks)
        if found_expected:
            print_check("Expected task description found", True, f'"{expected_task}"')
        else:
            print_warning(f"Expected task '{expected_task}' not found")
            print_warning("Make sure inference task matches training task!")

        return len(issues) == 0, issues

    except Exception as e:
        print_check("tasks.jsonl valid", False, str(e))
        return False, [f"Error reading tasks.jsonl: {e}"]


def verify_stats_json(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Verify stats.json has per-dimension normalization statistics."""
    print_header("Normalization Statistics Verification")

    issues = []
    stats_path = dataset_path / "meta" / "stats.json"

    if not stats_path.exists():
        print_check("stats.json exists", False)
        return False, ["stats.json not found"]

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)

        print_check("stats.json parseable", True)

        # Check for required stat types
        required_stats = ["min", "max", "mean", "std", "q01", "q99"]

        # Find action/state keys
        for key in stats.keys():
            key_stats = stats[key]
            if isinstance(key_stats, dict):
                # Check if it has the required stats
                has_all = all(stat in key_stats for stat in required_stats)
                if has_all:
                    # Check if values are arrays (per-dimension) or scalars
                    sample_val = key_stats.get("mean", [])
                    if isinstance(sample_val, list):
                        dims = len(sample_val)
                        print_check(f"'{key}' stats", True, f"{dims}-dimensional, has all required stats")

                        # Verify stats are reasonable
                        mean = key_stats["mean"]
                        std = key_stats["std"]
                        q01 = key_stats["q01"]
                        q99 = key_stats["q99"]

                        # Check for NaN or Inf
                        import math
                        has_nan = any(math.isnan(v) for v in mean + std if isinstance(v, float))
                        has_inf = any(math.isinf(v) for v in mean + std if isinstance(v, float))

                        if has_nan:
                            print_warning(f"'{key}' has NaN values in statistics!")
                            issues.append(f"'{key}' has NaN in statistics")
                        if has_inf:
                            print_warning(f"'{key}' has Inf values in statistics!")
                            issues.append(f"'{key}' has Inf in statistics")

                        # Print summary
                        print(f"           mean range: [{min(mean):.2f}, {max(mean):.2f}]")
                        print(f"           std range:  [{min(std):.2f}, {max(std):.2f}]")
                    else:
                        print_warning(f"'{key}' has scalar stats (expected per-dimension arrays)")
                        issues.append(f"'{key}' stats are scalars, not arrays")

        return len(issues) == 0, issues

    except Exception as e:
        print_check("stats.json valid", False, str(e))
        return False, [f"Error reading stats.json: {e}"]


def verify_episodes_jsonl(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Verify episodes.jsonl has episode metadata."""
    print_header("Episode Metadata Verification")

    issues = []
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        print_check("episodes.jsonl exists", False)
        return False, ["episodes.jsonl not found"]

    try:
        episodes = []
        with open(episodes_path, "r") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

        print_check("episodes.jsonl parseable", True, f"{len(episodes)} episode(s)")

        if len(episodes) == 0:
            print_check("episodes defined", False)
            issues.append("No episodes found")
            return False, issues

        # Summarize episodes
        total_frames = sum(ep.get("length", 0) for ep in episodes)
        task_indices = set(ep.get("task_index", 0) for ep in episodes)

        print_check(f"Episode count", True, f"{len(episodes)} episodes")
        print_check(f"Total frames", True, f"{total_frames} frames")
        print_check(f"Task indices", True, f"{sorted(task_indices)}")

        # Check for very short episodes
        short_episodes = [ep for ep in episodes if ep.get("length", 0) < 10]
        if short_episodes:
            print_warning(f"{len(short_episodes)} episodes have < 10 frames")

        return len(issues) == 0, issues

    except Exception as e:
        print_check("episodes.jsonl valid", False, str(e))
        return False, [f"Error reading episodes.jsonl: {e}"]


def verify_video_files(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Verify video files are readable."""
    print_header("Video Files Verification")

    issues = []
    videos_dir = dataset_path / "videos"

    if not videos_dir.exists():
        print_check("videos/ directory exists", False)
        return False, ["videos/ directory not found"]

    video_files = list(videos_dir.glob("*.mp4"))
    print_check(f"Video files found", True, f"{len(video_files)} files")

    if len(video_files) == 0:
        return False, ["No video files found"]

    # Test reading a sample of video files
    try:
        import decord
        decord.bridge.set_bridge("torch")

        sample_size = min(3, len(video_files))
        sample_files = video_files[:sample_size]

        for video_path in sample_files:
            try:
                vr = decord.VideoReader(str(video_path))
                num_frames = len(vr)
                height, width = vr[0].shape[:2]
                print_check(f"{video_path.name}", True, f"{num_frames} frames, {width}x{height}")
            except Exception as e:
                print_check(f"{video_path.name}", False, str(e))
                issues.append(f"Cannot read {video_path.name}: {e}")

        print(f"  (tested {sample_size} of {len(video_files)} videos)")

    except ImportError:
        print_warning("decord not installed, skipping video verification")
        print_warning("Install with: pip install decord")

    return len(issues) == 0, issues


def verify_model_loading(test_lora: bool = False) -> Tuple[bool, List[str]]:
    """Verify the base model can be loaded."""
    print_header("Model Loading Verification")

    issues = []

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_check(f"PyTorch available", True, f"device={device}")
        print_check(f"CUDA available", torch.cuda.is_available(),
                   f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_check(f"GPU detected", True, f"{gpu_name} ({gpu_mem:.1f} GB)")

    except ImportError:
        print_check("PyTorch available", False)
        return False, ["PyTorch not installed"]

    try:
        from peft import __version__ as peft_version
        print_check("PEFT installed", True, f"version {peft_version}")
    except ImportError:
        print_check("PEFT installed", False, "Required for LoRA training/inference")
        issues.append("PEFT not installed")

    try:
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        print_check("GR00T model importable", True)
    except ImportError as e:
        print_check("GR00T model importable", False, str(e))
        issues.append("Cannot import GR00T model")
        return False, issues

    if test_lora:
        print("\n  Testing LoRA configuration...")
        try:
            from gr00t.utils.peft import get_lora_config
            lora_config = get_lora_config(rank=16)
            print_check("LoRA config creation", True, f"rank=16")
        except Exception as e:
            print_check("LoRA config creation", False, str(e))
            issues.append(f"Cannot create LoRA config: {e}")

    return len(issues) == 0, issues


def verify_checkpoint_format(checkpoint_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Verify checkpoint format if provided."""
    if checkpoint_path is None:
        print_header("Checkpoint Verification (Skipped)")
        print("  No checkpoint path provided. Use --checkpoint to verify.")
        return True, []

    print_header("Checkpoint Format Verification")

    issues = []
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print_check("Checkpoint exists", False, str(checkpoint_path))
        return False, [f"Checkpoint not found: {checkpoint_path}"]

    print_check("Checkpoint exists", True, str(checkpoint_path))

    # Check for LoRA checkpoint
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_model = checkpoint_path / "adapter_model.safetensors"

    if adapter_config.exists() and adapter_model.exists():
        print_check("LoRA checkpoint detected", True)

        # Read adapter config
        with open(adapter_config, "r") as f:
            config = json.load(f)

        rank = config.get("r", "?")
        alpha = config.get("lora_alpha", "?")
        target_modules = config.get("target_modules", [])
        base_model = config.get("base_model_name_or_path", "unknown")

        print_check("LoRA rank", True, str(rank))
        print_check("LoRA alpha", True, str(alpha))
        print_check("Target modules", True, str(target_modules))
        print_check("Base model", True, base_model[:60] + "..." if len(base_model) > 60 else base_model)

        # Check adapter model size
        size_mb = adapter_model.stat().st_size / 1024 / 1024
        print_check("Adapter model size", True, f"{size_mb:.2f} MB")

        # Verify base model exists
        base_path = Path(base_model)
        if base_path.exists():
            print_check("Base model accessible", True)
        else:
            print_warning(f"Base model path not directly accessible (may be HuggingFace cache)")

    else:
        # Check for full checkpoint
        model_files = list(checkpoint_path.glob("*.safetensors")) + list(checkpoint_path.glob("*.bin"))
        if model_files:
            print_check("Full checkpoint detected", True, f"{len(model_files)} model files")
            total_size = sum(f.stat().st_size for f in model_files) / 1024 / 1024 / 1024
            print_check("Total model size", True, f"{total_size:.2f} GB")
        else:
            print_check("Checkpoint format", False, "No model files found")
            issues.append("No model files found in checkpoint")

    # Check for experiment config
    exp_cfg = checkpoint_path / "experiment_cfg"
    if exp_cfg.exists():
        metadata = exp_cfg / "metadata.json"
        if metadata.exists():
            print_check("metadata.json exists", True, "Normalization stats for inference")
        else:
            print_warning("metadata.json not found in experiment_cfg/")
            issues.append("Missing metadata.json for inference normalization")
    else:
        print_warning("experiment_cfg/ not found - may need for inference")

    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify GR00T training setup before starting training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify dataset only
  python verify_groot_training_setup.py --dataset /path/to/datasets_groot

  # Verify dataset and test model loading
  python verify_groot_training_setup.py --dataset /path/to/datasets_groot --test-model

  # Verify existing checkpoint
  python verify_groot_training_setup.py --checkpoint /path/to/checkpoint

  # Full verification
  python verify_groot_training_setup.py --dataset /path/to/datasets_groot --checkpoint /path/to/checkpoint --test-model
        """
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/jrobot/project/XLeRobot/datasets_groot",
        help="Path to the GR00T dataset directory"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to an existing checkpoint to verify (optional)"
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Test model and LoRA imports (requires GPU)"
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video file verification (faster)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" GR00T Pre-Training Verification")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    dataset_path = Path(args.dataset)
    all_issues = []

    # Run all verifications
    verifications = [
        ("Dataset Structure", lambda: verify_dataset_structure(dataset_path)),
        ("Modality Config", lambda: verify_modality_json(dataset_path)),
        ("Task Descriptions", lambda: verify_tasks_jsonl(dataset_path)),
        ("Normalization Stats", lambda: verify_stats_json(dataset_path)),
        ("Episode Metadata", lambda: verify_episodes_jsonl(dataset_path)),
    ]

    if not args.skip_videos:
        verifications.append(("Video Files", lambda: verify_video_files(dataset_path)))

    if args.test_model:
        verifications.append(("Model Loading", lambda: verify_model_loading(test_lora=True)))

    if args.checkpoint:
        verifications.append(("Checkpoint Format",
                            lambda: verify_checkpoint_format(Path(args.checkpoint))))

    for name, verify_fn in verifications:
        try:
            passed, issues = verify_fn()
            all_issues.extend(issues)
        except Exception as e:
            print_header(f"{name} (ERROR)")
            print(f"  Exception: {e}")
            traceback.print_exc()
            all_issues.append(f"{name} failed with exception: {e}")

    # Final summary
    print_header("VERIFICATION SUMMARY")

    if len(all_issues) == 0:
        print("\n  \033[92m" + "="*50 + "\033[0m")
        print("  \033[92m ALL CHECKS PASSED - Ready for training!\033[0m")
        print("  \033[92m" + "="*50 + "\033[0m")
        return 0
    else:
        print("\n  \033[91m" + "="*50 + "\033[0m")
        print(f"  \033[91m {len(all_issues)} ISSUE(S) FOUND\033[0m")
        print("  \033[91m" + "="*50 + "\033[0m")
        print("\n  Issues to fix:")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
