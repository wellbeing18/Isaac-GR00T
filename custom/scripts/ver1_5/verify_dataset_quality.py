#!/usr/bin/env python3
"""
Dataset Quality Verification Tool for LeRobot/GR00T Training

This script verifies collected dataset quality by analyzing:
1. Movement patterns - detecting stationary periods at start/end
2. Action velocity - checking if movement speed is appropriate for 30Hz
3. Joint coverage - ensuring all joints are exercised
4. Comparison with reference datasets (e.g., youliangtan/so101-table-cleanup)

It also provides commands for visual verification using LeRobot tools.

Usage:
    # Basic quality check
    python custom/scripts/verify_dataset_quality.py \
        --dataset /path/to/your/dataset

    # Compare with reference dataset
    python custom/scripts/verify_dataset_quality.py \
        --dataset /path/to/your/dataset \
        --reference youliangtan/so101-table-cleanup

    # Generate visualization commands
    python custom/scripts/verify_dataset_quality.py \
        --dataset /path/to/your/dataset \
        --show-viz-commands
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Joint names for SO-101 robot
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Quality thresholds
THRESHOLDS = {
    "max_stationary_start_percent": 10.0,  # Max % of episode that can be stationary at start
    "max_stationary_end_percent": 10.0,    # Max % of episode that can be stationary at end
    "min_movement_threshold": 0.5,          # Degrees - movement below this is "stationary"
    "ideal_velocity_range": (0.5, 5.0),     # Degrees/frame - ideal movement speed at 30Hz
    "min_joint_range": 10.0,                # Degrees - minimum range each joint should cover
}


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load LeRobot dataset from parquet files."""
    data_dir = dataset_path / "data" / "chunk-000"

    if not data_dir.exists():
        # Try alternative structure
        data_dir = dataset_path / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found in {dataset_path}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Import here to handle different environments
    try:
        import pyarrow.parquet as pq
        dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
    except Exception:
        dfs = [pd.read_parquet(f) for f in parquet_files]

    return pd.concat(dfs, ignore_index=True)


def download_reference_dataset(repo_id: str, local_path: Path) -> Path:
    """Download reference dataset from HuggingFace."""
    if local_path.exists() and (local_path / "data").exists():
        print(f"Reference dataset already exists at {local_path}")
        return local_path

    print(f"Downloading reference dataset: {repo_id}")
    local_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        repo_id,
        "--local-dir", str(local_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Downloaded to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {e.stderr.decode()}")
        raise

    return local_path


def analyze_episode(episode_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Analyze a single episode for quality metrics."""
    actions = np.stack(episode_df["action"].values)
    n_frames = len(actions)

    # Calculate per-frame movements
    diffs = np.abs(np.diff(actions, axis=0))
    max_diffs = np.max(diffs, axis=1)  # Max movement across all joints per frame

    # Find first and last significant movement
    movement_frames = np.where(max_diffs > threshold)[0]

    if len(movement_frames) == 0:
        first_movement = n_frames
        last_movement = 0
    else:
        first_movement = movement_frames[0]
        last_movement = movement_frames[-1] + 1

    # Calculate stationary periods
    start_stationary = first_movement
    end_stationary = n_frames - last_movement - 1
    active_frames = last_movement - first_movement if last_movement > first_movement else 0

    # Calculate joint ranges
    joint_ranges = np.max(actions, axis=0) - np.min(actions, axis=0)

    # Calculate velocity statistics during active period
    if active_frames > 1:
        active_diffs = diffs[first_movement:last_movement]
        avg_velocity = np.mean(active_diffs)
        max_velocity = np.max(active_diffs)
    else:
        avg_velocity = 0
        max_velocity = 0

    return {
        "total_frames": n_frames,
        "start_stationary_frames": start_stationary,
        "end_stationary_frames": end_stationary,
        "active_frames": active_frames,
        "start_stationary_percent": start_stationary / n_frames * 100,
        "end_stationary_percent": end_stationary / n_frames * 100,
        "active_percent": active_frames / n_frames * 100,
        "joint_ranges": joint_ranges,
        "avg_velocity": avg_velocity,
        "max_velocity": max_velocity,
    }


def print_quality_report(stats: dict, thresholds: dict = THRESHOLDS) -> tuple[int, int]:
    """Print quality report and return (warnings, errors) count."""
    warnings = 0
    errors = 0

    print("\n" + "=" * 70)
    print("DATASET QUALITY REPORT")
    print("=" * 70)

    # Episode statistics
    print(f"\nTotal Episodes: {stats['num_episodes']}")
    print(f"Total Frames: {stats['total_frames']}")
    print(f"FPS: {stats.get('fps', 30)}")

    # Stationary period analysis
    print("\n--- STATIONARY PERIOD ANALYSIS ---")
    avg_start = np.mean([e["start_stationary_percent"] for e in stats["episodes"]])
    avg_end = np.mean([e["end_stationary_percent"] for e in stats["episodes"]])
    avg_active = np.mean([e["active_percent"] for e in stats["episodes"]])

    start_status = "OK" if avg_start <= thresholds["max_stationary_start_percent"] else "WARNING"
    end_status = "OK" if avg_end <= thresholds["max_stationary_end_percent"] else "WARNING"

    if start_status == "WARNING":
        warnings += 1
    if end_status == "WARNING":
        warnings += 1

    print(f"Average start stationary: {avg_start:.1f}% [{start_status}] (threshold: <{thresholds['max_stationary_start_percent']}%)")
    print(f"Average end stationary: {avg_end:.1f}% [{end_status}] (threshold: <{thresholds['max_stationary_end_percent']}%)")
    print(f"Average active movement: {avg_active:.1f}%")

    # Per-episode breakdown
    print("\nPer-Episode Breakdown:")
    print(f"{'Ep':<4} {'Frames':<8} {'Start%':<10} {'Active%':<10} {'End%':<10} {'Status'}")
    print("-" * 52)

    for i, ep in enumerate(stats["episodes"][:10]):  # Show first 10
        status = "OK"
        if ep["start_stationary_percent"] > thresholds["max_stationary_start_percent"]:
            status = "WARN:start"
        if ep["end_stationary_percent"] > thresholds["max_stationary_end_percent"]:
            status = "WARN:end" if status == "OK" else status + "+end"

        print(f"{i:<4} {ep['total_frames']:<8} {ep['start_stationary_percent']:<10.1f} "
              f"{ep['active_percent']:<10.1f} {ep['end_stationary_percent']:<10.1f} {status}")

    if len(stats["episodes"]) > 10:
        print(f"... and {len(stats['episodes']) - 10} more episodes")

    # Joint range analysis
    print("\n--- JOINT RANGE ANALYSIS ---")
    avg_ranges = np.mean([e["joint_ranges"] for e in stats["episodes"]], axis=0)

    print(f"{'Joint':<15} {'Avg Range':<12} {'Status'}")
    print("-" * 35)
    for i, name in enumerate(JOINT_NAMES):
        range_val = avg_ranges[i]
        status = "OK" if range_val >= thresholds["min_joint_range"] else "LOW"
        if status == "LOW":
            warnings += 1
        print(f"{name:<15} {range_val:<12.2f}° {status}")

    # Velocity analysis
    print("\n--- VELOCITY ANALYSIS ---")
    avg_vel = np.mean([e["avg_velocity"] for e in stats["episodes"]])
    max_vel = np.max([e["max_velocity"] for e in stats["episodes"]])

    ideal_min, ideal_max = thresholds["ideal_velocity_range"]
    vel_status = "OK" if ideal_min <= avg_vel <= ideal_max else "CHECK"

    print(f"Average velocity: {avg_vel:.3f}°/frame [{vel_status}]")
    print(f"Max velocity: {max_vel:.3f}°/frame")
    print(f"Ideal range: {ideal_min}-{ideal_max}°/frame")

    if avg_vel < ideal_min:
        print("  Note: Movement may be too slow for 30Hz sampling")
    elif avg_vel > ideal_max:
        print("  Note: Movement may be too fast - could miss intermediate positions")

    # Overall status
    print("\n" + "=" * 70)
    if errors > 0:
        print(f"RESULT: {errors} ERRORS, {warnings} WARNINGS")
    elif warnings > 0:
        print(f"RESULT: {warnings} WARNINGS - Review recommendations below")
    else:
        print("RESULT: PASSED - Dataset quality looks good!")
    print("=" * 70)

    return warnings, errors


def compare_with_reference(your_stats: dict, ref_stats: dict, ref_name: str) -> None:
    """Compare your dataset with a reference dataset."""
    print("\n" + "=" * 70)
    print(f"COMPARISON WITH REFERENCE: {ref_name}")
    print("=" * 70)

    your_avg_start = np.mean([e["start_stationary_percent"] for e in your_stats["episodes"]])
    ref_avg_start = np.mean([e["start_stationary_percent"] for e in ref_stats["episodes"]])

    your_avg_active = np.mean([e["active_percent"] for e in your_stats["episodes"]])
    ref_avg_active = np.mean([e["active_percent"] for e in ref_stats["episodes"]])

    your_avg_vel = np.mean([e["avg_velocity"] for e in your_stats["episodes"]])
    ref_avg_vel = np.mean([e["avg_velocity"] for e in ref_stats["episodes"]])

    print(f"\n{'Metric':<25} {'Your Dataset':<15} {'Reference':<15} {'Diff'}")
    print("-" * 65)
    print(f"{'Start stationary %':<25} {your_avg_start:<15.1f} {ref_avg_start:<15.1f} {your_avg_start - ref_avg_start:+.1f}")
    print(f"{'Active movement %':<25} {your_avg_active:<15.1f} {ref_avg_active:<15.1f} {your_avg_active - ref_avg_active:+.1f}")
    print(f"{'Avg velocity (°/frame)':<25} {your_avg_vel:<15.3f} {ref_avg_vel:<15.3f} {your_avg_vel - ref_avg_vel:+.3f}")

    # Joint ranges comparison
    your_ranges = np.mean([e["joint_ranges"] for e in your_stats["episodes"]], axis=0)
    ref_ranges = np.mean([e["joint_ranges"] for e in ref_stats["episodes"]], axis=0)

    print(f"\nJoint Ranges (degrees):")
    print(f"{'Joint':<15} {'Yours':<12} {'Reference':<12} {'Diff'}")
    print("-" * 45)
    for i, name in enumerate(JOINT_NAMES):
        diff = your_ranges[i] - ref_ranges[i]
        print(f"{name:<15} {your_ranges[i]:<12.1f} {ref_ranges[i]:<12.1f} {diff:+.1f}")


def generate_visualizations(stats: dict, output_dir: Path, dataset_name: str) -> None:
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Stationary periods per episode
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = range(len(stats["episodes"]))
    start_pcts = [e["start_stationary_percent"] for e in stats["episodes"]]
    active_pcts = [e["active_percent"] for e in stats["episodes"]]
    end_pcts = [e["end_stationary_percent"] for e in stats["episodes"]]

    # Stacked bar chart
    ax = axes[0, 0]
    ax.bar(episodes, start_pcts, label="Start Stationary", color="red", alpha=0.7)
    ax.bar(episodes, active_pcts, bottom=start_pcts, label="Active", color="green", alpha=0.7)
    bottoms = [s + a for s, a in zip(start_pcts, active_pcts)]
    ax.bar(episodes, end_pcts, bottom=bottoms, label="End Stationary", color="orange", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Percentage")
    ax.set_title("Episode Composition")
    ax.legend()
    ax.axhline(y=THRESHOLDS["max_stationary_start_percent"], color="red", linestyle="--", alpha=0.5)

    # Velocity distribution
    ax = axes[0, 1]
    velocities = [e["avg_velocity"] for e in stats["episodes"]]
    ax.bar(episodes, velocities, color="blue", alpha=0.7)
    ax.axhline(y=THRESHOLDS["ideal_velocity_range"][0], color="green", linestyle="--", label="Ideal Min")
    ax.axhline(y=THRESHOLDS["ideal_velocity_range"][1], color="green", linestyle="--", label="Ideal Max")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Velocity (°/frame)")
    ax.set_title("Movement Velocity per Episode")
    ax.legend()

    # Joint ranges
    ax = axes[1, 0]
    avg_ranges = np.mean([e["joint_ranges"] for e in stats["episodes"]], axis=0)
    colors = ["green" if r >= THRESHOLDS["min_joint_range"] else "red" for r in avg_ranges]
    ax.bar(JOINT_NAMES, avg_ranges, color=colors, alpha=0.7)
    ax.axhline(y=THRESHOLDS["min_joint_range"], color="red", linestyle="--", label=f"Min ({THRESHOLDS['min_joint_range']}°)")
    ax.set_xlabel("Joint")
    ax.set_ylabel("Range (degrees)")
    ax.set_title("Average Joint Range")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    # Active frames
    ax = axes[1, 1]
    active_frames = [e["active_frames"] for e in stats["episodes"]]
    total_frames = [e["total_frames"] for e in stats["episodes"]]
    ax.bar(episodes, total_frames, label="Total", color="lightblue", alpha=0.7)
    ax.bar(episodes, active_frames, label="Active", color="blue", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Frames")
    ax.set_title("Active vs Total Frames")
    ax.legend()

    plt.suptitle(f"Dataset Quality Analysis: {dataset_name}", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"quality_analysis_{dataset_name.replace('/', '_')}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def print_viz_commands(dataset_path: Path, repo_id: Optional[str] = None) -> None:
    """Print commands for visual verification using LeRobot tools."""
    print("\n" + "=" * 70)
    print("VISUAL VERIFICATION COMMANDS")
    print("=" * 70)

    print("\n1. DATASET VISUALIZER (opens web interface):")
    print("-" * 50)
    if repo_id:
        print(f"   lerobot-dataset-viz --repo-id {repo_id}")
    else:
        # Try to infer repo_id from dataset structure
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                inferred_repo = info.get("repo_id", "local/your_dataset")
        else:
            inferred_repo = "local/your_dataset"
        print(f"   lerobot-dataset-viz --repo-id {inferred_repo} --root {dataset_path}")

    print("\n2. REPLAY EPISODE ON ROBOT:")
    print("-" * 50)
    print("   # Make sure robot is connected!")
    print(f"""   lerobot-replay \\
       --robot.type=so101_follower \\
       --robot.port=/dev/ttyACM1 \\
       --robot.id=xlerobot_left_arm \\
       --dataset.repo_id=local/your_dataset \\
       --dataset.root={dataset_path} \\
       --dataset.episode=0""")

    print("\n3. VIEW VIDEOS DIRECTLY:")
    print("-" * 50)
    video_dir = dataset_path / "videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("**/*.mp4"))
        if video_files:
            print(f"   # Found {len(video_files)} video files")
            print(f"   # Example: ffplay \"{video_files[0]}\"")
            for v in video_files[:3]:
                print(f"   ffplay \"{v}\"")
        else:
            print("   # No .mp4 files found - dataset may use images")
    else:
        print(f"   # No videos directory at {video_dir}")

    print("\n4. COMPARE WITH REFERENCE DATASET:")
    print("-" * 50)
    print("""   # Download youliangtan's reference dataset
   huggingface-cli download --repo-type dataset \\
       youliangtan/so101-table-cleanup \\
       --local-dir /tmp/reference_dataset

   # Visualize reference
   lerobot-dataset-viz --repo-id youliangtan/so101-table-cleanup""")


def print_recommendations(warnings: int, stats: dict) -> None:
    """Print recommendations based on analysis."""
    if warnings == 0:
        return

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    avg_start = np.mean([e["start_stationary_percent"] for e in stats["episodes"]])

    if avg_start > THRESHOLDS["max_stationary_start_percent"]:
        print("""
1. REDUCE STATIONARY START PERIOD:
   - Be in starting position DURING the reset phase (before episode starts)
   - When you hear "Recording episode X", recording has ALREADY started
   - Start moving IMMEDIATELY - don't wait to get ready

   Current: {:.1f}% stationary at start
   Target: <{:.1f}%
""".format(avg_start, THRESHOLDS["max_stationary_start_percent"]))

    avg_vel = np.mean([e["avg_velocity"] for e in stats["episodes"]])
    ideal_min, ideal_max = THRESHOLDS["ideal_velocity_range"]

    if avg_vel < ideal_min:
        print(f"""
2. MOVEMENT SPEED:
   - At 30Hz, each frame is ~33ms apart
   - Current avg velocity ({avg_vel:.3f}°/frame) may be too slow
   - Move slightly faster to capture more dynamic motion
   - Ideal range: {ideal_min}-{ideal_max}°/frame
""")

    avg_ranges = np.mean([e["joint_ranges"] for e in stats["episodes"]], axis=0)
    low_joints = [JOINT_NAMES[i] for i, r in enumerate(avg_ranges) if r < THRESHOLDS["min_joint_range"]]

    if low_joints:
        print(f"""
3. JOINT COVERAGE:
   - The following joints have limited range: {', '.join(low_joints)}
   - Ensure demonstrations exercise all joints through meaningful ranges
   - Current min threshold: {THRESHOLDS['min_joint_range']}°
""")


def main():
    parser = argparse.ArgumentParser(
        description="Verify LeRobot dataset quality for GR00T training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        required=True,
        help="Path to your LeRobot dataset",
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        default=None,
        help="Reference dataset to compare against (HuggingFace repo ID, e.g., youliangtan/so101-table-cleanup)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Directory to save visualization plots (default: dataset/quality_report/)",
    )
    parser.add_argument(
        "--show-viz-commands",
        action="store_true",
        help="Show commands for visual verification using LeRobot tools",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=THRESHOLDS["max_stationary_start_percent"],
        help=f"Max acceptable stationary % at start (default: {THRESHOLDS['max_stationary_start_percent']})",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=THRESHOLDS["min_movement_threshold"],
        help=f"Min movement to count as active (degrees, default: {THRESHOLDS['min_movement_threshold']})",
    )

    args = parser.parse_args()

    # Update thresholds from args
    THRESHOLDS["max_stationary_start_percent"] = args.threshold_start
    THRESHOLDS["min_movement_threshold"] = args.movement_threshold

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    try:
        df = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} frames")

    # Analyze each episode
    episodes = sorted(df["episode_index"].unique())
    print(f"Found {len(episodes)} episodes")

    episode_stats = []
    for ep_idx in episodes:
        ep_data = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        stats = analyze_episode(ep_data, threshold=THRESHOLDS["min_movement_threshold"])
        episode_stats.append(stats)

    # Compile overall stats
    dataset_stats = {
        "num_episodes": len(episodes),
        "total_frames": len(df),
        "fps": 30,  # Assumed
        "episodes": episode_stats,
    }

    # Print report
    warnings, errors = print_quality_report(dataset_stats)

    # Compare with reference if provided
    if args.reference:
        ref_path = Path(f"/tmp/reference_dataset_{args.reference.replace('/', '_')}")
        try:
            download_reference_dataset(args.reference, ref_path)
            ref_df = load_dataset(ref_path)

            ref_episodes = sorted(ref_df["episode_index"].unique())
            ref_episode_stats = []
            for ep_idx in ref_episodes:
                ep_data = ref_df[ref_df["episode_index"] == ep_idx].sort_values("frame_index")
                stats = analyze_episode(ep_data, threshold=THRESHOLDS["min_movement_threshold"])
                ref_episode_stats.append(stats)

            ref_stats = {
                "num_episodes": len(ref_episodes),
                "total_frames": len(ref_df),
                "episodes": ref_episode_stats,
            }

            compare_with_reference(dataset_stats, ref_stats, args.reference)
        except Exception as e:
            print(f"\nCould not compare with reference: {e}")

    # Generate visualizations
    output_dir = args.output_dir or (args.dataset / "quality_report")
    generate_visualizations(dataset_stats, output_dir, args.dataset.name)

    # Print recommendations
    print_recommendations(warnings, dataset_stats)

    # Show visualization commands
    if args.show_viz_commands:
        print_viz_commands(args.dataset)

    # Save report
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        # Convert numpy arrays and numpy int types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            return obj

        serializable_stats = {
            "num_episodes": int(dataset_stats["num_episodes"]),
            "total_frames": int(dataset_stats["total_frames"]),
            "fps": int(dataset_stats["fps"]),
            "episodes": [
                {k: convert_to_serializable(v) for k, v in ep.items()}
                for ep in dataset_stats["episodes"]
            ],
            "warnings": warnings,
            "errors": errors,
        }
        json.dump(serializable_stats, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
