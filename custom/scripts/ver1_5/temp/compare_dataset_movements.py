#!/usr/bin/env python3
"""
Compare Dataset Movements: Your Data vs Reference Data

Generates a visualization comparing joint movements from your dataset
against a reference dataset (e.g., youliangtan/so101-table-cleanup).

This is useful for debugging data quality issues by visualizing:
- Movement timing (when does motion start?)
- Movement velocity (how fast are demonstrations?)
- Joint range coverage

Usage:
    # Compare with default reference dataset
    python compare_dataset_movements.py \
        --your-dataset /path/to/your/dataset

    # Compare specific episodes
    python compare_dataset_movements.py \
        --your-dataset /path/to/your/dataset \
        --your-episode 0 \
        --ref-episode 0

    # Specify output path
    python compare_dataset_movements.py \
        --your-dataset /path/to/your/dataset \
        --output custom/jdocs/lora/new_investigations/pics/dataset_movement_comparison.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Joint names for SO-101 robot
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def download_reference_dataset(repo_id: str, local_path: Path) -> Path:
    """Download reference dataset from HuggingFace if needed."""
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


def load_episode_data(dataset_path: Path, episode_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load state and action data for a specific episode.

    Returns:
        states: (T, 6) array of joint states
        actions: (T, 6) array of joint actions
    """
    # Try different parquet path structures
    data_dir = dataset_path / "data" / "chunk-000"
    if not data_dir.exists():
        data_dir = dataset_path / "data"

    parquet_path = data_dir / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        # Try finding all parquets and filtering
        all_parquets = sorted(data_dir.glob("*.parquet"))
        if not all_parquets:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Load all and filter by episode_index
        dfs = [pd.read_parquet(p) for p in all_parquets]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    else:
        df = pd.read_parquet(parquet_path)

    if len(df) == 0:
        raise ValueError(f"No data found for episode {episode_idx}")

    # Extract states and actions
    # Handle different column formats
    if "observation.state" in df.columns:
        states = np.stack(df["observation.state"].values)
    elif "state" in df.columns:
        states = np.stack(df["state"].values)
    else:
        raise KeyError("Could not find state column")

    if "action" in df.columns:
        actions = np.stack(df["action"].values)
    else:
        raise KeyError("Could not find action column")

    return states, actions


def plot_dataset_comparison(
    your_states: np.ndarray,
    your_actions: np.ndarray,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    your_name: str,
    ref_name: str,
    your_episode: int,
    ref_episode: int,
    output_path: Path,
    use_actions: bool = True,
):
    """
    Generate SIDE-BY-SIDE comparison plot of dataset movements.

    Layout: 6 rows (joints) x 2 columns (Your Data | Reference Data)

    Args:
        your_states: Your dataset states (T1, 6)
        your_actions: Your dataset actions (T1, 6)
        ref_states: Reference dataset states (T2, 6)
        ref_actions: Reference dataset actions (T2, 6)
        your_name: Name for your dataset
        ref_name: Name for reference dataset
        your_episode: Your episode index
        ref_episode: Reference episode index
        output_path: Where to save the plot
        use_actions: If True, plot actions; if False, plot states
    """
    # Choose data to plot
    if use_actions:
        your_data = your_actions
        ref_data = ref_actions
        data_type = "Actions"
    else:
        your_data = your_states
        ref_data = ref_states
        data_type = "States"

    # Create figure with 6 rows x 2 columns (side by side)
    fig, axes = plt.subplots(6, 2, figsize=(20, 18))

    fig.suptitle(
        f"Dataset Comparison: Your Data vs Reference Data (Episode {your_episode})",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Time axes
    your_time = np.arange(len(your_data)) / 30.0  # Assuming 30Hz
    ref_time = np.arange(len(ref_data)) / 30.0

    # Colors
    your_color = '#1f77b4'  # Blue
    ref_color = '#ff7f0e'   # Orange

    # Calculate motion start for both datasets
    def find_motion_start(data, threshold=0.5):
        vel = np.abs(np.diff(data, axis=0))
        max_vel = np.max(vel, axis=1)
        motion_frames = np.where(max_vel > threshold)[0]
        return motion_frames[0] if len(motion_frames) > 0 else len(data)

    your_motion_start = find_motion_start(your_data)
    ref_motion_start = find_motion_start(ref_data)

    for i, joint_name in enumerate(JOINT_NAMES):
        # Left column: Your Data
        ax_your = axes[i, 0]
        ax_your.plot(
            your_time, your_data[:, i],
            color=your_color,
            linewidth=1.5,
            alpha=0.9
        )

        # Mark motion start
        ax_your.axvline(
            x=your_motion_start / 30.0,
            color='red',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            label=f'Motion start: {your_motion_start/30:.1f}s'
        )

        # Calculate statistics for your data
        your_range = your_data[:, i].max() - your_data[:, i].min()
        your_vel = np.mean(np.abs(np.diff(your_data[your_motion_start:, i]))) if your_motion_start < len(your_data) - 1 else 0

        ax_your.set_ylabel(f"{joint_name}\n(degrees)", fontsize=11)
        ax_your.set_title(
            f"Your Data - {joint_name}\nRange: {your_range:.1f}° | Vel: {your_vel:.2f}°/frame",
            fontsize=10,
            fontweight='bold'
        )
        ax_your.grid(True, alpha=0.3)
        ax_your.legend(loc='upper right', fontsize=8)

        # Right column: Reference Data
        ax_ref = axes[i, 1]
        ax_ref.plot(
            ref_time, ref_data[:, i],
            color=ref_color,
            linewidth=1.5,
            alpha=0.9
        )

        # Mark motion start
        ax_ref.axvline(
            x=ref_motion_start / 30.0,
            color='red',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            label=f'Motion start: {ref_motion_start/30:.1f}s'
        )

        # Calculate statistics for reference data
        ref_range = ref_data[:, i].max() - ref_data[:, i].min()
        ref_vel = np.mean(np.abs(np.diff(ref_data[ref_motion_start:, i]))) if ref_motion_start < len(ref_data) - 1 else 0

        ax_ref.set_title(
            f"Reference Data - {joint_name}\nRange: {ref_range:.1f}° | Vel: {ref_vel:.2f}°/frame",
            fontsize=10,
            fontweight='bold'
        )
        ax_ref.grid(True, alpha=0.3)
        ax_ref.legend(loc='upper right', fontsize=8)

        # Only show x-label on bottom row
        if i == len(JOINT_NAMES) - 1:
            ax_your.set_xlabel("Time (seconds)", fontsize=11)
            ax_ref.set_xlabel("Time (seconds)", fontsize=11)

    # Add column headers
    axes[0, 0].annotate(
        f'YOUR DATA: {your_name}',
        xy=(0.5, 1.15), xycoords='axes fraction',
        fontsize=12, fontweight='bold', color=your_color,
        ha='center', va='bottom'
    )
    axes[0, 1].annotate(
        f'REFERENCE: {ref_name}',
        xy=(0.5, 1.15), xycoords='axes fraction',
        fontsize=12, fontweight='bold', color=ref_color,
        ha='center', va='bottom'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def print_comparison_stats(
    your_states: np.ndarray,
    your_actions: np.ndarray,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
):
    """Print comparison statistics."""
    print("\n" + "=" * 70)
    print("DATASET COMPARISON STATISTICS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Your Dataset':<20} {'Reference':<20} {'Diff'}")
    print("-" * 70)

    # Episode length
    print(f"{'Episode length':<20} {len(your_actions):<20} {len(ref_actions):<20} {len(your_actions) - len(ref_actions):+d}")
    print(f"{'Duration (s)':<20} {len(your_actions)/30:<20.1f} {len(ref_actions)/30:<20.1f} {(len(your_actions) - len(ref_actions))/30:+.1f}")

    # Movement detection (when does significant motion start?)
    def find_motion_start(data, threshold=0.5):
        """Find first frame where max joint velocity exceeds threshold."""
        vel = np.abs(np.diff(data, axis=0))
        max_vel = np.max(vel, axis=1)
        motion_frames = np.where(max_vel > threshold)[0]
        return motion_frames[0] if len(motion_frames) > 0 else len(data)

    your_start = find_motion_start(your_actions)
    ref_start = find_motion_start(ref_actions)

    print(f"\n{'Motion starts at':<20} {f'frame {your_start} ({your_start/30:.1f}s)':<20} {f'frame {ref_start} ({ref_start/30:.1f}s)':<20}")

    # Average velocity during active motion
    your_vel = np.mean(np.abs(np.diff(your_actions[your_start:], axis=0)))
    ref_vel = np.mean(np.abs(np.diff(ref_actions[ref_start:], axis=0)))

    print(f"{'Avg velocity':<20} {f'{your_vel:.3f}°/frame':<20} {f'{ref_vel:.3f}°/frame':<20} {your_vel - ref_vel:+.3f}")

    # Joint ranges
    print(f"\n{'Joint Ranges (degrees):'}")
    print(f"{'Joint':<15} {'Your Range':<15} {'Ref Range':<15} {'Diff'}")
    print("-" * 55)

    for i, name in enumerate(JOINT_NAMES):
        your_range = your_actions[:, i].max() - your_actions[:, i].min()
        ref_range = ref_actions[:, i].max() - ref_actions[:, i].min()
        diff = your_range - ref_range
        print(f"{name:<15} {your_range:<15.1f} {ref_range:<15.1f} {diff:+.1f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare dataset movements: Your Data vs Reference Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--your-dataset", "-d",
        type=Path,
        default=Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place"),
        help="Path to your dataset",
    )
    parser.add_argument(
        "--your-episode",
        type=int,
        default=0,
        help="Episode index from your dataset (default: 0)",
    )
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="youliangtan/so101-table-cleanup",
        help="Reference dataset HuggingFace repo ID (default: youliangtan/so101-table-cleanup)",
    )
    parser.add_argument(
        "--ref-episode",
        type=int,
        default=0,
        help="Episode index from reference dataset (default: 0)",
    )
    parser.add_argument(
        "--ref-local-path",
        type=Path,
        default=Path("/tmp/so101-table-cleanup"),
        help="Local path to store/find reference dataset",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("custom/jdocs/lora/new_investigations/pics/dataset_movement_comparison.png"),
        help="Output path for comparison plot",
    )
    parser.add_argument(
        "--use-states",
        action="store_true",
        help="Plot states instead of actions",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DATASET MOVEMENT COMPARISON")
    print("=" * 70)
    print(f"Your dataset: {args.your_dataset}")
    print(f"Your episode: {args.your_episode}")
    print(f"Reference: {args.ref_dataset}")
    print(f"Reference episode: {args.ref_episode}")
    print("=" * 70)

    # Load your data
    print(f"\nLoading your dataset...")
    try:
        your_states, your_actions = load_episode_data(args.your_dataset, args.your_episode)
        print(f"  Loaded episode {args.your_episode}: {len(your_states)} frames")
    except Exception as e:
        print(f"Error loading your dataset: {e}")
        sys.exit(1)

    # Download/load reference data
    print(f"\nLoading reference dataset...")
    try:
        ref_path = download_reference_dataset(args.ref_dataset, args.ref_local_path)
        ref_states, ref_actions = load_episode_data(ref_path, args.ref_episode)
        print(f"  Loaded episode {args.ref_episode}: {len(ref_states)} frames")
    except Exception as e:
        print(f"Error loading reference dataset: {e}")
        sys.exit(1)

    # Print comparison statistics
    print_comparison_stats(your_states, your_actions, ref_states, ref_actions)

    # Generate comparison plot
    print(f"\nGenerating comparison plot...")
    plot_dataset_comparison(
        your_states=your_states,
        your_actions=your_actions,
        ref_states=ref_states,
        ref_actions=ref_actions,
        your_name=args.your_dataset.name,
        ref_name=args.ref_dataset.split("/")[-1],
        your_episode=args.your_episode,
        ref_episode=args.ref_episode,
        output_path=args.output,
        use_actions=not args.use_states,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
