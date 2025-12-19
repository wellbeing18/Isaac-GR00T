#!/usr/bin/env python3
"""
Dataset Bias Analysis Script for GR00T Training Data.

This script analyzes directional bias in robot arm trajectories to detect
potential issues that may cause problems during closed-loop inference.

Key Analysis:
1. Direction bias in first N steps (does arm always go left/right?)
2. Joint range statistics (min/max/mean for each joint)
3. Starting position distribution
4. Visualization of trajectory patterns

Usage:
    python analyze_dataset_bias.py --dataset /path/to/dataset
    python analyze_dataset_bias.py --dataset /path/to/dataset --output-dir analysis_results
    python analyze_dataset_bias.py --dataset /path/to/dataset --steps 100 --joint 0

Reference: Investigation of "swing left" issue in inference
    - See: investigation_swing_left_bias.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def import_modality_config(config_path: str = None):
    """Import the modality config module to register NEW_EMBODIMENT."""
    if config_path is None:
        config_path = PROJECT_ROOT / "custom/scripts/ver1_6/so101_config_1_6.py"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {config_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_dataset(dataset_path: str):
    """Load dataset using LeRobotEpisodeLoader."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    modality_config = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    return loader


def analyze_direction_bias(loader, steps: int = 100, joint_idx: int = 0, threshold: float = 5.0,
                           max_episodes: int = None):
    """
    Analyze directional bias in the first N steps of each episode.

    Args:
        loader: Dataset loader
        steps: Number of steps to analyze from start
        joint_idx: Which joint to analyze (0=shoulder_pan for SO-101)
        threshold: Degrees threshold to classify as LEFT/RIGHT vs CENTER
        max_episodes: Maximum number of episodes to analyze (None=all)

    Returns:
        dict with analysis results
    """
    results = {
        "episodes": [],
        "left_count": 0,
        "right_count": 0,
        "center_count": 0,
        "step0_values": [],
        "stepN_values": [],
        "deltas": [],
    }

    num_episodes = len(loader) if max_episodes is None else min(max_episodes, len(loader))

    for ep_id in range(num_episodes):
        traj = loader[ep_id]

        # Get joint values at step 0 and step N
        step0_val = traj['action.single_arm'].iloc[0][joint_idx]
        stepN_idx = min(steps, len(traj) - 1)
        stepN_val = traj['action.single_arm'].iloc[stepN_idx][joint_idx]

        delta = stepN_val - step0_val

        # Classify direction
        if delta < -threshold:
            direction = "LEFT"
            results["left_count"] += 1
        elif delta > threshold:
            direction = "RIGHT"
            results["right_count"] += 1
        else:
            direction = "CENTER"
            results["center_count"] += 1

        results["episodes"].append({
            "ep_id": ep_id,
            "step0": step0_val,
            "stepN": stepN_val,
            "delta": delta,
            "direction": direction,
        })
        results["step0_values"].append(step0_val)
        results["stepN_values"].append(stepN_val)
        results["deltas"].append(delta)

    results["total"] = len(loader)
    results["left_pct"] = 100 * results["left_count"] / results["total"]
    results["right_pct"] = 100 * results["right_count"] / results["total"]
    results["center_pct"] = 100 * results["center_count"] / results["total"]

    return results


def analyze_joint_ranges(loader, max_episodes: int = None):
    """
    Analyze min/max/mean for all joints across all episodes.

    Args:
        loader: Dataset loader
        max_episodes: Maximum number of episodes to analyze (None=all)

    Returns:
        dict with joint statistics
    """
    all_actions = []
    all_states = []

    num_episodes = len(loader) if max_episodes is None else min(max_episodes, len(loader))

    for ep_id in range(num_episodes):
        traj = loader[ep_id]
        actions = np.array([x for x in traj['action.single_arm']])
        states = np.array([x for x in traj['state.single_arm']])
        all_actions.append(actions)
        all_states.append(states)

    all_actions = np.vstack(all_actions)
    all_states = np.vstack(all_states)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    results = {
        "action": {},
        "state": {},
    }

    for i, name in enumerate(joint_names):
        results["action"][name] = {
            "min": float(np.min(all_actions[:, i])),
            "max": float(np.max(all_actions[:, i])),
            "mean": float(np.mean(all_actions[:, i])),
            "std": float(np.std(all_actions[:, i])),
        }
        results["state"][name] = {
            "min": float(np.min(all_states[:, i])),
            "max": float(np.max(all_states[:, i])),
            "mean": float(np.mean(all_states[:, i])),
            "std": float(np.std(all_states[:, i])),
        }

    return results


def create_bias_visualization(bias_results: dict, joint_ranges: dict, output_path: str,
                              joint_name: str = "shoulder_pan", steps: int = 100):
    """Create visualization of dataset bias."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    step0_values = bias_results["step0_values"]
    stepN_values = bias_results["stepN_values"]
    deltas = bias_results["deltas"]

    # Plot 1: Step 0 to Step N trajectory for each episode
    ax = axes[0, 0]
    for ep in bias_results["episodes"]:
        color = 'blue' if ep["direction"] == "LEFT" else ('red' if ep["direction"] == "RIGHT" else 'gray')
        ax.plot([0, steps], [ep["step0"], ep["stepN"]], color=color, alpha=0.5)
        ax.scatter([0, steps], [ep["step0"], ep["stepN"]], color=color, s=20)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Center')
    ax.set_title(f'Dataset: {joint_name} from Step 0 to Step {steps}\n(Blue=LEFT, Red=RIGHT, Gray=CENTER)')
    ax.set_xlabel('Step')
    ax.set_ylabel(f'{joint_name} (degrees)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Histogram of stepN values
    ax = axes[0, 1]
    ax.hist(stepN_values, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Center (0°)')
    ax.axvline(x=np.mean(stepN_values), color='blue', linestyle='-', linewidth=2,
               label=f'Mean ({np.mean(stepN_values):.1f}°)')
    ax.set_title(f'Distribution of {joint_name} at Step {steps}')
    ax.set_xlabel(f'{joint_name} (degrees)')
    ax.set_ylabel('Count')
    ax.legend()

    # Plot 3: Delta histogram per episode
    ax = axes[1, 0]
    colors = ['blue' if d < -5 else ('red' if d > 5 else 'gray') for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axhline(y=-5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'Delta (step{steps} - step0) for each episode\n(Blue=LEFT, Red=RIGHT, Gray=CENTER)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Delta (degrees)')

    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    left_count = bias_results["left_count"]
    right_count = bias_results["right_count"]
    center_count = bias_results["center_count"]
    total = bias_results["total"]

    bars = ax.bar(['LEFT\n(negative)', 'CENTER', 'RIGHT\n(positive)'],
                  [left_count, center_count, right_count],
                  color=['blue', 'gray', 'red'], alpha=0.7)
    ax.set_title(f'Direction Bias in First {steps} Steps\n({left_count}/{total} = {bias_results["left_pct"]:.0f}% go LEFT)')
    ax.set_ylabel('Number of Episodes')

    # Add percentage labels
    for bar, count in zip(bars, [left_count, center_count, right_count]):
        pct = 100 * count / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({pct:.0f}%)', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def print_report(bias_results: dict, joint_ranges: dict, steps: int = 100, joint_name: str = "shoulder_pan"):
    """Print analysis report to console."""

    print("=" * 70)
    print("DATASET BIAS ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n1. DIRECTION BIAS (first {steps} steps, {joint_name}):")
    print("-" * 70)
    print(f"   LEFT (negative):  {bias_results['left_count']:3d} / {bias_results['total']} ({bias_results['left_pct']:.0f}%)")
    print(f"   CENTER:           {bias_results['center_count']:3d} / {bias_results['total']} ({bias_results['center_pct']:.0f}%)")
    print(f"   RIGHT (positive): {bias_results['right_count']:3d} / {bias_results['total']} ({bias_results['right_pct']:.0f}%)")

    # Bias warning
    if bias_results['left_pct'] > 70 or bias_results['right_pct'] > 70:
        dominant = "LEFT" if bias_results['left_pct'] > bias_results['right_pct'] else "RIGHT"
        pct = max(bias_results['left_pct'], bias_results['right_pct'])
        print(f"\n   ⚠️  WARNING: Strong {dominant} bias detected ({pct:.0f}%)!")
        print(f"       This may cause closed-loop inference issues.")
        print(f"       Consider: data augmentation (horizontal flip) or collecting more diverse data.")

    print(f"\n2. {joint_name.upper()} STATISTICS:")
    print("-" * 70)
    stats = joint_ranges["action"][joint_name]
    print(f"   Action range: [{stats['min']:.1f}°, {stats['max']:.1f}°]")
    print(f"   Action mean:  {stats['mean']:.1f}° ± {stats['std']:.1f}°")

    print(f"\n   Step 0 mean:    {np.mean(bias_results['step0_values']):.1f}°")
    print(f"   Step {steps} mean: {np.mean(bias_results['stepN_values']):.1f}°")
    print(f"   Delta mean:     {np.mean(bias_results['deltas']):+.1f}°")

    print(f"\n3. ALL JOINT RANGES (action):")
    print("-" * 70)
    for joint, stats in joint_ranges["action"].items():
        print(f"   {joint:15s}: [{stats['min']:7.1f}°, {stats['max']:7.1f}°] mean={stats['mean']:6.1f}°")

    print("\n" + "=" * 70)

    print("\n4. PER-EPISODE DETAILS:")
    print("-" * 70)
    for ep in bias_results["episodes"]:
        print(f"   Ep{ep['ep_id']:2d}: step0={ep['step0']:6.1f}, step{steps}={ep['stepN']:6.1f}, "
              f"delta={ep['delta']:+6.1f} -> {ep['direction']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset bias for GR00T training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_dataset_bias.py --dataset datasets/so101_pick_place_groot

    # Custom output directory
    python analyze_dataset_bias.py --dataset datasets/my_dataset --output-dir my_analysis

    # Analyze different number of steps
    python analyze_dataset_bias.py --dataset datasets/my_dataset --steps 50
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot",
        help="Path to dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as script)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=100,
        help="Number of steps to analyze from start (default: 100)"
    )
    parser.add_argument(
        "--joint", "-j",
        type=int,
        default=0,
        help="Joint index to analyze (default: 0 = shoulder_pan)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to modality config file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--max-episodes", "-m",
        type=int,
        default=None,
        help="Max episodes to analyze (default: all)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    joint_name = joint_names[args.joint] if args.joint < len(joint_names) else f"joint_{args.joint}"

    print(f"Loading dataset: {args.dataset}")
    print(f"Analyzing: {joint_name} (joint {args.joint}) over first {args.steps} steps")
    if args.max_episodes:
        print(f"Limiting to first {args.max_episodes} episodes")
    print()

    # Import config and load dataset
    import_modality_config(args.config)
    loader = load_dataset(args.dataset)

    num_episodes = len(loader) if args.max_episodes is None else min(args.max_episodes, len(loader))
    print(f"Dataset loaded: {len(loader)} episodes (analyzing {num_episodes})")

    # Run analysis
    bias_results = analyze_direction_bias(loader, steps=args.steps, joint_idx=args.joint,
                                          max_episodes=args.max_episodes)
    joint_ranges = analyze_joint_ranges(loader, max_episodes=args.max_episodes)

    # Create visualization
    plot_path = output_dir / "dataset_bias_analysis.png"
    create_bias_visualization(bias_results, joint_ranges, str(plot_path), joint_name, args.steps)
    print(f"\nVisualization saved to: {plot_path}")

    # Print report
    if not args.quiet:
        print()
        print_report(bias_results, joint_ranges, args.steps, joint_name)

    # Return exit code based on bias severity
    if bias_results['left_pct'] > 80 or bias_results['right_pct'] > 80:
        print("\n⚠️  SEVERE BIAS DETECTED - Consider data augmentation before training!")
        return 1
    elif bias_results['left_pct'] > 70 or bias_results['right_pct'] > 70:
        print("\n⚠️  MODERATE BIAS DETECTED - May cause inference issues")
        return 0
    else:
        print("\n✓ Dataset appears reasonably balanced")
        return 0


if __name__ == "__main__":
    sys.exit(main())
