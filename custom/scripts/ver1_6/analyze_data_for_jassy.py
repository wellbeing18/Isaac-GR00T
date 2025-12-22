#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for Jassy's SO-ARM101 Pick-and-Place Task

Analyzes:
1. Gripper timing patterns (when does gripper open relative to approach)
2. Velocity profiles per phase
3. Grasp position distribution
4. Directional bias
5. Pre-grasp pause detection
6. Comparison with baseline datasets

Usage:
    python analyze_data_for_jassy.py --dataset /path/to/dataset --output-dir /path/to/output
    python analyze_data_for_jassy.py  # Uses default paths
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class GripperEvent:
    """Represents a gripper open/close event."""
    frame: int
    event_type: str  # 'open' or 'close'
    gripper_value: float
    arm_position: np.ndarray  # 5 arm joint values
    arm_velocity: float  # Total arm velocity at this moment


@dataclass
class GraspAttempt:
    """Represents a single grasp attempt within an episode."""
    episode_id: int
    grasp_frame: int  # Frame when gripper closes
    open_frame: int  # Frame when gripper opened before this grasp
    approach_duration: int  # Frames between open and close
    pre_grasp_velocity: float  # Velocity just before grasp
    stationary_frames: int  # Frames arm was stationary before grasp
    grasp_position: np.ndarray  # Joint positions at grasp
    success: bool  # Did gripper actually grab something (heuristic)


class DatasetAnalyzer:
    """Analyzes a single dataset for gripper timing, velocity, and trajectory patterns."""

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    GRIPPER_OPEN_THRESHOLD = 20.0  # degrees - gripper is "open" above this
    GRIPPER_CLOSED_THRESHOLD = 10.0  # degrees - gripper is "closed" below this
    STATIONARY_THRESHOLD = 0.5  # degrees/frame - arm is "stationary" below this

    def __init__(self, dataset_path: str, name: str = None, max_episodes: int = None):
        self.dataset_path = Path(dataset_path)
        self.name = name or self.dataset_path.name
        self.max_episodes = max_episodes

        # Load data
        self.episodes = self._load_episodes()
        self.num_episodes = len(self.episodes)

        # Analysis results
        self.gripper_events: List[GripperEvent] = []
        self.grasp_attempts: List[GraspAttempt] = []
        self.velocity_stats: Dict = {}
        self.direction_stats: Dict = {}

    def _load_episodes(self) -> List[Dict]:
        """Load all episodes from parquet files."""
        data_dir = self.dataset_path / "data" / "chunk-000"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        if self.max_episodes:
            parquet_files = parquet_files[:self.max_episodes]

        episodes = []
        for ep_path in parquet_files:
            ep_id = int(ep_path.stem.split("_")[1])
            df = pd.read_parquet(ep_path)

            # Handle both action formats
            if 'action' in df.columns:
                actions = np.vstack(df['action'].values)
            else:
                raise ValueError(f"No action column found in {ep_path}")

            if 'observation.state' in df.columns:
                states = np.vstack(df['observation.state'].values)
            else:
                states = actions.copy()  # Fallback

            episodes.append({
                'id': ep_id,
                'actions': actions,
                'states': states,
                'length': len(actions)
            })

        return episodes

    def analyze_gripper_timing(self) -> Dict:
        """Analyze when gripper opens/closes relative to arm movement."""
        all_events = []
        all_attempts = []

        for ep in self.episodes:
            ep_id = ep['id']
            actions = ep['actions']
            gripper_values = actions[:, 5]  # Gripper is index 5
            arm_values = actions[:, :5]  # Arm joints are 0-4

            # Calculate arm velocity
            arm_deltas = np.diff(arm_values, axis=0)
            arm_velocities = np.sqrt(np.sum(arm_deltas**2, axis=1))
            arm_velocities = np.concatenate([[0], arm_velocities])  # Prepend 0 for first frame

            # Detect gripper state transitions
            prev_state = 'closed' if gripper_values[0] < self.GRIPPER_CLOSED_THRESHOLD else 'open'
            last_open_frame = 0 if prev_state == 'open' else -1

            for i in range(1, len(gripper_values)):
                curr_val = gripper_values[i]

                if prev_state == 'closed' and curr_val > self.GRIPPER_OPEN_THRESHOLD:
                    # Gripper opened
                    event = GripperEvent(
                        frame=i,
                        event_type='open',
                        gripper_value=curr_val,
                        arm_position=arm_values[i],
                        arm_velocity=arm_velocities[i]
                    )
                    all_events.append(event)
                    last_open_frame = i
                    prev_state = 'open'

                elif prev_state == 'open' and curr_val < self.GRIPPER_CLOSED_THRESHOLD:
                    # Gripper closed (grasp attempt)
                    event = GripperEvent(
                        frame=i,
                        event_type='close',
                        gripper_value=curr_val,
                        arm_position=arm_values[i],
                        arm_velocity=arm_velocities[i]
                    )
                    all_events.append(event)

                    # Record grasp attempt
                    approach_duration = i - last_open_frame if last_open_frame >= 0 else 0

                    # Count stationary frames before grasp
                    stationary_count = 0
                    for j in range(i-1, max(0, i-60), -1):  # Look back up to 2 seconds
                        if arm_velocities[j] < self.STATIONARY_THRESHOLD:
                            stationary_count += 1
                        else:
                            break

                    # Pre-grasp velocity (average of last 10 frames before grasp)
                    start_idx = max(0, i-10)
                    pre_grasp_vel = np.mean(arm_velocities[start_idx:i])

                    attempt = GraspAttempt(
                        episode_id=ep_id,
                        grasp_frame=i,
                        open_frame=last_open_frame,
                        approach_duration=approach_duration,
                        pre_grasp_velocity=pre_grasp_vel,
                        stationary_frames=stationary_count,
                        grasp_position=arm_values[i].copy(),
                        success=True  # Assume success (no way to verify without visual check)
                    )
                    all_attempts.append(attempt)
                    prev_state = 'closed'

        self.gripper_events = all_events
        self.grasp_attempts = all_attempts

        # Compute statistics
        if all_attempts:
            approach_durations = [a.approach_duration for a in all_attempts]
            pre_grasp_velocities = [a.pre_grasp_velocity for a in all_attempts]
            stationary_frames = [a.stationary_frames for a in all_attempts]

            return {
                'num_grasp_attempts': len(all_attempts),
                'approach_duration': {
                    'mean': np.mean(approach_durations),
                    'std': np.std(approach_durations),
                    'min': np.min(approach_durations),
                    'max': np.max(approach_durations),
                    'mean_seconds': np.mean(approach_durations) / 30.0
                },
                'pre_grasp_velocity': {
                    'mean': np.mean(pre_grasp_velocities),
                    'std': np.std(pre_grasp_velocities),
                    'max': np.max(pre_grasp_velocities)
                },
                'stationary_frames_before_grasp': {
                    'mean': np.mean(stationary_frames),
                    'std': np.std(stationary_frames),
                    'max': np.max(stationary_frames),
                    'pct_with_pause': 100 * sum(1 for s in stationary_frames if s > 5) / len(stationary_frames)
                }
            }
        return {'num_grasp_attempts': 0}

    def analyze_velocity_profiles(self) -> Dict:
        """Analyze velocity patterns across all episodes."""
        all_velocities = []
        all_velocities_by_joint = {name: [] for name in self.ARM_JOINT_NAMES}
        episode_velocities = []

        for ep in self.episodes:
            actions = ep['actions']
            arm_values = actions[:, :5]

            # Calculate per-frame velocity
            deltas = np.diff(arm_values, axis=0)
            total_vel = np.sqrt(np.sum(deltas**2, axis=1))
            all_velocities.extend(total_vel)
            episode_velocities.append({
                'mean': np.mean(total_vel),
                'max': np.max(total_vel),
                'std': np.std(total_vel)
            })

            # Per-joint velocities
            for i, name in enumerate(self.ARM_JOINT_NAMES):
                all_velocities_by_joint[name].extend(np.abs(deltas[:, i]))

        all_velocities = np.array(all_velocities)

        self.velocity_stats = {
            'overall': {
                'mean': np.mean(all_velocities),
                'std': np.std(all_velocities),
                'max': np.max(all_velocities),
                'p50': np.percentile(all_velocities, 50),
                'p95': np.percentile(all_velocities, 95),
                'p99': np.percentile(all_velocities, 99),
                'mean_deg_per_sec': np.mean(all_velocities) * 30,
                'max_deg_per_sec': np.max(all_velocities) * 30
            },
            'per_joint': {},
            'per_episode': episode_velocities
        }

        for name in self.ARM_JOINT_NAMES:
            vels = np.array(all_velocities_by_joint[name])
            self.velocity_stats['per_joint'][name] = {
                'mean': np.mean(vels),
                'std': np.std(vels),
                'max': np.max(vels),
                'mean_deg_per_sec': np.mean(vels) * 30,
                'max_deg_per_sec': np.max(vels) * 30
            }

        return self.velocity_stats

    def analyze_directional_bias(self, steps: int = 100) -> Dict:
        """Analyze directional bias in first N steps."""
        left_count = 0
        right_count = 0
        center_count = 0
        deltas = []

        for ep in self.episodes:
            actions = ep['actions']
            step0_shoulder_pan = actions[0, 0]
            stepN_idx = min(steps, len(actions) - 1)
            stepN_shoulder_pan = actions[stepN_idx, 0]

            delta = stepN_shoulder_pan - step0_shoulder_pan
            deltas.append(delta)

            if delta < -5:
                left_count += 1
            elif delta > 5:
                right_count += 1
            else:
                center_count += 1

        total = len(self.episodes)
        self.direction_stats = {
            'left_count': left_count,
            'right_count': right_count,
            'center_count': center_count,
            'left_pct': 100 * left_count / total if total > 0 else 0,
            'right_pct': 100 * right_count / total if total > 0 else 0,
            'center_pct': 100 * center_count / total if total > 0 else 0,
            'mean_delta': np.mean(deltas),
            'std_delta': np.std(deltas),
            'deltas': deltas
        }

        return self.direction_stats

    def analyze_grasp_positions(self) -> Dict:
        """Analyze where grasps occur in joint space."""
        if not self.grasp_attempts:
            self.analyze_gripper_timing()

        if not self.grasp_attempts:
            return {'num_grasps': 0}

        positions = np.array([a.grasp_position for a in self.grasp_attempts])

        result = {
            'num_grasps': len(positions),
            'per_joint': {}
        }

        for i, name in enumerate(self.ARM_JOINT_NAMES):
            joint_positions = positions[:, i]
            result['per_joint'][name] = {
                'mean': float(np.mean(joint_positions)),
                'std': float(np.std(joint_positions)),
                'min': float(np.min(joint_positions)),
                'max': float(np.max(joint_positions)),
                'range': float(np.max(joint_positions) - np.min(joint_positions))
            }

        return result

    def get_all_actions(self) -> np.ndarray:
        """Get all actions concatenated."""
        return np.vstack([ep['actions'] for ep in self.episodes])

    def get_episode_lengths(self) -> np.ndarray:
        """Get array of episode lengths."""
        return np.array([ep['length'] for ep in self.episodes])

    def get_all_velocities(self) -> np.ndarray:
        """Get all arm velocities."""
        all_vels = []
        for ep in self.episodes:
            arm_values = ep['actions'][:, :5]
            deltas = np.diff(arm_values, axis=0)
            total_vel = np.sqrt(np.sum(deltas**2, axis=1))
            all_vels.extend(total_vel)
        return np.array(all_vels)


def compare_datasets(user_data: DatasetAnalyzer, baseline_data: DatasetAnalyzer) -> Dict:
    """Compare two datasets and return comparison metrics."""

    # Velocity comparison
    user_vels = user_data.get_all_velocities()
    baseline_vels = baseline_data.get_all_velocities()

    # Gripper timing comparison
    user_timing = user_data.analyze_gripper_timing()
    baseline_timing = baseline_data.analyze_gripper_timing()

    # Directional bias comparison
    user_bias = user_data.analyze_directional_bias()
    baseline_bias = baseline_data.analyze_directional_bias()

    comparison = {
        'velocity': {
            'user_mean': float(np.mean(user_vels)),
            'baseline_mean': float(np.mean(baseline_vels)),
            'ratio': float(np.mean(user_vels) / np.mean(baseline_vels)) if np.mean(baseline_vels) > 0 else 0,
            'user_max': float(np.max(user_vels)),
            'baseline_max': float(np.max(baseline_vels)),
        },
        'gripper_timing': {
            'user_approach_duration': user_timing.get('approach_duration', {}).get('mean', 0),
            'baseline_approach_duration': baseline_timing.get('approach_duration', {}).get('mean', 0),
            'user_pre_grasp_velocity': user_timing.get('pre_grasp_velocity', {}).get('mean', 0),
            'baseline_pre_grasp_velocity': baseline_timing.get('pre_grasp_velocity', {}).get('mean', 0),
            'user_stationary_pct': user_timing.get('stationary_frames_before_grasp', {}).get('pct_with_pause', 0),
            'baseline_stationary_pct': baseline_timing.get('stationary_frames_before_grasp', {}).get('pct_with_pause', 0),
        },
        'direction_bias': {
            'user_left_pct': user_bias.get('left_pct', 0),
            'baseline_left_pct': baseline_bias.get('left_pct', 0),
            'user_right_pct': user_bias.get('right_pct', 0),
            'baseline_right_pct': baseline_bias.get('right_pct', 0),
        }
    }

    return comparison


def create_visualizations(user_data: DatasetAnalyzer, baseline_data: DatasetAnalyzer,
                          output_dir: Path) -> List[str]:
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # 1. Velocity Comparison Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    user_vels = user_data.get_all_velocities()
    baseline_vels = baseline_data.get_all_velocities()

    # Histogram
    ax = axes[0]
    ax.hist(user_vels, bins=50, alpha=0.6, label=f'{user_data.name}', density=True)
    ax.hist(baseline_vels, bins=50, alpha=0.6, label=f'{baseline_data.name}', density=True)
    ax.axvline(np.mean(user_vels), color='blue', linestyle='--', label=f'User mean: {np.mean(user_vels):.2f}°/frame')
    ax.axvline(np.mean(baseline_vels), color='orange', linestyle='--', label=f'Baseline mean: {np.mean(baseline_vels):.2f}°/frame')
    ax.set_xlabel('Velocity (degrees/frame)')
    ax.set_ylabel('Density')
    ax.set_title('Velocity Distribution Comparison')
    ax.legend()

    # Box plot
    ax = axes[1]
    ax.boxplot([user_vels, baseline_vels], labels=[user_data.name, baseline_data.name])
    ax.set_ylabel('Velocity (degrees/frame)')
    ax.set_title('Velocity Distribution')

    plt.tight_layout()
    out_path = output_dir / 'velocity_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    created_files.append(str(out_path))

    # 2. Gripper Timing Analysis
    user_timing = user_data.analyze_gripper_timing()
    baseline_timing = baseline_data.analyze_gripper_timing()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Approach duration comparison
    ax = axes[0, 0]
    if user_data.grasp_attempts and baseline_data.grasp_attempts:
        user_durations = [a.approach_duration / 30.0 for a in user_data.grasp_attempts]  # Convert to seconds
        baseline_durations = [a.approach_duration / 30.0 for a in baseline_data.grasp_attempts]
        ax.hist(user_durations, bins=20, alpha=0.6, label=f'{user_data.name}')
        ax.hist(baseline_durations, bins=20, alpha=0.6, label=f'{baseline_data.name}')
        ax.axvline(np.mean(user_durations), color='blue', linestyle='--')
        ax.axvline(np.mean(baseline_durations), color='orange', linestyle='--')
    ax.set_xlabel('Approach Duration (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Time Between Gripper Open and Close')
    ax.legend()

    # Pre-grasp velocity comparison
    ax = axes[0, 1]
    if user_data.grasp_attempts and baseline_data.grasp_attempts:
        user_pre_vels = [a.pre_grasp_velocity for a in user_data.grasp_attempts]
        baseline_pre_vels = [a.pre_grasp_velocity for a in baseline_data.grasp_attempts]
        ax.hist(user_pre_vels, bins=20, alpha=0.6, label=f'{user_data.name}')
        ax.hist(baseline_pre_vels, bins=20, alpha=0.6, label=f'{baseline_data.name}')
    ax.set_xlabel('Pre-Grasp Velocity (degrees/frame)')
    ax.set_ylabel('Count')
    ax.set_title('Arm Velocity Just Before Grasp')
    ax.legend()

    # Stationary frames before grasp
    ax = axes[1, 0]
    if user_data.grasp_attempts and baseline_data.grasp_attempts:
        user_stationary = [a.stationary_frames for a in user_data.grasp_attempts]
        baseline_stationary = [a.stationary_frames for a in baseline_data.grasp_attempts]
        ax.hist(user_stationary, bins=20, alpha=0.6, label=f'{user_data.name}')
        ax.hist(baseline_stationary, bins=20, alpha=0.6, label=f'{baseline_data.name}')
    ax.set_xlabel('Stationary Frames Before Grasp')
    ax.set_ylabel('Count')
    ax.set_title('Pause Duration Before Closing Gripper')
    ax.legend()

    # Summary bar chart
    ax = axes[1, 1]
    metrics = ['Approach\n(sec)', 'Pre-Grasp\nVelocity', 'Pause\n(frames)']
    user_vals = [
        user_timing.get('approach_duration', {}).get('mean_seconds', 0),
        user_timing.get('pre_grasp_velocity', {}).get('mean', 0),
        user_timing.get('stationary_frames_before_grasp', {}).get('mean', 0)
    ]
    baseline_vals = [
        baseline_timing.get('approach_duration', {}).get('mean_seconds', 0),
        baseline_timing.get('pre_grasp_velocity', {}).get('mean', 0),
        baseline_timing.get('stationary_frames_before_grasp', {}).get('mean', 0)
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, user_vals, width, label=user_data.name)
    ax.bar(x + width/2, baseline_vals, width, label=baseline_data.name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Gripper Timing Summary')
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / 'gripper_timing_analysis.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    created_files.append(str(out_path))

    # 3. Grasp Position Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data in zip(axes, [user_data, baseline_data]):
        if data.grasp_attempts:
            positions = np.array([a.grasp_position for a in data.grasp_attempts])
            # Plot shoulder_pan vs shoulder_lift
            scatter = ax.scatter(positions[:, 0], positions[:, 1],
                                 c=range(len(positions)), cmap='viridis', alpha=0.7)
            ax.set_xlabel('shoulder_pan (degrees)')
            ax.set_ylabel('shoulder_lift (degrees)')
            ax.set_title(f'{data.name}\nGrasp Positions (n={len(positions)})')
            plt.colorbar(scatter, ax=ax, label='Grasp Index')

    plt.tight_layout()
    out_path = output_dir / 'grasp_position_heatmap.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    created_files.append(str(out_path))

    # 4. Directional Bias Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data in zip(axes, [user_data, baseline_data]):
        bias = data.direction_stats
        labels = ['LEFT', 'CENTER', 'RIGHT']
        counts = [bias['left_count'], bias['center_count'], bias['right_count']]
        colors = ['blue', 'gray', 'red']
        bars = ax.bar(labels, counts, color=colors, alpha=0.7)

        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            pct = 100 * count / total if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.0f}%', ha='center', va='bottom')

        ax.set_title(f'{data.name}\nDirectional Bias in First 100 Steps')
        ax.set_ylabel('Number of Episodes')

    plt.tight_layout()
    out_path = output_dir / 'direction_bias_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    created_files.append(str(out_path))

    # 5. Episode Trajectory Comparison (first 3 episodes from each)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, data in enumerate([user_data, baseline_data]):
        for col in range(min(3, len(data.episodes))):
            ax = axes[row, col]
            ep = data.episodes[col]
            actions = ep['actions']

            # Plot each arm joint
            for i, name in enumerate(data.ARM_JOINT_NAMES):
                ax.plot(actions[:, i], label=name, alpha=0.8)

            ax.set_xlabel('Frame')
            ax.set_ylabel('Position (degrees)')
            ax.set_title(f'{data.name} - Episode {ep["id"]}')
            if col == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    created_files.append(str(out_path))

    return created_files


def validate_hypotheses(user_data: DatasetAnalyzer, baseline_data: DatasetAnalyzer,
                        comparison: Dict) -> Dict:
    """Validate user's hypotheses with data."""

    hypotheses = {}

    # Hypothesis 1: Moving too fast
    vel_ratio = comparison['velocity']['ratio']
    if vel_ratio > 1.3:
        hypotheses['moving_too_fast'] = {
            'verdict': 'LIKELY TRUE',
            'evidence': f'Your velocity is {vel_ratio:.1f}x higher than baseline ({comparison["velocity"]["user_mean"]:.2f} vs {comparison["velocity"]["baseline_mean"]:.2f} deg/frame)',
            'recommendation': 'Slow down movements, especially during approach phase'
        }
    elif vel_ratio > 1.1:
        hypotheses['moving_too_fast'] = {
            'verdict': 'POSSIBLY TRUE',
            'evidence': f'Your velocity is slightly higher ({vel_ratio:.1f}x) than baseline',
            'recommendation': 'Consider slowing down during critical phases'
        }
    else:
        hypotheses['moving_too_fast'] = {
            'verdict': 'FALSE',
            'evidence': f'Your velocity ({comparison["velocity"]["user_mean"]:.2f}) is similar to or slower than baseline ({comparison["velocity"]["baseline_mean"]:.2f})',
            'recommendation': 'Velocity is not the issue'
        }

    # Hypothesis 2: Gripper opens late
    user_approach = comparison['gripper_timing']['user_approach_duration']
    baseline_approach = comparison['gripper_timing']['baseline_approach_duration']

    if baseline_approach > 0 and user_approach / baseline_approach < 0.7:
        hypotheses['gripper_opens_late'] = {
            'verdict': 'LIKELY TRUE',
            'evidence': f'Your approach phase is {user_approach/30:.1f}s vs baseline {baseline_approach/30:.1f}s',
            'recommendation': 'Open gripper earlier - at least 1 second before reaching target'
        }
    else:
        hypotheses['gripper_opens_late'] = {
            'verdict': 'NEEDS VERIFICATION',
            'evidence': f'Approach duration: your {user_approach/30:.1f}s vs baseline {baseline_approach/30:.1f}s',
            'recommendation': 'Consider opening gripper earlier for better timing'
        }

    # Hypothesis 3: Directional bias (60% left bias)
    user_left_pct = comparison['direction_bias']['user_left_pct']
    if user_left_pct > 60:
        hypotheses['directional_bias'] = {
            'verdict': 'TRUE',
            'evidence': f'{user_left_pct:.0f}% of episodes move LEFT initially (vs {comparison["direction_bias"]["baseline_left_pct"]:.0f}% baseline)',
            'recommendation': 'Augmentation may help but consider collecting more balanced data'
        }
    elif user_left_pct > 45:
        hypotheses['directional_bias'] = {
            'verdict': 'MODERATE',
            'evidence': f'{user_left_pct:.0f}% LEFT vs {comparison["direction_bias"]["baseline_left_pct"]:.0f}% baseline',
            'recommendation': 'Bias is moderate - horizontal augmentation should help'
        }
    else:
        hypotheses['directional_bias'] = {
            'verdict': 'FALSE (after augmentation)',
            'evidence': f'{user_left_pct:.0f}% LEFT - well balanced',
            'recommendation': 'Directional bias is not the main issue'
        }

    # Hypothesis 4: Pre-grasp pause
    user_pause_pct = comparison['gripper_timing']['user_stationary_pct']
    baseline_pause_pct = comparison['gripper_timing']['baseline_stationary_pct']

    if user_pause_pct < baseline_pause_pct * 0.5:
        hypotheses['no_pause_before_grasp'] = {
            'verdict': 'LIKELY TRUE',
            'evidence': f'Only {user_pause_pct:.0f}% of your grasps have a pause vs {baseline_pause_pct:.0f}% baseline',
            'recommendation': 'Pause for 0.3-0.5 seconds before closing gripper'
        }
    else:
        hypotheses['no_pause_before_grasp'] = {
            'verdict': 'FALSE',
            'evidence': f'{user_pause_pct:.0f}% of grasps have pause (baseline: {baseline_pause_pct:.0f}%)',
            'recommendation': 'Pause behavior is adequate'
        }

    # Hypothesis 5: Pre-grasp velocity too high
    user_pre_vel = comparison['gripper_timing']['user_pre_grasp_velocity']
    baseline_pre_vel = comparison['gripper_timing']['baseline_pre_grasp_velocity']

    if baseline_pre_vel > 0 and user_pre_vel / baseline_pre_vel > 1.5:
        hypotheses['moving_during_grasp'] = {
            'verdict': 'TRUE - CRITICAL',
            'evidence': f'Arm velocity at grasp: {user_pre_vel:.2f} deg/frame vs baseline {baseline_pre_vel:.2f}',
            'recommendation': 'Stop arm movement before closing gripper - this is likely causing blocks to slide'
        }
    else:
        hypotheses['moving_during_grasp'] = {
            'verdict': 'FALSE',
            'evidence': f'Pre-grasp velocity {user_pre_vel:.2f} is similar to baseline {baseline_pre_vel:.2f}',
            'recommendation': 'Pre-grasp motion is adequate'
        }

    return hypotheses


def generate_report(user_data: DatasetAnalyzer, baseline_data: DatasetAnalyzer,
                    comparison: Dict, hypotheses: Dict, output_dir: Path) -> str:
    """Generate the final markdown report."""

    report = []
    report.append("# Data Quality Analysis Report: SO-ARM101 Pick-and-Place\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**User Dataset:** {user_data.name} ({user_data.num_episodes} episodes)\n")
    report.append(f"**Baseline Dataset:** {baseline_data.name} ({baseline_data.num_episodes} episodes)\n")
    report.append("\n---\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    critical_issues = [h for h, v in hypotheses.items() if 'TRUE' in v['verdict'] and 'CRITICAL' in v['verdict']]
    likely_issues = [h for h, v in hypotheses.items() if 'LIKELY TRUE' in v['verdict']]
    confirmed_issues = [h for h, v in hypotheses.items() if v['verdict'] == 'TRUE']

    if critical_issues or likely_issues or confirmed_issues:
        report.append("### Issues Identified:\n")
        for issue in critical_issues:
            report.append(f"- **CRITICAL**: {issue.replace('_', ' ').title()} - {hypotheses[issue]['evidence']}\n")
        for issue in likely_issues:
            report.append(f"- **Likely**: {issue.replace('_', ' ').title()} - {hypotheses[issue]['evidence']}\n")
        for issue in confirmed_issues:
            if issue not in critical_issues:
                report.append(f"- **Confirmed**: {issue.replace('_', ' ').title()} - {hypotheses[issue]['evidence']}\n")
    else:
        report.append("No major issues identified. Data quality appears adequate.\n")

    report.append("\n---\n")

    # Hypothesis Validation
    report.append("## Hypothesis Validation\n")
    report.append("| Hypothesis | Verdict | Evidence | Recommendation |\n")
    report.append("|------------|---------|----------|----------------|\n")
    for hyp, details in hypotheses.items():
        hyp_name = hyp.replace('_', ' ').title()
        report.append(f"| {hyp_name} | **{details['verdict']}** | {details['evidence']} | {details['recommendation']} |\n")

    report.append("\n---\n")

    # Velocity Comparison
    report.append("## Velocity Analysis\n")
    report.append("| Metric | Your Data | Baseline | Ratio |\n")
    report.append("|--------|-----------|----------|-------|\n")
    report.append(f"| Mean Velocity | {comparison['velocity']['user_mean']:.3f} °/frame | {comparison['velocity']['baseline_mean']:.3f} °/frame | {comparison['velocity']['ratio']:.2f}x |\n")
    report.append(f"| Max Velocity | {comparison['velocity']['user_max']:.2f} °/frame | {comparison['velocity']['baseline_max']:.2f} °/frame | - |\n")
    report.append(f"| Mean (deg/sec) | {comparison['velocity']['user_mean']*30:.1f} °/s | {comparison['velocity']['baseline_mean']*30:.1f} °/s | - |\n")

    report.append("\n**Reference:** Good velocity is typically 0.75-1.0 °/frame (~22-30 °/sec at 30 FPS)\n")

    report.append("\n---\n")

    # Gripper Timing
    report.append("## Gripper Timing Analysis\n")
    report.append("| Metric | Your Data | Baseline |\n")
    report.append("|--------|-----------|----------|\n")
    report.append(f"| Approach Duration | {comparison['gripper_timing']['user_approach_duration']/30:.2f} sec | {comparison['gripper_timing']['baseline_approach_duration']/30:.2f} sec |\n")
    report.append(f"| Pre-Grasp Velocity | {comparison['gripper_timing']['user_pre_grasp_velocity']:.3f} °/frame | {comparison['gripper_timing']['baseline_pre_grasp_velocity']:.3f} °/frame |\n")
    report.append(f"| Grasps with Pause | {comparison['gripper_timing']['user_stationary_pct']:.0f}% | {comparison['gripper_timing']['baseline_stationary_pct']:.0f}% |\n")

    report.append("\n**Key Insight:** The approach phase should be long enough for the robot to align with the target. Arm should be nearly stationary when gripper closes.\n")

    report.append("\n---\n")

    # Directional Bias
    report.append("## Directional Bias Analysis\n")
    report.append("| Direction | Your Data | Baseline |\n")
    report.append("|-----------|-----------|----------|\n")
    report.append(f"| LEFT | {comparison['direction_bias']['user_left_pct']:.0f}% | {comparison['direction_bias']['baseline_left_pct']:.0f}% |\n")
    report.append(f"| RIGHT | {comparison['direction_bias']['user_right_pct']:.0f}% | {comparison['direction_bias']['baseline_right_pct']:.0f}% |\n")

    if comparison['direction_bias']['user_left_pct'] > 60:
        report.append("\n**Warning:** Strong LEFT bias detected. This can cause the model to always swing left at inference start.\n")

    report.append("\n---\n")

    # Grasp Position Analysis
    user_grasp = user_data.analyze_grasp_positions()
    baseline_grasp = baseline_data.analyze_grasp_positions()

    report.append("## Grasp Position Distribution\n")
    report.append("### Shoulder Pan Range at Grasp:\n")
    if 'per_joint' in user_grasp and 'shoulder_pan' in user_grasp['per_joint']:
        u_sp = user_grasp['per_joint']['shoulder_pan']
        report.append(f"- Your data: [{u_sp['min']:.1f}°, {u_sp['max']:.1f}°] (range: {u_sp['range']:.1f}°)\n")
    if 'per_joint' in baseline_grasp and 'shoulder_pan' in baseline_grasp['per_joint']:
        b_sp = baseline_grasp['per_joint']['shoulder_pan']
        report.append(f"- Baseline: [{b_sp['min']:.1f}°, {b_sp['max']:.1f}°] (range: {b_sp['range']:.1f}°)\n")

    report.append("\n**Note:** Limited grasp position range means the model won't know how to grasp objects outside that range.\n")

    report.append("\n---\n")

    # Recommendations
    report.append("## Data Collection Recommendations\n")
    report.append("### Immediate Actions:\n")

    recommendations = []
    for hyp, details in hypotheses.items():
        if 'TRUE' in details['verdict']:
            recommendations.append(f"- {details['recommendation']}")

    if recommendations:
        for rec in recommendations:
            report.append(f"{rec}\n")
    else:
        report.append("- No critical issues found. Consider general quality improvements.\n")

    report.append("\n### General Best Practices:\n")
    report.append("1. **Open gripper early:** At least 1 second before reaching the target\n")
    report.append("2. **Slow down approach:** Final 30 frames before grasp should be slow and deliberate\n")
    report.append("3. **Pause before grasp:** Include 0.3-0.5 seconds of stationary time before closing gripper\n")
    report.append("4. **Top-down approach:** Come from above, not from the side\n")
    report.append("5. **Balanced coverage:** Ensure blocks are placed at various positions (left, center, right)\n")
    report.append("6. **Consistent speed:** Aim for 0.75-1.0 °/frame (~22-30 °/sec)\n")
    report.append("7. **Include failures carefully:** If you miss, show a clean recovery, don't flail\n")

    report.append("\n---\n")

    # Visualizations Reference
    report.append("## Visualizations\n")
    report.append("The following plots have been generated:\n")
    report.append("- `velocity_comparison.png` - Velocity distribution comparison\n")
    report.append("- `gripper_timing_analysis.png` - Gripper timing patterns\n")
    report.append("- `grasp_position_heatmap.png` - Where grasps occur in joint space\n")
    report.append("- `direction_bias_comparison.png` - Directional bias in first 100 steps\n")
    report.append("- `trajectory_comparison.png` - Sample episode trajectories\n")

    report.append("\n---\n")

    # Next Steps
    report.append("## Next Steps\n")
    report.append("1. Review the visualizations to understand the patterns\n")
    report.append("2. Watch sample videos from baseline dataset to see good demonstration style\n")
    report.append("3. Collect new data following the recommendations above\n")
    report.append("4. Re-run this analysis to verify improvements\n")
    report.append("5. Train with the improved dataset\n")

    report_text = ''.join(report)

    # Save report
    report_path = output_dir / 'data_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset analysis for Jassy")
    parser.add_argument("--dataset", "-d", type=str,
                        default="/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot_augmented",
                        help="Path to user's dataset")
    parser.add_argument("--baseline", "-b", type=str,
                        default="/home/jrobot/project/Isaac-GR00T/datasets/so100_strawberry_grape",
                        help="Path to baseline dataset")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="/home/jrobot/project/Isaac-GR00T/custom/jdocs/jassy",
                        help="Output directory for report and visualizations")
    parser.add_argument("--max-episodes", "-m", type=int, default=50,
                        help="Maximum episodes to analyze per dataset")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA QUALITY ANALYSIS FOR JASSY")
    print("=" * 70)

    # Load datasets
    print(f"\nLoading user dataset: {args.dataset}")
    user_data = DatasetAnalyzer(args.dataset, name="Your Data", max_episodes=args.max_episodes)
    print(f"  Loaded {user_data.num_episodes} episodes")

    print(f"\nLoading baseline dataset: {args.baseline}")
    baseline_data = DatasetAnalyzer(args.baseline, name="Baseline (so100_strawberry_grape)", max_episodes=args.max_episodes)
    print(f"  Loaded {baseline_data.num_episodes} episodes")

    # Run analysis
    print("\nAnalyzing datasets...")
    print("  - Gripper timing patterns...")
    user_data.analyze_gripper_timing()
    baseline_data.analyze_gripper_timing()

    print("  - Velocity profiles...")
    user_data.analyze_velocity_profiles()
    baseline_data.analyze_velocity_profiles()

    print("  - Directional bias...")
    user_data.analyze_directional_bias()
    baseline_data.analyze_directional_bias()

    print("  - Grasp positions...")
    user_data.analyze_grasp_positions()
    baseline_data.analyze_grasp_positions()

    # Compare datasets
    print("\nComparing datasets...")
    comparison = compare_datasets(user_data, baseline_data)

    # Validate hypotheses
    print("Validating hypotheses...")
    hypotheses = validate_hypotheses(user_data, baseline_data, comparison)

    # Create visualizations
    print("\nCreating visualizations...")
    created_files = create_visualizations(user_data, baseline_data, output_dir)
    for f in created_files:
        print(f"  Created: {f}")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(user_data, baseline_data, comparison, hypotheses, output_dir)
    print(f"  Report saved to: {output_dir / 'data_analysis_report.md'}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nHypothesis Verdicts:")
    for hyp, details in hypotheses.items():
        status_emoji = "✓" if "FALSE" in details['verdict'] else ("⚠️" if "LIKELY" in details['verdict'] else "❌")
        print(f"  {status_emoji} {hyp.replace('_', ' ').title()}: {details['verdict']}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
