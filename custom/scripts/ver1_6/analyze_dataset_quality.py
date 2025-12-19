"""Comprehensive dataset quality analysis script.

Analyzes:
1. Action discontinuities (sudden jumps)
2. State-action alignment
3. Joint range distribution
4. Action velocity/acceleration patterns
5. Episode length consistency
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_dataset(dataset_path: str, max_episodes: int = 50):
    """Analyze dataset for quality issues."""
    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data" / "chunk-000"
    episodes = sorted(data_dir.glob("episode_*.parquet"))[:max_episodes]

    print(f"Dataset: {dataset_path.name}")
    print(f"Analyzing {len(episodes)} episodes")
    print("=" * 70)

    # Collect data
    all_actions = []
    all_states = []
    episode_lengths = []
    action_deltas = []  # Frame-to-frame changes
    large_jumps = []  # Episodes with large jumps

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    for ep_path in episodes:
        ep_id = int(ep_path.stem.split("_")[1])
        df = pd.read_parquet(ep_path)

        actions = np.vstack(df['action'].values)
        states = np.vstack(df['observation.state'].values)

        all_actions.append(actions)
        all_states.append(states)
        episode_lengths.append(len(actions))

        # Calculate frame-to-frame deltas
        deltas = np.diff(actions, axis=0)
        action_deltas.append(deltas)

        # Detect large jumps (>10 degrees between frames)
        max_delta = np.abs(deltas).max(axis=0)
        if max_delta.max() > 10:
            large_jumps.append({
                "ep_id": ep_id,
                "max_delta": max_delta,
                "max_joint": joint_names[max_delta.argmax()],
                "max_value": max_delta.max()
            })

    all_actions_flat = np.vstack(all_actions)
    all_states_flat = np.vstack(all_states)
    all_deltas_flat = np.vstack(action_deltas)

    # 1. Episode Length Analysis
    print("\n1. EPISODE LENGTH ANALYSIS:")
    print("-" * 70)
    ep_lens = np.array(episode_lengths)
    print(f"   Min: {ep_lens.min()}, Max: {ep_lens.max()}, Mean: {ep_lens.mean():.1f}, Std: {ep_lens.std():.1f}")
    short_eps = sum(1 for l in ep_lens if l < 100)
    long_eps = sum(1 for l in ep_lens if l > 2000)
    print(f"   Short episodes (<100 frames): {short_eps}")
    print(f"   Long episodes (>2000 frames): {long_eps}")

    # 2. Action Discontinuity Analysis
    print("\n2. ACTION DISCONTINUITY ANALYSIS (frame-to-frame):")
    print("-" * 70)
    for i, name in enumerate(joint_names):
        deltas_i = all_deltas_flat[:, i]
        max_d = np.abs(deltas_i).max()
        mean_d = np.abs(deltas_i).mean()
        std_d = deltas_i.std()
        p99 = np.percentile(np.abs(deltas_i), 99)
        print(f"   {name:15s}: max={max_d:6.2f}°, mean={mean_d:5.2f}°, std={std_d:5.2f}°, p99={p99:5.2f}°")

    if large_jumps:
        print(f"\n   WARNING: {len(large_jumps)} episodes have jumps > 10°:")
        for jump in large_jumps[:10]:  # Show first 10
            print(f"      Ep {jump['ep_id']:3d}: {jump['max_joint']} jumped {jump['max_value']:.1f}°")
    else:
        print(f"\n   OK: No episodes with jumps > 10°")

    # 3. State-Action Alignment Analysis
    print("\n3. STATE-ACTION ALIGNMENT:")
    print("-" * 70)
    # Check if state and action generally match (delayed by 1 frame typically)
    for i, name in enumerate(joint_names[:5]):  # Skip gripper
        state_mean = all_states_flat[:, i].mean()
        action_mean = all_actions_flat[:, i].mean()
        diff = abs(state_mean - action_mean)
        status = "OK" if diff < 5 else "MISMATCH"
        print(f"   {name:15s}: state_mean={state_mean:7.1f}°, action_mean={action_mean:7.1f}°, diff={diff:5.1f}° [{status}]")

    # 4. Joint Range Distribution
    print("\n4. JOINT RANGE DISTRIBUTION:")
    print("-" * 70)
    for i, name in enumerate(joint_names):
        values = all_actions_flat[:, i]
        min_v, max_v = values.min(), values.max()
        range_v = max_v - min_v
        mean_v = values.mean()
        std_v = values.std()
        # Check for limited range (robot only using small portion of workspace)
        if range_v < 30 and name != "gripper":
            status = "LIMITED RANGE"
        elif std_v < 5:
            status = "LOW VARIATION"
        else:
            status = "OK"
        print(f"   {name:15s}: [{min_v:7.1f}°, {max_v:7.1f}°] range={range_v:6.1f}° std={std_v:5.1f}° [{status}]")

    # 5. Starting Position Consistency
    print("\n5. STARTING POSITION CONSISTENCY:")
    print("-" * 70)
    start_positions = np.vstack([a[0] for a in all_actions])
    for i, name in enumerate(joint_names):
        starts = start_positions[:, i]
        min_s, max_s = starts.min(), starts.max()
        range_s = max_s - min_s
        mean_s = starts.mean()
        std_s = starts.std()
        status = "OK" if std_s < 10 else "HIGH VARIATION"
        print(f"   {name:15s}: range=[{min_s:7.1f}°, {max_s:7.1f}°] mean={mean_s:6.1f}° std={std_s:5.1f}° [{status}]")

    # 6. Velocity Profile Analysis
    print("\n6. VELOCITY PROFILE (degrees per frame):")
    print("-" * 70)
    for i, name in enumerate(joint_names):
        velocities = all_deltas_flat[:, i]
        mean_vel = np.abs(velocities).mean()
        max_vel = np.abs(velocities).max()
        # Assuming 30 FPS, convert to deg/sec
        mean_vel_sec = mean_vel * 30
        max_vel_sec = max_vel * 30
        print(f"   {name:15s}: mean={mean_vel_sec:6.1f}°/s, max={max_vel_sec:7.1f}°/s")

    return {
        "episode_lengths": ep_lens,
        "large_jumps": large_jumps,
        "actions": all_actions_flat,
        "states": all_states_flat,
        "deltas": all_deltas_flat
    }

if __name__ == "__main__":
    print("=" * 70)
    print("OUR DATASET (so101_pick_place_groot)")
    print("=" * 70)
    our_data = analyze_dataset(
        "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot",
        max_episodes=45
    )

    print("\n" + "=" * 70)
    print("SO100 EXAMPLE DATASET (finish_sandwich)")
    print("=" * 70)
    so100_data = analyze_dataset(
        "/home/jrobot/project/Isaac-GR00T/examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich",
        max_episodes=50
    )

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Plot action distributions for each joint
    for i, name in enumerate(joint_names):
        ax = axes[i // 3, i % 3]
        ax.hist(our_data["actions"][:, i], bins=50, alpha=0.5, label="Our Dataset", density=True)
        ax.hist(so100_data["actions"][:, i], bins=50, alpha=0.5, label="SO100 Example", density=True)
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("Degrees")
        ax.legend()

    plt.suptitle("Action Distribution Comparison: Our Dataset vs SO100 Example", fontsize=14)
    plt.tight_layout()
    plt.savefig("/home/jrobot/project/Isaac-GR00T/custom/scripts/ver1_6/dataset_quality_comparison.png", dpi=150)
    print(f"\nComparison plot saved to: custom/scripts/ver1_6/dataset_quality_comparison.png")
