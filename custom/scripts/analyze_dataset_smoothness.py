#!/usr/bin/env python3
"""
Analyze Training Data Smoothness
Checks if the training data itself contains high-frequency jitter/noise.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def analyze_smoothness(dataset_path):
    data_dir = Path(dataset_path) / "data"
    episodes = sorted(list(data_dir.rglob("episode_*.parquet")))
    
    print(f"Found {len(episodes)} episodes.")
    
    all_jerks = []
    all_vels = []
    
    # We focus on arm joints (first 6)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print(f"\nAnalyzing trajectory smoothness (Velocity & Jerk)...")
    
    for ep_path in episodes:
        df = pd.read_parquet(ep_path)
        
        # Get actions (or states)
        # We check ACTIONS because that's what the model learns to predict
        # Parquet column is just "action" which is already (N, 6) in this dataset format?
        # Or it might be flattened?
        # Based on previous info, "action" column contains the 6D action directly?
        try:
            raw_actions = np.stack(df["action"].values)
            if raw_actions.shape[1] == 6:
                actions = raw_actions[:, :5]
                gripper = raw_actions[:, 5:]
            else:
                # If shape is different, try to split
                print(f"Unexpected action shape: {raw_actions.shape}")
                continue
        except KeyError:
            # Fallback to split columns if they exist
            actions = np.stack(df["action.single_arm"].values)
            gripper = np.stack(df["action.gripper"].values)
            if gripper.ndim == 1: gripper = gripper[:, np.newaxis]
        
        traj = np.concatenate([actions, gripper], axis=1) # (T, 6)
        
        # Calculate finite differences (Velocity)
        vel = np.diff(traj, axis=0) # (T-1, 6)
        
        # Calculate finite differences of velocity (Acceleration)
        acc = np.diff(vel, axis=0) # (T-2, 6)
        
        # Calculate Jerk (change in acceleration) -> Metric for "shakiness"
        jerk = np.diff(acc, axis=0) # (T-3, 6)
        
        all_vels.append(np.mean(np.abs(vel), axis=0))
        all_jerks.append(np.mean(np.abs(jerk), axis=0))

    # Aggregated stats
    avg_vel = np.mean(all_vels, axis=0)
    avg_jerk = np.mean(all_jerks, axis=0)
    
    print("\n=== Dataset Smoothness Report ===")
    print(f"{'Joint':<15} {'Avg Step Change (deg)':<25} {'Avg Jerk (deg/s^3)':<25}")
    print("-" * 70)
    
    for i, name in enumerate(joint_names):
        print(f"{name:<15} {avg_vel[i]:<25.4f} {avg_jerk[i]:<25.4f}")
        
    print("\nInterpretation:")
    print("- High 'Step Change' means the robot moves fast.")
    print("- High 'Jerk' means the motion is shaky/trembling.")
    print("- A Jerk > 0.5 deg/step^3 usually indicates significant teleop noise.")

    # Detect oscillating episodes
    # If direction changes frequently (sign of velocity flips)
    print("\n=== Oscillation Detection ===")
    oscillation_counts = []
    for ep_path in episodes:
        df = pd.read_parquet(ep_path)
        
        try:
            raw_actions = np.stack(df["action"].values)
            if raw_actions.shape[1] == 6:
                actions = raw_actions[:, :5]
            else:
                continue
        except KeyError:
            actions = np.stack(df["action.single_arm"].values)
            
        vel = np.diff(actions, axis=0)
        
        # Count sign flips per joint
        # sign(v[t]) != sign(v[t+1])
        # Use a small threshold to ignore zero-velocity jitter
        vel_sign = np.sign(vel)
        # Only count flips where velocity magnitude is non-trivial
        # But for now, raw sign flip is a good proxy for "fighting"
        sign_flips = np.sum(np.diff(vel_sign, axis=0) != 0, axis=0)
        oscillation_counts.append(sign_flips / len(vel)) # Flips per step

    avg_flips = np.mean(oscillation_counts, axis=0)
    print(f"{'Joint':<15} {'Direction Flips/Step':<25}")
    print("-" * 70)
    for i, name in enumerate(joint_names[:-1]): # Skip gripper
        print(f"{name:<15} {avg_flips[i]:<25.4f}")
        
    if np.any(avg_flips > 0.3):
        print("\n[CRITICAL] High frequency direction flipping detected in training data!")
        print("The human operator likely had shaky hands or the recording frequency was aliased.")
    else:
        print("\n[PASS] Training data seems relatively smooth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
    args = parser.parse_args()
    
    analyze_smoothness(args.dataset)

