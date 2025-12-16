#!/usr/bin/env python3
"""
Normalization & Proprioception Checker
Verifies if the robot's current state is within the training distribution.
"""
import json
import numpy as np
import sys
import os
from pathlib import Path
import yaml
import time
import argparse

# Add parent path to allow imports if needed, though we'll keep this standalone-ish
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LeRobot for robot connection
try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.utils.errors import DeviceAlreadyConnectedError
except ImportError:
    print("Error: Could not import LeRobot. Make sure you are in the 'groot' environment.")
    sys.exit(1)

def load_stats(dataset_path):
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"Error: Stats file not found at {stats_path}")
        return None
    
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

def get_robot_state(hw_config_path):
    # Load config
    with open(hw_config_path, "r") as f:
        hw = yaml.safe_load(f)
    
    robot_cfg = hw.get("robot_arms", {}).get("left", {})
    port = robot_cfg.get("port", "/dev/ttyACM0")
    
    # Init robot (no cameras needed for this check)
    # Using dummy camera config to satisfy SO101Follower requirements
    cameras_cfg = {
        "head": OpenCVCameraConfig(index_or_path=99, fps=30, width=640, height=480), # Dummy
    }
    
    print(f"Connecting to robot on {port}...")
    try:
        # Note: We might need to mock cameras if they fail to connect
        # But SO101Follower requires valid camera configs usually. 
        # Let's try to minimal init.
        config = SO101FollowerConfig(port=port, id="xlerobot_left_arm", cameras={})
        robot = SO101Follower(config)
        
        if not robot.is_connected:
            robot.connect()
        
        obs = robot.get_observation()
        # Extract state
        state = np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ])
        
        robot.disconnect()
        return state
        
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        return None

def check_normalization(state, stats):
    print("\n=== Normalization Check ===")
    print(f"{'Joint':<15} {'Current':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'Z-Score':<10} {'Status'}")
    print("-" * 95)
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Stats structure: stats["observation.state"]["min"][joint_idx]
    s_state = stats["observation.state"]
    
    warnings = []
    
    for i, name in enumerate(joint_names):
        val = state[i]
        d_min = s_state["min"][i]
        d_max = s_state["max"][i]
        d_mean = s_state["mean"][i]
        d_std = s_state["std"][i]
        
        z_score = (val - d_mean) / (d_std + 1e-6)
        
        status = "OK"
        if val < d_min:
            status = "LOW (OOD)"
            warnings.append(f"{name} is below training min ({val:.1f} < {d_min:.1f})")
        elif val > d_max:
            status = "HIGH (OOD)"
            warnings.append(f"{name} is above training max ({val:.1f} > {d_max:.1f})")
        elif abs(z_score) > 3.0:
            status = "RARE"
        
        print(f"{name:<15} {val:<10.2f} {d_min:<10.2f} {d_max:<10.2f} {d_mean:<10.2f} {d_std:<10.2f} {z_score:<10.2f} {status}")

    print("\n=== Conclusion ===")
    if warnings:
        print("WARNING: Robot is Out-Of-Distribution (OOD) for the following joints:")
        for w in warnings:
            print(f"  - {w}")
        print("-> The model has NEVER seen this pose. Behavior is undefined (likely oscillation).")
    else:
        print("PASS: Robot state is within valid training distribution.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
    parser.add_argument("--config", default="custom/cfgs/so101_hardware.yaml")
    args = parser.parse_args()
    
    print(f"Loading stats from: {args.dataset}")
    stats = load_stats(args.dataset)
    if not stats: return
    
    print(f"Reading robot state...")
    state = get_robot_state(args.config)
    
    # If robot connection fails, allow manual input for debugging
    if state is None:
        print("\nCould not connect to robot. Enter state manually? (y/n)")
        # For automation, we'll skip this interactive part if running via tool
        return

    print(f"\nCaptured Robot State: {state}")
    check_normalization(state, stats)

if __name__ == "__main__":
    main()


