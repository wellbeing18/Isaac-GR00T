#!/usr/bin/env python3
"""
Analyze Action Consistency (Multimodality Check)
Finds nearest neighbors in state space and checks if their actions are consistent.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import argparse

def analyze_consistency(dataset_path):
    data_dir = Path(dataset_path) / "data"
    episodes = sorted(list(data_dir.rglob("episode_*.parquet")))
    
    print(f"Loading {len(episodes)} episodes...")
    
    all_states = []
    all_actions = []
    
    for ep_path in episodes:
        df = pd.read_parquet(ep_path)
        
        # Extract State
        # Assuming state is (N, 6)
        try:
            states = np.stack(df["observation.state"].values)
        except KeyError:
            print(f"Skipping {ep_path}: no observation.state")
            continue
            
        # Extract Action
        try:
            raw_actions = np.stack(df["action"].values)
            if raw_actions.shape[1] == 6:
                actions = raw_actions
            else:
                continue
        except KeyError:
            try:
                a_arm = np.stack(df["action.single_arm"].values)
                a_grip = np.stack(df["action.gripper"].values)
                if a_grip.ndim == 1: a_grip = a_grip[:, np.newaxis]
                actions = np.concatenate([a_arm, a_grip], axis=1)
            except KeyError:
                continue
                
        all_states.append(states)
        all_actions.append(actions)
        
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Total samples: {len(all_states)}")
    
    # Fit Nearest Neighbors
    print("Fitting Nearest Neighbors...")
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(all_states)
    
    # Query: The typical "Home" position that causes vibration
    # From stable inference logs
    query_state = np.array([
        [-0.26, -98.55, 96.53, 49.76, -1.32, 0.71], # Center-ish
        [-6.15, -96.88, 96.35, 51.60, -2.20, 0.35], # Left-ish
        [ 4.66, -96.88, 96.53, 58.37, -4.04, 0.42], # Right-ish
    ])
    
    distances, indices = nbrs.kneighbors(query_state)
    
    print("\n=== Action Consistency Analysis ===")
    
    for i, q in enumerate(query_state):
        print(f"\nQuery State {i}: {np.round(q, 2)}")
        
        neighbors_idx = indices[i]
        neighbor_dists = distances[i]
        neighbor_actions = all_actions[neighbors_idx]
        
        # Calculate consistency metrics
        # We look at the immediate action (next step)
        # But wait, action in dataset is usually "target for next step" or "next position"
        
        # Action Variance (how much do neighbors disagree?)
        action_std = np.std(neighbor_actions, axis=0)
        action_mean = np.mean(neighbor_actions, axis=0)
        
        # Norm of std dev (overall "fuzziness")
        total_std = np.linalg.norm(action_std)
        
        print(f"  Nearest Neighbor Distances: {np.min(neighbor_dists):.4f} - {np.max(neighbor_dists):.4f}")
        print(f"  Action Std Dev (deg): {np.round(action_std, 2)}")
        print(f"  Total Inconsistency Score: {total_std:.4f}")
        
        if total_std > 2.0:
            print("  [FAIL] High Inconsistency! Neighbors have very different actions.")
            print("  Example Actions (Top 5):")
            for k in range(5):
                print(f"    {np.round(neighbor_actions[k], 2)} (Dist: {neighbor_dists[k]:.2f})")
        else:
            print("  [PASS] Consistent Actions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
    args = parser.parse_args()
    
    analyze_consistency(args.dataset)


