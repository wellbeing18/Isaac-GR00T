#!/usr/bin/env python3
"""
Deep Dive Diagnosis for GR00T Inference Issues
"""
import os
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# Import from existing scripts if possible, or replicate minimal logic
from eval_groot_openloop import load_policy_with_lora_support
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

def calculate_consistency(policy, dataset, num_trials=10, traj_id=0, step=0):
    """Run inference multiple times on the SAME input to check for stochasticity noise."""
    print(f"\n--- Consistency Test (Traj {traj_id}, Step {step}, {num_trials} trials) ---")
    
    obs = dataset.get_step_data(traj_id, step)
    
    predictions = []
    for i in range(num_trials):
        with torch.no_grad():
            action_dict = policy.get_action(obs)
            # Focus on single_arm
            pred = action_dict["action.single_arm"] # shape (16, 5) or similar
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            predictions.append(pred)
            
    predictions = np.array(predictions) # (num_trials, horizon, joints)
    
    # Calculate variance across trials per horizon step
    std_dev = np.std(predictions, axis=0) # (horizon, joints)
    mean_std = np.mean(std_dev)
    max_std = np.max(std_dev)
    
    print(f"Mean Std Dev across trials: {mean_std:.4f} degrees")
    print(f"Max Std Dev across trials: {max_std:.4f} degrees")
    
    # Check first step specifically (immediate action)
    std_first = np.std(predictions[:, 0, :], axis=0)
    print(f"Std Dev at Horizon 0 (immediate action): {np.mean(std_first):.4f} (mean), {np.max(std_first):.4f} (max)")
    
    return mean_std, predictions

def check_trajectory_smoothness(policy, dataset, traj_id=0, steps=20):
    """Run inference on sequential frames and check consistency of OVERLAPPING predictions."""
    print(f"\n--- Temporal Consistency Test (Traj {traj_id}, {steps} steps) ---")
    
    # Store predictions: list of (step_index, prediction_horizon_array)
    history = []
    
    for i in range(steps):
        obs = dataset.get_step_data(traj_id, i)
        with torch.no_grad():
            action_dict = policy.get_action(obs)
            pred = action_dict["action.single_arm"]
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            history.append(pred) # (horizon, joints)

    # Analyze overlap
    # At step T, prediction[0] corresponds to time T
    # At step T-1, prediction[1] corresponds to time T
    # We compare these overlaps.
    
    diffs = []
    for t in range(1, steps):
        # Compare prediction for time 't' made at different previous times
        
        # Prediction made at t (horizon 0)
        pred_t_at_t = history[t][0]
        
        # Prediction made at t-1 (horizon 1)
        pred_t_at_t_minus_1 = history[t-1][1]
        
        diff = np.abs(pred_t_at_t - pred_t_at_t_minus_1)
        diffs.append(np.mean(diff))
        
        print(f"Step {t}: Diff between pred(t)@t and pred(t)@t-1: {np.mean(diff):.4f} deg")

    avg_inconsistency = np.mean(diffs)
    print(f"Average Temporal Inconsistency (Jumpiness): {avg_inconsistency:.4f} degrees")
    return avg_inconsistency

def compare_to_ground_truth(policy, dataset, traj_id=0, steps=50):
    """Standard Open Loop check but printing specific values."""
    print(f"\n--- Ground Truth Comparison (Traj {traj_id}, {steps} steps) ---")
    
    maes = []
    for i in range(steps):
        obs = dataset.get_step_data(traj_id, i)
        
        # GT Action (immediate)
        gt_action = obs["action.single_arm"][0] # (horizon, joints) -> take 0
        
        with torch.no_grad():
            action_dict = policy.get_action(obs)
            pred_action = action_dict["action.single_arm"][0]
            if isinstance(pred_action, torch.Tensor):
                pred_action = pred_action.cpu().numpy()
                
        mae = np.mean(np.abs(pred_action - gt_action))
        maes.append(mae)
        
        if i % 10 == 0:
            print(f"Step {i}: MAE {mae:.4f}")
            print(f"  GT:   {gt_action}")
            print(f"  Pred: {pred_action}")
            
    print(f"Overall MAE: {np.mean(maes):.4f}")
    return np.mean(maes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    # Load Policy
    print("Loading Policy...")
    policy = load_policy_with_lora_support(
        model_path=args.checkpoint,
        denoising_steps=args.denoising_steps
    )
    
    # Load Dataset
    print("Loading Dataset...")
    data_cfg = load_data_config("so100_dualcam")
    modality_config = data_cfg.modality_config()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=modality_config,
        video_backend="torchvision_av",
        embodiment_tag="new_embodiment"
    )

    # 1. Consistency Test
    print(f"Checking consistency with {args.denoising_steps} denoising steps...")
    calc_std, _ = calculate_consistency(policy, dataset, num_trials=10, traj_id=0, step=20)
    
    # 2. Temporal Smoothness
    print("Checking temporal smoothness...")
    calc_jumpiness = check_trajectory_smoothness(policy, dataset, steps=20)
    
    # 3. Ground Truth
    print("Checking ground truth tracking...")
    calc_mae = compare_to_ground_truth(policy, dataset, steps=30)
    
    # Diagnosis Logic
    print("\n=== DIAGNOSIS REPORT ===")
    if calc_std > 1.0:
        print(f"[FAIL] High Noise: Model output varies by {calc_std:.4f} deg for SAME input.")
        print("-> Likely cause of 'tight vibration'.")
        print("-> RECOMMENDATION: Increase denoising steps.")
    else:
        print(f"[PASS] Low Noise: {calc_std:.4f} deg.")
        
    if calc_jumpiness > 2.0:
        print(f"[FAIL] Temporal Inconsistency: Predictions jump by {calc_jumpiness:.4f} deg between frames.")
        print("-> Causes 'fighting' against previous commands.")
    else:
        print(f"[PASS] Temporal Consistency: {calc_jumpiness:.4f} deg.")
        
    if calc_mae > 10.0:
        print(f"[FAIL] Poor Tracking: MAE {calc_mae:.4f} deg.")
        print("-> Model is not following the trajectory well.")
    else:
        print(f"[PASS] Good Tracking: MAE {calc_mae:.4f} deg.")

if __name__ == "__main__":
    main()

