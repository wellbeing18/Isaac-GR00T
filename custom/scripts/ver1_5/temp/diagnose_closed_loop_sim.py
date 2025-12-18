#!/usr/bin/env python3
"""
Closed-Loop Simulation for GR00T Error Accumulation Analysis

This script simulates real inference by feeding predicted states back to the model
instead of using ground truth states from the dataset. This quantifies the error
accumulation that occurs in real robot deployment.

Key insight: Open-loop evaluation uses ground truth states, so errors don't compound.
In real deployment, errors compound because each step uses the ACTUAL robot state
which includes previous prediction errors.

Usage:
    # Run closed-loop simulation on dataset
    python diagnose_closed_loop_sim.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/datasets_groot \
        --steps 100

    # Compare with open-loop
    python diagnose_closed_loop_sim.py \
        --checkpoint /path/to/checkpoint \
        --compare-openloop
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*video decoding.*")
warnings.filterwarnings("ignore", message=".*albumentations.*")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Joint names for SO-101 arm
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class SimulationStep:
    """Data for a single simulation step."""
    step: int
    state_gt: np.ndarray  # Ground truth state
    state_sim: np.ndarray  # Simulated state (accumulated errors)
    action_pred: np.ndarray  # Predicted action
    action_gt: np.ndarray  # Ground truth action
    state_error: np.ndarray  # Difference between sim and gt state
    action_error: np.ndarray  # Difference between pred and gt action


@dataclass
class SimulationResult:
    """Results from a closed-loop simulation run."""
    trajectory_id: int
    num_steps: int
    # Open-loop metrics (using GT state)
    openloop_mse: float
    openloop_mae: float
    openloop_per_joint_mse: Dict[str, float]
    # Closed-loop metrics (using simulated state)
    closedloop_mse: float
    closedloop_mae: float
    closedloop_per_joint_mse: Dict[str, float]
    # State drift metrics
    final_state_drift: np.ndarray
    total_state_drift: float
    max_state_drift_per_joint: Dict[str, float]
    # Time series
    state_drift_over_time: List[float]


def load_policy(
    model_path: str,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
    denoising_steps: int = 4,
):
    """Load GR00T policy with LoRA support."""
    sys.path.insert(0, str(Path(__file__).parent))
    from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora

    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    if is_lora_checkpoint(model_path):
        print("[MODEL] Loading LoRA checkpoint...")
        policy = load_groot_with_lora(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            merge_weights=True,
        )
    else:
        print("[MODEL] Loading full checkpoint...")
        policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
        )

    return policy


def load_dataset(
    dataset_path: str,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
):
    """Load GR00T dataset."""
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    import decord
    decord.bridge.set_bridge("native")

    data_cfg = load_data_config(data_config)
    modality_config = data_cfg.modality_config()

    dataset = LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=modality_config,
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=embodiment_tag,
    )

    return dataset


def get_observation_with_state(
    dataset,
    traj_id: int,
    step: int,
    override_state: Optional[np.ndarray] = None,
) -> Dict:
    """
    Get observation from dataset, optionally overriding the state.

    This is the key function for closed-loop simulation:
    - In open-loop: Use ground truth state from dataset
    - In closed-loop: Use simulated state (with accumulated errors)
    """
    obs = dataset.get_step_data(traj_id, step)

    if override_state is not None:
        # Override state with simulated state
        obs["state.single_arm"] = override_state[:5][np.newaxis, :].astype(np.float64)
        obs["state.gripper"] = override_state[5:6][np.newaxis, :].astype(np.float64)

    return obs


def run_inference(policy, obs: Dict) -> np.ndarray:
    """Run policy inference and return first action (6D)."""
    with torch.no_grad():
        action = policy.get_action(obs)

    single_arm = action["action.single_arm"][0]  # First timestep
    gripper = action["action.gripper"][0]

    return np.concatenate([single_arm, gripper])


def simulate_state_transition(
    current_state: np.ndarray,
    action: np.ndarray,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Simulate state transition: next_state = action + noise

    In SO101, actions are absolute joint positions, so:
    next_state ≈ action (the robot moves to the commanded position)

    Args:
        current_state: Current joint positions (6D)
        action: Commanded joint positions (6D)
        noise_std: Optional noise to simulate execution error

    Returns:
        Simulated next state
    """
    next_state = action.copy()

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=action.shape)
        next_state += noise

    return next_state


def run_closed_loop_simulation(
    policy,
    dataset,
    traj_id: int,
    num_steps: int = 100,
    execution_noise_std: float = 0.0,
) -> SimulationResult:
    """
    Run closed-loop simulation on a trajectory.

    Compares:
    1. Open-loop: Each step uses ground truth state
    2. Closed-loop: Each step uses simulated state (previous prediction + noise)
    """
    traj_length = dataset.trajectory_lengths[traj_id]
    num_steps = min(num_steps, traj_length - 1)

    print(f"\n[SIM] Running simulation on trajectory {traj_id} ({num_steps} steps)")

    # Storage for results
    openloop_errors = []
    closedloop_errors = []
    state_drifts = []
    steps_data = []

    # Initialize simulated state from ground truth at step 0
    obs_0 = dataset.get_step_data(traj_id, 0)
    simulated_state = np.concatenate([
        obs_0["state.single_arm"][0],
        obs_0["state.gripper"][0]
    ])

    for step in range(num_steps):
        # Get ground truth observation
        obs_gt = dataset.get_step_data(traj_id, step)
        state_gt = np.concatenate([
            obs_gt["state.single_arm"][0],
            obs_gt["state.gripper"][0]
        ])
        action_gt = np.concatenate([
            obs_gt["action.single_arm"][0],
            obs_gt["action.gripper"][0]
        ])

        # OPEN-LOOP: Inference with ground truth state
        action_pred_openloop = run_inference(policy, obs_gt)
        openloop_error = np.abs(action_pred_openloop - action_gt)
        openloop_errors.append(openloop_error)

        # CLOSED-LOOP: Inference with simulated state
        obs_sim = get_observation_with_state(dataset, traj_id, step, simulated_state)
        action_pred_closedloop = run_inference(policy, obs_sim)
        closedloop_error = np.abs(action_pred_closedloop - action_gt)
        closedloop_errors.append(closedloop_error)

        # Calculate state drift
        state_error = np.abs(simulated_state - state_gt)
        state_drifts.append(np.sum(state_error))

        # Record step data
        steps_data.append(SimulationStep(
            step=step,
            state_gt=state_gt.copy(),
            state_sim=simulated_state.copy(),
            action_pred=action_pred_closedloop.copy(),
            action_gt=action_gt.copy(),
            state_error=state_error.copy(),
            action_error=closedloop_error.copy(),
        ))

        # Update simulated state for next step (closed-loop feedback)
        simulated_state = simulate_state_transition(
            simulated_state,
            action_pred_closedloop,
            noise_std=execution_noise_std,
        )

        if (step + 1) % 20 == 0:
            print(f"[SIM] Step {step+1}/{num_steps}, state drift: {state_drifts[-1]:.2f}°")

    # Calculate statistics
    openloop_errors = np.array(openloop_errors)
    closedloop_errors = np.array(closedloop_errors)

    # Per-joint MSE
    openloop_per_joint = {
        name: float(np.mean(openloop_errors[:, i]**2))
        for i, name in enumerate(JOINT_NAMES)
    }
    closedloop_per_joint = {
        name: float(np.mean(closedloop_errors[:, i]**2))
        for i, name in enumerate(JOINT_NAMES)
    }

    # Final state drift
    final_step = steps_data[-1]
    final_drift = final_step.state_error
    max_drift_per_joint = {
        name: float(max(s.state_error[i] for s in steps_data))
        for i, name in enumerate(JOINT_NAMES)
    }

    result = SimulationResult(
        trajectory_id=traj_id,
        num_steps=num_steps,
        openloop_mse=float(np.mean(openloop_errors**2)),
        openloop_mae=float(np.mean(openloop_errors)),
        openloop_per_joint_mse=openloop_per_joint,
        closedloop_mse=float(np.mean(closedloop_errors**2)),
        closedloop_mae=float(np.mean(closedloop_errors)),
        closedloop_per_joint_mse=closedloop_per_joint,
        final_state_drift=final_drift,
        total_state_drift=float(np.sum(final_drift)),
        max_state_drift_per_joint=max_drift_per_joint,
        state_drift_over_time=state_drifts,
    )

    return result, steps_data


def plot_simulation_results(
    result: SimulationResult,
    steps_data: List[SimulationStep],
    save_path: Optional[str] = None,
):
    """Generate visualization of simulation results."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    steps = [s.step for s in steps_data]

    # Plot 1: State drift over time
    ax = axes[0, 0]
    ax.plot(steps, result.state_drift_over_time, 'b-', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Total State Drift (degrees)")
    ax.set_title("State Drift Accumulation Over Time")
    ax.grid(True, alpha=0.3)

    # Plot 2: Open-loop vs Closed-loop MSE comparison
    ax = axes[0, 1]
    x = np.arange(len(JOINT_NAMES))
    width = 0.35
    openloop_vals = [result.openloop_per_joint_mse[j] for j in JOINT_NAMES]
    closedloop_vals = [result.closedloop_per_joint_mse[j] for j in JOINT_NAMES]
    ax.bar(x - width/2, openloop_vals, width, label='Open-loop', color='green', alpha=0.7)
    ax.bar(x + width/2, closedloop_vals, width, label='Closed-loop', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([j[:8] for j in JOINT_NAMES], rotation=45, ha='right')
    ax.set_ylabel("MSE (degrees²)")
    ax.set_title("Per-Joint MSE: Open-loop vs Closed-loop")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3-6: Per-joint trajectories (first 4 joints)
    for i, (ax, joint_name) in enumerate(zip(axes.flat[2:], JOINT_NAMES[:4])):
        gt_states = [s.state_gt[i] for s in steps_data]
        sim_states = [s.state_sim[i] for s in steps_data]

        ax.plot(steps, gt_states, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(steps, sim_states, 'r--', label='Simulated', linewidth=2, alpha=0.8)
        ax.fill_between(steps, gt_states, sim_states, alpha=0.2, color='red')
        ax.set_xlabel("Step")
        ax.set_ylabel("Position (degrees)")
        ax.set_title(f"{joint_name} - State Comparison")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Closed-Loop Simulation - Trajectory {result.trajectory_id}\n"
        f"Open-loop MSE: {result.openloop_mse:.2f} | Closed-loop MSE: {result.closedloop_mse:.2f} | "
        f"Final Drift: {result.total_state_drift:.2f}°",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def print_results(result: SimulationResult):
    """Print simulation results."""
    print("\n" + "="*70)
    print(f"CLOSED-LOOP SIMULATION RESULTS - Trajectory {result.trajectory_id}")
    print("="*70)

    print(f"\nSteps simulated: {result.num_steps}")

    print(f"\n{'Metric':<25} {'Open-Loop':>15} {'Closed-Loop':>15} {'Ratio':>10}")
    print("-"*70)
    print(f"{'Overall MSE':<25} {result.openloop_mse:>14.2f}° {result.closedloop_mse:>14.2f}° {result.closedloop_mse/max(result.openloop_mse, 0.01):>9.1f}x")
    print(f"{'Overall MAE':<25} {result.openloop_mae:>14.2f}° {result.closedloop_mae:>14.2f}° {result.closedloop_mae/max(result.openloop_mae, 0.01):>9.1f}x")

    print(f"\nPer-Joint MSE (degrees²):")
    print(f"{'Joint':<15} {'Open-Loop':>12} {'Closed-Loop':>12} {'Max Drift':>12}")
    print("-"*55)
    for joint in JOINT_NAMES:
        ol = result.openloop_per_joint_mse[joint]
        cl = result.closedloop_per_joint_mse[joint]
        md = result.max_state_drift_per_joint[joint]
        print(f"{joint:<15} {ol:>11.2f}° {cl:>11.2f}° {md:>11.2f}°")

    print(f"\nState Drift Analysis:")
    print(f"  Final total drift:  {result.total_state_drift:.2f}°")
    print(f"  Final per-joint:    {np.round(result.final_state_drift, 2)}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    ratio = result.closedloop_mse / max(result.openloop_mse, 0.01)
    if ratio > 5:
        print("  [CRITICAL] Closed-loop MSE is >5x higher than open-loop!")
        print("             Error accumulation is SEVERE.")
        print("             Real robot will drift significantly.")
    elif ratio > 2:
        print("  [WARNING] Closed-loop MSE is 2-5x higher than open-loop.")
        print("            Moderate error accumulation.")
        print("            Consider more training or error correction.")
    else:
        print("  [OK] Closed-loop MSE is similar to open-loop.")
        print("       Model is relatively robust to state errors.")

    if result.total_state_drift > 50:
        print(f"\n  [CRITICAL] Final state drift ({result.total_state_drift:.0f}°) is very high!")
        print("             Robot will be in completely wrong position.")
    elif result.total_state_drift > 20:
        print(f"\n  [WARNING] Final state drift ({result.total_state_drift:.0f}°) is moderate.")
        print("            May need periodic state reset.")


def main():
    parser = argparse.ArgumentParser(
        description="Closed-Loop Simulation for Error Accumulation Analysis"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to GR00T checkpoint"
    )
    parser.add_argument(
        "--dataset", "-d", type=str,
        default="/home/jrobot/project/XLeRobot/datasets_groot",
        help="Path to dataset"
    )
    parser.add_argument(
        "--traj-id", type=int, default=0,
        help="Trajectory ID to simulate"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of steps to simulate"
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.0,
        help="Execution noise std (degrees)"
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--save-plot", type=str,
        help="Save plot to this path"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Save results JSON to this path"
    )

    args = parser.parse_args()

    print("="*70)
    print("CLOSED-LOOP SIMULATION FOR ERROR ACCUMULATION")
    print("="*70)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Dataset:         {args.dataset}")
    print(f"Trajectory ID:   {args.traj_id}")
    print(f"Steps:           {args.steps}")
    print(f"Execution noise: {args.noise_std}°")
    print("="*70)

    # Load model
    print("\n[LOAD] Loading model...")
    policy = load_policy(
        args.checkpoint,
        denoising_steps=args.denoising_steps,
    )

    # Load dataset
    print("\n[LOAD] Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"[LOAD] Dataset has {len(dataset.trajectory_lengths)} trajectories")

    # Run simulation
    result, steps_data = run_closed_loop_simulation(
        policy=policy,
        dataset=dataset,
        traj_id=args.traj_id,
        num_steps=args.steps,
        execution_noise_std=args.noise_std,
    )

    # Print results
    print_results(result)

    # Generate plot
    if args.save_plot:
        plot_simulation_results(result, steps_data, args.save_plot)
    else:
        # Save to default location
        default_plot = f"closedloop_sim_traj{args.traj_id}.png"
        plot_simulation_results(result, steps_data, default_plot)

    # Save results
    if args.output:
        result_dict = {
            "trajectory_id": result.trajectory_id,
            "num_steps": result.num_steps,
            "openloop_mse": result.openloop_mse,
            "openloop_mae": result.openloop_mae,
            "openloop_per_joint_mse": result.openloop_per_joint_mse,
            "closedloop_mse": result.closedloop_mse,
            "closedloop_mae": result.closedloop_mae,
            "closedloop_per_joint_mse": result.closedloop_per_joint_mse,
            "total_state_drift": result.total_state_drift,
            "max_state_drift_per_joint": result.max_state_drift_per_joint,
            "state_drift_over_time": result.state_drift_over_time,
        }
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\n[SAVE] Results saved to: {args.output}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
