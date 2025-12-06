#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SO101 GR00T Inference Script - All-in-one version
# Adapted from examples/SO-100/eval_gr00t_so100.py for dual-camera SO101 setup
#
# FPS Configuration (per 6_fps_upgrade_30hz.md):
#   - Action frequency: 30 Hz (synchronized with video)
#   - Camera FPS: 30 fps
#   - Action interval: 0.033s (1/30 Hz)
#
# Usage:
#   python infer_groot_so101.py --model-path /path/to/checkpoint
#   python infer_groot_so101.py --help
#

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# LeRobot imports for robot control (v0.4.1 API)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError

# GR00T imports for inference
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

from tqdm import tqdm


#################################################################################
# Camera Quality Check Utility (for diagnostic mode)
#################################################################################


def check_camera_quality(frame: np.ndarray, black_threshold: int = 10) -> dict:
    """
    Check camera frame quality for corruption detection.

    Returns dict with:
    - is_corrupt: bool
    - corruption_score: float (0-1, higher = more corruption)
    - black_row_count: int
    - mean_brightness: float
    """
    if frame is None or frame.size == 0:
        return {
            "is_corrupt": True,
            "corruption_score": 1.0,
            "black_row_count": 0,
            "mean_brightness": 0.0,
        }

    # Convert to grayscale for analysis
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] == 3 else frame[:, :, 0]
    else:
        gray = frame

    # Find rows that are mostly black (potential corruption)
    row_means = np.mean(gray, axis=1)
    black_rows = int(np.sum(row_means < black_threshold))

    # Calculate corruption score
    total_rows = gray.shape[0]
    corruption_score = black_rows / total_rows

    # Consider corrupt if more than 5% of rows are black
    is_corrupt = corruption_score > 0.05

    return {
        "is_corrupt": is_corrupt,
        "corruption_score": float(corruption_score),
        "black_row_count": black_rows,
        "mean_brightness": float(np.mean(gray)),
    }


#################################################################################
# LoRA Checkpoint Detection and Loading Utilities
#################################################################################


def is_lora_checkpoint(model_path: str) -> bool:
    """
    Detect if the given path is a LoRA (PEFT) checkpoint.

    LoRA checkpoints have:
    - adapter_config.json (PEFT configuration)
    - adapter_model.safetensors (LoRA weights only, ~13MB vs ~10GB for full)
    """
    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"
    adapter_model = model_path / "adapter_model.safetensors"

    return adapter_config.exists() and adapter_model.exists()


def get_lora_config(model_path: str) -> dict:
    """Load and return the PEFT/LoRA configuration."""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    with open(adapter_config_path, "r") as f:
        return json.load(f)


def load_groot_with_lora(
    model_path: str,
    embodiment_tag: str,
    modality_config: dict,
    modality_transform,
    denoising_steps: int = 4,
    merge_weights: bool = True,
) -> Gr00tPolicy:
    """
    Load GR00T model with LoRA adapters merged.

    This is the critical fix for Issue #1 from the pre-flight checklist:
    GR00T training saves PEFT adapters separately, but the standard
    Gr00tPolicy doesn't load them. This function:

    1. Loads the base GR00T model
    2. Loads the PEFT LoRA adapter
    3. Merges LoRA weights into base model (or keeps as adapter)
    4. Returns a policy ready for inference

    Args:
        model_path: Path to the LoRA checkpoint (with adapter_config.json)
        embodiment_tag: Embodiment tag for the model
        modality_config: Modality configuration
        modality_transform: Modality transform for normalization
        denoising_steps: Number of denoising steps
        merge_weights: If True, merge LoRA into base and unload adapter
                      If False, keep as PEFT model (uses more memory)

    Returns:
        Gr00tPolicy with LoRA weights loaded
    """
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    model_path = Path(model_path)

    # Load adapter config to get base model path
    lora_config = get_lora_config(model_path)
    base_model_path = lora_config.get("base_model_name_or_path")

    print(f"[LoRA] Detected LoRA checkpoint")
    print(f"[LoRA] Base model: {base_model_path}")
    print(f"[LoRA] LoRA rank: {lora_config.get('r')}")
    print(f"[LoRA] LoRA alpha: {lora_config.get('lora_alpha')}")
    print(f"[LoRA] Target modules: {lora_config.get('target_modules')}")

    # Step 1: Load base GR00T model
    print(f"[LoRA] Step 1/3: Loading base GR00T model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
    base_model.eval()

    # Step 2: Load PEFT adapter onto base model
    print(f"[LoRA] Step 2/3: Loading PEFT adapter from {model_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(model_path))

    # Step 3: Merge LoRA weights into base weights (optional but recommended)
    if merge_weights:
        print(f"[LoRA] Step 3/3: Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()
        print(f"[LoRA] Merge complete! Model is now a standard GR00T model with finetuned weights.")
    else:
        merged_model = peft_model
        print(f"[LoRA] Keeping as PEFT model (adapter mode).")

    # Move to device
    merged_model.to(device=device)

    # Create a Gr00tPolicy wrapper
    # We need to manually set up the policy since we're not using from_pretrained
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.device = device
    policy._modality_config = modality_config
    policy._modality_transform = modality_transform
    policy.model = merged_model

    # Set denoising steps
    policy.model.action_head.num_inference_timesteps = denoising_steps

    # Load metadata for normalization
    exp_cfg_dir = model_path / "experiment_cfg"
    if exp_cfg_dir.exists():
        metadata_path = exp_cfg_dir / "metadata.json"
        if metadata_path.exists():
            print(f"[LoRA] Loading normalization metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                metadatas = json.load(f)

            from gr00t.data.dataset import DatasetMetadata
            from gr00t.model.policy import EmbodimentTag

            # Get embodiment tag enum
            if isinstance(embodiment_tag, str):
                embodiment_tag_enum = EmbodimentTag(embodiment_tag)
            else:
                embodiment_tag_enum = embodiment_tag

            policy.embodiment_tag = embodiment_tag_enum

            metadata_dict = metadatas.get(embodiment_tag_enum.value)
            if metadata_dict:
                metadata = DatasetMetadata.model_validate(metadata_dict)
                policy._modality_transform.set_metadata(metadata)
                policy.metadata = metadata
                print(f"[LoRA] Loaded normalization stats for '{embodiment_tag_enum.value}'")
            else:
                print(f"[LoRA] WARNING: No metadata found for embodiment '{embodiment_tag_enum.value}'")

    # Load horizons from modality config
    policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
    policy._video_horizon = len(policy._video_delta_indices)
    if "state" in modality_config:
        policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
        policy._state_horizon = len(policy._state_delta_indices)
    policy._action_delta_indices = np.array(modality_config["action"].delta_indices)
    # Note: Gr00tPolicy doesn't expose _action_horizon/indices publicly by default,
    # but they are used internally for prediction logic
    
    # Try setting them if attributes exist, otherwise rely on internal load
    if hasattr(policy, '_action_horizon'):
        policy._action_horizon = len(policy._action_delta_indices)

    print(f"[LoRA] GR00T model with LoRA loaded successfully!")
    print(f"[LoRA] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return policy


#################################################################################
# SO101 Robot Class with Dual Camera Support (LeRobot v0.4.1 API)
#################################################################################


class SO101Robot:
    """
    SO101 Robot control with dual camera support.
    Uses SO101Follower from lerobot v0.4.1.
    """

    # Motor name to index mapping for state/action arrays
    MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(
        self,
        port: str = "/dev/ttyACM2",
        robot_id: str = "xlerobot_left_arm",
        calibrate: bool = False,
        head_cam_idx: int = 4,
        wrist_cam_idx: int = 6,
        use_degrees: bool = True,
    ):
        self.calibrate = calibrate
        self.robot_id = robot_id
        self.head_cam_idx = head_cam_idx
        self.wrist_cam_idx = wrist_cam_idx

        # Handle calibration if needed - delete existing calibration
        if self.calibrate:
            import shutil
            # LeRobot v0.4+ stores calibration in huggingface cache
            calibration_folder = os.path.expanduser(f"~/.cache/huggingface/lerobot/calibration/robots/so101_follower")
            print(f"========> Deleting calibration folder: {calibration_folder}")
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Configure dual cameras (lerobot v0.4.1 format)
        cameras = {
            "head": OpenCVCameraConfig(
                index_or_path=head_cam_idx,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_idx,
                fps=30,
                width=640,
                height=480,
            ),
        }

        # Create the robot config with robot ID for calibration
        self.config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            cameras=cameras,
            use_degrees=use_degrees,
        )

        # Create the robot
        self.robot = SO101Follower(self.config)

    @contextmanager
    def activate(self):
        """Context manager for robot activation."""
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        """Connect to the robot and cameras."""
        if self.robot.is_connected:
            raise DeviceAlreadyConnectedError(
                "SO101Robot is already connected. Do not run connect() twice."
            )

        # Connect (includes calibration if needed)
        self.robot.connect(calibrate=self.calibrate)
        print("================> SO101 Robot fully connected (dual cameras) =================")

    def move_to_initial_pose(self):
        """Move robot to initial pose (in degrees)."""
        # Initial pose: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        initial_action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,  # Half open
        }
        self.robot.send_action(initial_action)
        time.sleep(2)
        print("-------------------------------- Moved to initial pose")

    def go_home(self, training_aligned: bool = False):
        """Move robot to home position.

        Args:
            training_aligned: If True, move to a position within training data range.
                             If False, move to the default home (all zeros, gripper open).
        """
        print("-------------------------------- Moving to home pose")
        if training_aligned:
            # Training-aligned home position (within training data ranges)
            # Based on training data means:
            #   shoulder_pan: mean=1.7, shoulder_lift: mean=-19.7
            #   elbow_flex: mean=21.7, wrist_flex: mean=59.3
            #   wrist_roll: mean=0.4, gripper: mean=4.6
            home_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": -20.0,
                "elbow_flex.pos": 20.0,
                "wrist_flex.pos": 60.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 5.0,  # Slightly open, NOT 50°!
            }
            print("  Using training-aligned home (within training range)")
        else:
            home_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 50.0,
            }
        self.robot.send_action(home_action)
        time.sleep(2)

    def get_observation(self):
        """Get full observation from robot."""
        return self.robot.get_observation()

    def get_current_state(self) -> np.ndarray:
        """Get current robot state as numpy array (6D: 5 arm + 1 gripper)."""
        obs = self.get_observation()
        # Extract motor positions in order
        state = np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ])
        return state

    def get_head_image(self) -> np.ndarray:
        """Get head camera image in RGB format."""
        obs = self.get_observation()
        return obs["head"]  # Already RGB from OpenCVCameraConfig default

    def get_wrist_image(self) -> np.ndarray:
        """Get wrist camera image in RGB format."""
        obs = self.get_observation()
        return obs["wrist"]  # Already RGB from OpenCVCameraConfig default

    def get_dual_images(self) -> tuple:
        """Get both camera images as (head_img, wrist_img) in RGB format."""
        obs = self.get_observation()
        return obs["head"], obs["wrist"]

    def set_target_state(self, target_state):
        """Send action to robot.

        Args:
            target_state: numpy array or torch tensor of shape (6,)
                         [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        if isinstance(target_state, torch.Tensor):
            target_state = target_state.numpy()

        action = {
            "shoulder_pan.pos": float(target_state[0]),
            "shoulder_lift.pos": float(target_state[1]),
            "elbow_flex.pos": float(target_state[2]),
            "wrist_flex.pos": float(target_state[3]),
            "wrist_roll.pos": float(target_state[4]),
            "gripper.pos": float(target_state[5]),
        }
        self.robot.send_action(action)

    def disconnect(self):
        """Disconnect robot and cameras."""
        self.robot.disconnect()
        print("================> SO101 Robot disconnected")


#################################################################################
# GR00T Local Inference Class
#################################################################################


class Gr00tLocalInference:
    """
    All-in-one GR00T model loading and inference for SO101 dual-camera setup.

    Automatically detects and handles:
    - Full model checkpoints (standard Gr00tPolicy loading)
    - LoRA/PEFT checkpoints (loads adapter and merges weights)
    """

    def __init__(
        self,
        model_path: str,
        data_config: str = "so100_dualcam",
        embodiment_tag: str = "new_embodiment",
        task: str = "pick the red cube from the table",  # Must match training task
        denoising_steps: int = 4,
    ):
        self.task = task
        self.data_config_name = data_config

        print(f"Loading GR00T model from: {model_path}")
        print(f"Data config: {data_config}")
        print(f"Embodiment tag: {embodiment_tag}")
        print(f"Task: {task}")

        # Load data config for modality configuration and transforms
        data_cfg = load_data_config(data_config)
        modality_config = data_cfg.modality_config()
        modality_transform = data_cfg.transform()

        # Auto-detect checkpoint type and load appropriately
        if is_lora_checkpoint(model_path):
            print(f"\n{'='*60}")
            print(f"[INFO] Detected LoRA checkpoint - using PEFT adapter loading")
            print(f"{'='*60}\n")

            # Use our custom LoRA loading function
            self.policy = load_groot_with_lora(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
                merge_weights=True,  # Merge LoRA into base for efficient inference
            )
        else:
            print(f"\n{'='*60}")
            print(f"[INFO] Full checkpoint detected - using standard loading")
            print(f"{'='*60}\n")

            # Standard loading for full checkpoints
            self.policy = Gr00tPolicy(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
            )

        print(f"Model loaded successfully! Denoising steps: {denoising_steps}")

    def get_action(self, front_img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray) -> dict:
        """
        Run inference and return action dictionary.

        Args:
            front_img: Head/front camera image (H, W, 3) RGB
            wrist_img: Wrist camera image (H, W, 3) RGB
            state: Robot state (6,) - 5 arm joints + 1 gripper

        Returns:
            Action dictionary with 'action.single_arm' and 'action.gripper' keys
        """
        # Build observation dict matching so100_dualcam config
        # Keys: video.front, video.wrist, state.single_arm, state.gripper
        obs_dict = {
            "video.front": front_img[np.newaxis, :, :, :],  # (1, H, W, 3)
            "video.wrist": wrist_img[np.newaxis, :, :, :],  # (1, H, W, 3)
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),  # (1, 5)
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),  # (1, 1)
            "annotation.human.task_description": [self.task],
        }

        # Get action from policy
        return self.policy.get_action(obs_dict)

    def set_task(self, task: str):
        """Update the task description."""
        self.task = task
        print(f"Task updated to: {task}")


#################################################################################
# Visualization Helper
#################################################################################


def view_dual_images(head_img, wrist_img, title="Camera Views"):
    """Display both camera images side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(head_img)
    ax1.set_title("Head Camera")
    ax1.axis("off")
    ax2.imshow(wrist_img)
    ax2.set_title("Wrist Camera")
    ax2.axis("off")
    plt.suptitle(title)
    plt.pause(0.001)
    plt.clf()


def view_single_image(img, title="Camera View"):
    """Display a single camera image."""
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.pause(0.001)
    plt.clf()


#################################################################################
# Main
#################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="SO101 GR00T Inference - All-in-one script for finetuned model"
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the finetuned GR00T checkpoint (e.g., /path/to/outputs/groot_mvp_lora_XXX/best)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="so100_dualcam",
        help="Data configuration name (default: so100_dualcam)",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag for the model (default: new_embodiment)",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Number of denoising steps (default: 4)",
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="pick the red cube from the table",
        help="Task description (must match training task, e.g., 'pick the red cube from the table')",
    )

    # Robot configuration
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM2",
        help="Serial port for robot (default: /dev/ttyACM2 for left follower arm)",
    )
    parser.add_argument(
        "--head-cam-idx",
        type=int,
        default=4,
        help="Head camera device index (default: 4)",
    )
    parser.add_argument(
        "--wrist-cam-idx",
        type=int,
        default=6,
        help="Wrist camera device index (default: 6)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Force robot calibration",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="xlerobot_left_arm",
        help="Robot ID for calibration storage (default: xlerobot_left_arm, matches data collection)",
    )

    # Execution configuration
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=12,
        help="Number of actions to execute per inference (default: 12 of 16)",
    )
    parser.add_argument(
        "--actions-to-execute",
        type=int,
        default=100,
        help="Total number of action chunks to execute (default: 100)",
    )
    parser.add_argument(
        "--action-interval",
        type=float,
        default=0.033,
        help="Time interval between actions in seconds (default: 0.033 = 30Hz)",
    )

    # Visualization
    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable image display (disabled by default)",
    )
    parser.add_argument(
        "--record-imgs",
        action="store_true",
        help="Record images to eval_images/ folder",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Save inference log to file (e.g., inference.log)",
    )
    parser.add_argument(
        "--go-home-first",
        action="store_true",
        help="Move robot to home position before starting inference (use if robot is in unknown state)",
    )

    # Diagnostic mode
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Enable comprehensive diagnostic logging (saves timing, states, actions to JSON)",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=str,
        default="diagnostic_output",
        help="Directory for diagnostic output files (default: diagnostic_output)",
    )

    args = parser.parse_args()

    # Setup diagnostic mode data structures
    diagnostic_data = None
    if args.diagnostic_mode:
        os.makedirs(args.diagnostic_output, exist_ok=True)
        diagnostic_data = {
            "config": {
                "model_path": args.model_path,
                "action_horizon": args.action_horizon,
                "action_interval": args.action_interval,
                "denoising_steps": args.denoising_steps,
                "task": args.task,
            },
            "timing": [],
            "states": [],
            "actions": [],
            "camera_metrics": [],
        }
        print(f"[DIAGNOSTIC] Mode enabled. Output: {args.diagnostic_output}")

    # Setup logging to file if requested
    log_file = None
    if args.log_file:
        import datetime
        log_file = open(args.log_file, "w")
        def log_print(*args_print, **kwargs):
            message = " ".join(str(a) for a in args_print)
            print(message, **kwargs)
            log_file.write(message + "\n")
            log_file.flush()
    else:
        log_print = print

    # Known training tasks (from datasets_groot/meta/tasks.jsonl)
    TRAINING_TASKS = [
        "grasp the red cube",
        "pick the red cube from the table",
        "pick the green cube from the table",
        "place the red cube in the white bowl",
        "place the green cube in the white bowl",
        "push the red cube to the green cube",
        "reach the red cube",
        "release",
    ]

    # Print configuration
    print("=" * 70)
    print("SO101 GR00T Inference - All-in-one")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Data config: {args.data_config}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Task: {args.task}")

    # Validate task description
    if args.task not in TRAINING_TASKS:
        print()
        print("⚠️  WARNING: Task description does not match training tasks!")
        print(f"   Your task: '{args.task}'")
        print(f"   Training tasks:")
        for t in TRAINING_TASKS:
            print(f"     - '{t}'")
        print()
        print("   GR00T uses task description for conditioning. Mismatched tasks may cause poor performance.")
        print("   Consider using: --task 'pick the red cube from the table'")
        print()

    print(f"Robot port: {args.port}")
    print(f"Robot ID: {args.robot_id}")
    print(f"Head camera index: {args.head_cam_idx}")
    print(f"Wrist camera index: {args.wrist_cam_idx}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Actions to execute: {args.actions_to_execute}")
    print("=" * 70)

    # Initialize inference
    print("\nLoading model...")
    inference = Gr00tLocalInference(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        task=args.task,
        denoising_steps=args.denoising_steps,
    )

    # Setup recording if enabled
    if args.record_imgs:
        os.makedirs("eval_images", exist_ok=True)
        for file in os.listdir("eval_images"):
            os.remove(os.path.join("eval_images", file))
        print("Recording images to eval_images/")

    # Initialize robot
    print("\nInitializing robot...")
    robot = SO101Robot(
        port=args.port,
        robot_id=args.robot_id,
        calibrate=args.calibrate,
        head_cam_idx=args.head_cam_idx,
        wrist_cam_idx=args.wrist_cam_idx,
    )

    # Action modality keys for concatenation
    MODALITY_KEYS = ["single_arm", "gripper"]
    image_count = 0

    # Run inference loop
    with robot.activate():
        # Go to home position first if requested
        if args.go_home_first:
            print("\n[RESET] Moving to training-aligned home position before inference...")
            print("  This ensures the robot starts from a state similar to training data.")
            robot.go_home(training_aligned=True)
            time.sleep(1.0)  # Wait for robot to settle
            state = robot.get_current_state()
            print(f"  Home state: {np.round(state, 2)}")
            print(f"    Expected: [0, -20, 20, 60, 0, 5] (training mean positions)")
            print("  Ready to start inference.\n")

        # WARMUP PHASE
        print("\nRunning warmup inference (10 steps)...")
        # Dummy inputs for warmup
        dummy_head = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_wrist = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_state = np.zeros(6)
        for _ in range(10):
            inference.get_action(dummy_head, dummy_wrist, dummy_state)
        print("Warmup complete!\n")

        print("\nStarting inference loop...")
        print(f"Press Ctrl+C to stop\n")

        try:
            last_inference_time = time.time()
            # Initialize previous action for temporal ensembling
            prev_action_chunk = None
            
            for chunk_idx in tqdm(range(args.actions_to_execute), desc="Executing action chunks"):
                # Get observations
                obs_start = time.time()
                head_img, wrist_img = robot.get_dual_images()
                state = robot.get_current_state()
                obs_time = time.time() - obs_start

                # Diagnostic: Check camera quality
                if diagnostic_data is not None:
                    head_quality = check_camera_quality(head_img)
                    wrist_quality = check_camera_quality(wrist_img)
                    diagnostic_data["camera_metrics"].append({
                        "chunk": chunk_idx,
                        "timestamp": time.time(),
                        "head": head_quality,
                        "wrist": wrist_quality,
                        "obs_time_ms": obs_time * 1000,
                    })
                    # Alert on corruption
                    if head_quality["is_corrupt"] or wrist_quality["is_corrupt"]:
                        print(f"\n[DIAGNOSTIC] Camera corruption detected at chunk {chunk_idx}!")
                        if head_quality["is_corrupt"]:
                            print(f"  Head: score={head_quality['corruption_score']:.2f}, black_rows={head_quality['black_row_count']}")
                        if wrist_quality["is_corrupt"]:
                            print(f"  Wrist: score={wrist_quality['corruption_score']:.2f}, black_rows={wrist_quality['black_row_count']}")
                        # Save corrupt frame
                        cv2.imwrite(
                            f"{args.diagnostic_output}/corrupt_head_{chunk_idx:05d}.jpg",
                            cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
                        )

                # Record first image to verify camera mapping
                if chunk_idx == 0 and args.record_imgs:
                    cv2.imwrite(f"eval_images/debug_head_00000.jpg", cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"eval_images/debug_wrist_00000.jpg", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                    print("\n[DEBUG] Saved debug_head_00000.jpg and debug_wrist_00000.jpg. PLEASE VERIFY THEY ARE CORRECT!")

                # Display images (disabled by default)
                if args.display:
                    view_dual_images(head_img, wrist_img, f"Chunk {chunk_idx}")

                # Get action from model
                start_time = time.time()
                action = inference.get_action(head_img, wrist_img, state)
                inference_time = time.time() - start_time

                # Diagnostic: Record timing and state/action data
                if diagnostic_data is not None:
                    diagnostic_data["timing"].append({
                        "chunk": chunk_idx,
                        "obs_time_ms": obs_time * 1000,
                        "inference_time_ms": inference_time * 1000,
                    })
                    diagnostic_data["states"].append({
                        "chunk": chunk_idx,
                        "state": state.tolist(),
                    })

                # Construct full action array for this chunk
                # Shape: (action_horizon, 6)
                current_action_chunk = np.zeros((args.action_horizon, 6))
                for i in range(args.action_horizon):
                    current_action_chunk[i] = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0
                    )

                # Temporal Ensembling (Exponential Moving Average)
                # Blend current prediction with previous prediction shifted by execution steps
                # Here we simplify: just smooth the transition if we have a previous chunk
                if prev_action_chunk is not None:
                    # Blend the first action of new chunk with expectation
                    # This is a simple low-pass filter on the trajectory start
                    alpha = 0.8 # Trust new observation 80%
                    current_action_chunk[0] = alpha * current_action_chunk[0] + (1-alpha) * prev_action_chunk[-1]
                
                prev_action_chunk = current_action_chunk

                # Diagnostic: Record predicted actions
                if diagnostic_data is not None:
                    diagnostic_data["actions"].append({
                        "chunk": chunk_idx,
                        "predicted_action_0": current_action_chunk[0].tolist(),
                        "predicted_action_full": current_action_chunk.tolist(),
                    })

                # Print debug info for first chunk or every 10th chunk
                if chunk_idx == 0 or chunk_idx % 10 == 0:
                    print(f"\n[DEBUG] Chunk {chunk_idx}")
                    print(f"  Task: '{inference.task}'")
                    print(f"  Current state (6 joints): {np.round(state, 2)}")
                    print(f"    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]")

                    # Check if current state is within training range
                    state_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    train_mins = [-48.6, -99.1, -84.5, 23.4, -51.3, 0.0]
                    train_maxs = [53.2, 64.0, 100.0, 92.4, 31.4, 29.3]

                    out_of_range = []
                    for i, (name, val, tmin, tmax) in enumerate(zip(state_names, state, train_mins, train_maxs)):
                        if val < tmin or val > tmax:
                            out_of_range.append(f"{name}={val:.1f} (range: [{tmin}, {tmax}])")

                    if out_of_range:
                        print(f"  ⚠️  OUT OF TRAINING RANGE: {out_of_range}")
                    else:
                        print(f"  ✓ All joints within training range")

                    # Show predicted actions
                    print(f"  Predicted Action [0]: {np.round(current_action_chunk[0], 2)}")
                    print(f"  Predicted Action [-1]: {np.round(current_action_chunk[-1], 2)}")

                    # Show action delta (change from current state)
                    delta = current_action_chunk[0] - state
                    print(f"  Action delta (pred - current): {np.round(delta, 2)}")

                # Execute action chunk
                exec_start = time.time()
                for action_idx in range(args.action_horizon):
                    # Use the (potentially ensembled) action
                    target_state = current_action_chunk[action_idx]
                    assert target_state.shape == (6,), f"Expected shape (6,), got {target_state.shape}"

                    # Send action to robot
                    robot.set_target_state(torch.from_numpy(target_state))
                    time.sleep(args.action_interval)

                    # Record image if enabled
                    if args.record_imgs:
                        head_img, _ = robot.get_dual_images()
                        img_bgr = cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
                        img_resized = cv2.resize(img_bgr, (320, 240))
                        cv2.imwrite(f"eval_images/img_{image_count:05d}.jpg", img_resized)
                        image_count += 1

                exec_time = time.time() - exec_start
                
                # Calculate control frequency
                total_loop_time = time.time() - last_inference_time
                control_freq = 1.0 / total_loop_time if total_loop_time > 0 else 0
                last_inference_time = time.time()

                if chunk_idx % 10 == 0:
                    print(f"Chunk {chunk_idx}: inference={inference_time:.3f}s, exec={exec_time:.3f}s, loop_freq={control_freq:.1f}Hz")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        # Return to home safely
        print("\nReturning to home position (training-aligned)...")
        robot.go_home(training_aligned=True)

    print("\nInference complete!")
    if args.record_imgs:
        print(f"Recorded {image_count} images to eval_images/")

    # Save diagnostic data
    if diagnostic_data is not None:
        # Calculate summary statistics
        timing_data = diagnostic_data["timing"]
        if timing_data:
            obs_times = [t["obs_time_ms"] for t in timing_data]
            inf_times = [t["inference_time_ms"] for t in timing_data]
            diagnostic_data["summary"] = {
                "total_chunks": len(timing_data),
                "obs_time_mean_ms": float(np.mean(obs_times)),
                "obs_time_max_ms": float(np.max(obs_times)),
                "inference_time_mean_ms": float(np.mean(inf_times)),
                "inference_time_max_ms": float(np.max(inf_times)),
            }

        camera_data = diagnostic_data["camera_metrics"]
        if camera_data:
            corrupt_count = sum(1 for c in camera_data if c["head"]["is_corrupt"] or c["wrist"]["is_corrupt"])
            diagnostic_data["summary"]["camera_corrupt_count"] = corrupt_count
            diagnostic_data["summary"]["camera_corruption_rate"] = corrupt_count / len(camera_data)

        # Save to JSON
        output_path = os.path.join(args.diagnostic_output, "diagnostic_results.json")
        with open(output_path, "w") as f:
            json.dump(diagnostic_data, f, indent=2)
        print(f"\n[DIAGNOSTIC] Results saved to: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        summary = diagnostic_data.get("summary", {})
        print(f"  Total chunks:           {summary.get('total_chunks', 0)}")
        print(f"  Obs time (mean):        {summary.get('obs_time_mean_ms', 0):.1f}ms")
        print(f"  Obs time (max):         {summary.get('obs_time_max_ms', 0):.1f}ms")
        print(f"  Inference time (mean):  {summary.get('inference_time_mean_ms', 0):.1f}ms")
        print(f"  Inference time (max):   {summary.get('inference_time_max_ms', 0):.1f}ms")
        print(f"  Camera corruptions:     {summary.get('camera_corrupt_count', 0)} ({summary.get('camera_corruption_rate', 0)*100:.1f}%)")
        print("="*60)


if __name__ == "__main__":
    main()
