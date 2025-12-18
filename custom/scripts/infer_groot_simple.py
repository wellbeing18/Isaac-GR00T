#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simple GR00T Inference Script for SO-101
# Based on NVIDIA official: examples/SO-100/eval_gr00t_so100.py
#
# Key differences from our async scripts:
# - NO threading/async complexity
# - Execute ALL action_horizon steps before new prediction
# - This matches NVIDIA's proven approach
#
# Usage:
#   python custom/scripts/infer_groot_simple.py \
#       --model-path /path/to/checkpoint \
#       --task "pick the red cube" \
#       --duration 60
#

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# LeRobot imports for robot control (v0.4.1 API)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError

# GR00T imports for inference
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "cfgs" / "so101_hardware.yaml"


def load_hardware_config(config_path: str = None) -> dict:
    """Load hardware configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[WARNING] Config file not found: {config_path}")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, script_name: str = "infer_groot_simple") -> logging.Logger:
    """Setup dual logging to both terminal and log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(f"Logging to: {log_file}")
    return logger


# Global logger
logger: logging.Logger = None


def get_logger() -> logging.Logger:
    global logger
    if logger is None:
        fallback = logging.getLogger("infer_groot_simple_fallback")
        if not fallback.handlers:
            fallback.addHandler(logging.StreamHandler(sys.stdout))
            fallback.setLevel(logging.INFO)
        return fallback
    return logger


#################################################################################
# LoRA Checkpoint Loading (reused from infer_groot_async.py)
#################################################################################


def is_lora_checkpoint(model_path: str) -> bool:
    """Detect if the given path is a LoRA (PEFT) checkpoint."""
    model_path = Path(model_path)
    return (model_path / "adapter_config.json").exists() and \
           (model_path / "adapter_model.safetensors").exists()


def get_lora_config(model_path: str) -> dict:
    """Load and return the PEFT/LoRA configuration."""
    with open(Path(model_path) / "adapter_config.json", "r") as f:
        return json.load(f)


def load_groot_with_lora(
    model_path: str,
    embodiment_tag: str,
    modality_config: dict,
    modality_transform,
    denoising_steps: int = 4,
    merge_weights: bool = True,
) -> Gr00tPolicy:
    """Load GR00T model with LoRA adapters merged."""
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.dataset import DatasetMetadata
    from gr00t.model.policy import EmbodimentTag

    model_path = Path(model_path)
    lora_config = get_lora_config(model_path)
    base_model_path = lora_config.get("base_model_name_or_path")

    get_logger().info(f"[LoRA] Detected LoRA checkpoint")
    get_logger().info(f"[LoRA] Base model: {base_model_path}")
    get_logger().info(f"[LoRA] LoRA rank: {lora_config.get('r')}")

    # Load base model
    get_logger().info(f"[LoRA] Step 1/3: Loading base GR00T model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
    base_model.eval()

    # Load PEFT adapter
    get_logger().info(f"[LoRA] Step 2/3: Loading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(model_path))

    # Merge weights
    if merge_weights:
        get_logger().info(f"[LoRA] Step 3/3: Merging LoRA weights...")
        merged_model = peft_model.merge_and_unload()
    else:
        merged_model = peft_model

    merged_model.to(device=device)

    # Create policy wrapper
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.device = device
    policy._modality_config = modality_config
    policy._modality_transform = modality_transform
    policy.model = merged_model
    policy.model.action_head.num_inference_timesteps = denoising_steps

    # Load normalization metadata and set transform metadata (CRITICAL!)
    exp_cfg_dir = model_path / "experiment_cfg"
    if exp_cfg_dir.exists():
        metadata_path = exp_cfg_dir / "metadata.json"
        if metadata_path.exists():
            get_logger().info(f"[LoRA] Loading normalization metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                metadatas = json.load(f)

            # Get embodiment tag enum
            if isinstance(embodiment_tag, str):
                embodiment_tag_enum = EmbodimentTag(embodiment_tag)
            else:
                embodiment_tag_enum = embodiment_tag

            policy.embodiment_tag = embodiment_tag_enum

            metadata_dict = metadatas.get(embodiment_tag_enum.value)
            if metadata_dict:
                metadata = DatasetMetadata.model_validate(metadata_dict)
                # CRITICAL: Set metadata on transform to enable normalization
                policy._modality_transform.set_metadata(metadata)
                policy.metadata = metadata
                get_logger().info(f"[LoRA] Loaded normalization stats for '{embodiment_tag_enum.value}'")
            else:
                get_logger().warning(f"[LoRA] WARNING: No metadata found for embodiment '{embodiment_tag_enum.value}'")

    # Load horizons from modality config
    policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
    policy._video_horizon = len(policy._video_delta_indices)
    if "state" in modality_config:
        policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
        policy._state_horizon = len(policy._state_delta_indices)
    policy._action_delta_indices = np.array(modality_config["action"].delta_indices)

    get_logger().info(f"[LoRA] GR00T model with LoRA loaded successfully!")
    get_logger().info(f"[LoRA] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return policy


#################################################################################
# Robot Interface (reused from infer_groot_async.py)
#################################################################################


class So101RobotInterface:
    """SO101 robot interface with dual cameras."""

    # Training-aligned home position
    HOME_POSITION_TRAINING = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -99.0,
        "elbow_flex.pos": 100.0,
        "wrist_flex.pos": 50.0,
        "wrist_roll.pos": -1.0,
        "gripper.pos": 0.5,
    }

    def __init__(
        self,
        serial_port: str = "/dev/ttyACM2",
        robot_id: str = "xlerobot_left_arm",
        head_cam_idx: int = 4,
        wrist_cam_idx: int = 6,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        fourcc: str = "MJPG",
        use_degrees: bool = True,
    ):
        self.robot_id = robot_id
        self.cam_width = width
        self.cam_height = height

        # Camera config
        cameras_cfg = {
            "head": OpenCVCameraConfig(
                index_or_path=head_cam_idx,
                fps=fps,
                width=width,
                height=height,
                fourcc=fourcc,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_idx,
                fps=fps,
                width=width,
                height=height,
                fourcc=fourcc,
            ),
        }

        # Robot config
        robot_cfg = SO101FollowerConfig(
            port=serial_port,
            id=robot_id,
            cameras=cameras_cfg,
            use_degrees=use_degrees,
        )

        self.robot = SO101Follower(robot_cfg)
        get_logger().info(f"SO101 Robot interface created (port={serial_port}, id={robot_id})")

    @contextmanager
    def activate(self):
        """Context manager for robot activation."""
        try:
            if not self.robot.is_connected:
                self.robot.connect()
                get_logger().info("================> SO101 Robot connected (dual cameras)")
        except DeviceAlreadyConnectedError:
            get_logger().info("Robot already connected")

        try:
            yield
        finally:
            self.robot.disconnect()
            get_logger().info("================> SO101 Robot disconnected")

    def go_home(self, training_aligned: bool = True):
        """Move robot to home position."""
        get_logger().info("-------------------------------- Moving to home pose")
        if training_aligned:
            self.robot.send_action(self.HOME_POSITION_TRAINING)
            get_logger().info("  Using training-aligned home")
        time.sleep(2)

    def get_current_state(self) -> np.ndarray:
        """Get current robot state as (6,) array."""
        obs = self.robot.get_observation()
        return np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ])

    def get_observation(self) -> dict:
        """Get full observation dict with cameras and state."""
        obs = self.robot.get_observation()
        state = np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ])
        return {
            "head": obs["head"],
            "wrist": obs["wrist"],
            "state": state,
        }

    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        action_dict = {
            "shoulder_pan.pos": float(action[0]),
            "shoulder_lift.pos": float(action[1]),
            "elbow_flex.pos": float(action[2]),
            "wrist_flex.pos": float(action[3]),
            "wrist_roll.pos": float(action[4]),
            "gripper.pos": float(action[5]),
        }
        self.robot.send_action(action_dict)


#################################################################################
# Model Wrapper
#################################################################################


class Gr00tLocalInference:
    """GR00T model loading and inference wrapper."""

    def __init__(
        self,
        model_path: str,
        data_config: str = "so100_dualcam",
        embodiment_tag: str = "new_embodiment",
        task: str = "pick the red cube from the table",
        denoising_steps: int = 4,
    ):
        self.task = task
        self.modality_keys = ["single_arm", "gripper"]

        get_logger().info(f"Loading GR00T model from: {model_path}")

        data_cfg = load_data_config(data_config)
        modality_config = data_cfg.modality_config()
        modality_transform = data_cfg.transform()

        if is_lora_checkpoint(model_path):
            get_logger().info(f"[INFO] Detected LoRA checkpoint")
            self.policy = load_groot_with_lora(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
                merge_weights=True,
            )
        else:
            get_logger().info(f"[INFO] Full checkpoint detected")
            self.policy = Gr00tPolicy(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
            )

        get_logger().info(f"Model loaded! Denoising steps: {denoising_steps}")

    def get_action(self, front_img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray) -> dict:
        """Run inference and return action dictionary."""
        obs_dict = {
            "video.front": front_img[np.newaxis, :, :, :],
            "video.wrist": wrist_img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.task],
        }
        return self.policy.get_action(obs_dict)

    def extract_action_chunk(self, action_dict: dict, horizon: int = 16) -> np.ndarray:
        """Extract action chunk as numpy array of shape (horizon, 6)."""
        actions = np.zeros((horizon, 6))
        for i in range(horizon):
            actions[i] = np.concatenate(
                [np.atleast_1d(action_dict[f"action.{key}"][i]) for key in self.modality_keys],
                axis=0,
            )
        return actions


#################################################################################
# Main Inference Loop - SIMPLE NVIDIA-STYLE
#################################################################################


def main():
    global logger

    # Load hardware config
    hw_config = load_hardware_config()
    robot_cfg = hw_config.get("robot_arms", {}).get("left", {})
    cam_cfg = hw_config.get("cameras", {})
    log_cfg = hw_config.get("logging", {})

    default_port = robot_cfg.get("port", "/dev/ttyACM0")
    default_head_cam = cam_cfg.get("head", {}).get("device_index", 8)
    default_wrist_cam = cam_cfg.get("wrist", {}).get("device_index", 4)
    default_log_dir = log_cfg.get("log_dir", "/home/jrobot/project/Isaac-GR00T/custom/logs")

    parser = argparse.ArgumentParser(
        description="Simple GR00T Inference Script - NVIDIA-style synchronous loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True, help="Path to the finetuned GR00T checkpoint")
    parser.add_argument("--data-config", type=str, default="so100_dualcam", help="Data configuration name")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment", help="Embodiment tag")
    parser.add_argument("--denoising-steps", type=int, default=4, help="Number of denoising steps")
    parser.add_argument("--task", type=str, default="pick the red cube from the table", help="Task description")

    # Robot configuration
    parser.add_argument("--port", type=str, default=default_port, help="Robot serial port")
    parser.add_argument("--head-cam-idx", type=int, default=default_head_cam, help="Head camera index")
    parser.add_argument("--wrist-cam-idx", type=int, default=default_wrist_cam, help="Wrist camera index")

    # Execution configuration
    parser.add_argument("--action-horizon", type=int, default=8, help="Number of actions to execute per prediction (NVIDIA default: 8-12)")
    parser.add_argument("--action-interval", type=float, default=0.02, help="Time between actions in seconds (NVIDIA: 0.02 = 50Hz)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration to run in seconds")
    parser.add_argument("--go-home-first", action="store_true", help="Move to home position before inference")

    # Output
    parser.add_argument("--log-dir", type=str, default=default_log_dir, help="Directory for log files")
    parser.add_argument("--record-imgs", action="store_true", help="Record images to eval_images/")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir, "infer_groot_simple")

    logger.info("=" * 70)
    logger.info("GR00T SIMPLE Inference - NVIDIA-style synchronous loop")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Action horizon: {args.action_horizon} (execute this many actions per prediction)")
    logger.info(f"Action interval: {args.action_interval}s ({1/args.action_interval:.0f}Hz)")
    logger.info(f"Duration: {args.duration}s")
    logger.info("=" * 70)

    # Setup image recording
    if args.record_imgs:
        os.makedirs("eval_images", exist_ok=True)
        for f in os.listdir("eval_images"):
            if f.endswith(('.jpg', '.png')):
                os.remove(os.path.join("eval_images", f))
        logger.info("[RECORD] Image recording enabled - saving to eval_images/")

    # Initialize robot
    logger.info("\n[INIT] Creating robot interface...")
    robot = So101RobotInterface(
        serial_port=args.port,
        head_cam_idx=args.head_cam_idx,
        wrist_cam_idx=args.wrist_cam_idx,
    )

    # Initialize model
    logger.info("\n[INIT] Loading GR00T model...")
    model = Gr00tLocalInference(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        task=args.task,
        denoising_steps=args.denoising_steps,
    )

    # Warmup model
    logger.info("\n[WARMUP] Running warmup inference...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_state = np.zeros(6)
    for _ in range(3):
        model.get_action(dummy_img, dummy_img, dummy_state)
    logger.info("[WARMUP] Complete!")

    # Statistics
    inference_count = 0
    action_count = 0
    total_inference_time = 0.0
    image_count = 0

    # Run inference
    with robot.activate():
        if args.go_home_first:
            logger.info("\n[RESET] Moving to training-aligned home position...")
            robot.go_home(training_aligned=True)
            time.sleep(1.0)
            state = robot.get_current_state()
            logger.info(f"  Home state: {np.round(state, 2)}")

        logger.info(f"\n[RUN] Starting SIMPLE inference for {args.duration}s...")
        logger.info(f"      Loop: get_obs -> inference -> execute {args.action_horizon} actions -> repeat")
        logger.info("      Press Ctrl+C to stop early\n")

        start_time = time.time()

        try:
            while time.time() - start_time < args.duration:
                loop_start = time.time()

                # 1. Get observation
                obs = robot.get_observation()
                head_img = obs["head"]
                wrist_img = obs["wrist"]
                state = obs["state"]

                # Record images if enabled
                if args.record_imgs:
                    head_bgr = cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
                    wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"eval_images/head_{image_count:05d}.jpg", head_bgr)
                    cv2.imwrite(f"eval_images/wrist_{image_count:05d}.jpg", wrist_bgr)
                    image_count += 1

                # 2. Run inference
                inference_start = time.time()
                action_dict = model.get_action(head_img, wrist_img, state)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Extract action chunk
                action_chunk = model.extract_action_chunk(action_dict, horizon=16)
                inference_count += 1

                # Log prediction details
                if inference_count <= 10:  # Log first 10 predictions in detail
                    logger.info(f"\n[INFERENCE #{inference_count}]")
                    logger.info(f"  Current state: {np.round(state, 1)}")
                    logger.info(f"  action[0]:     {np.round(action_chunk[0], 1)} (delta: {np.round(action_chunk[0] - state, 2)})")
                    logger.info(f"  action[{args.action_horizon-1}]:     {np.round(action_chunk[args.action_horizon-1], 1)} (delta: {np.round(action_chunk[args.action_horizon-1] - state, 2)})")
                    logger.info(f"  action[15]:    {np.round(action_chunk[15], 1)} (delta: {np.round(action_chunk[15] - state, 2)})")
                    logger.info(f"  Inference time: {inference_time*1000:.0f}ms")

                # 3. Execute ALL action_horizon steps (NVIDIA approach!)
                for i in range(args.action_horizon):
                    action = action_chunk[i]
                    robot.send_action(action)
                    action_count += 1
                    time.sleep(args.action_interval)

                    # Log first few action executions
                    if inference_count <= 3 and i < 3:
                        current = robot.get_current_state()
                        logger.info(f"  [EXEC step {i}] target={np.round(action, 1)}, actual={np.round(current, 1)}")

                # Progress logging
                elapsed = time.time() - start_time
                if inference_count % 10 == 0:
                    avg_inf_time = total_inference_time / inference_count * 1000
                    logger.info(f"[PROGRESS] t={elapsed:.0f}s | inferences={inference_count} | actions={action_count} | avg_inf={avg_inf_time:.0f}ms")

        except KeyboardInterrupt:
            logger.info("\n\n[STOP] Interrupted by user")

        # Return home
        logger.info("\n[RESET] Returning to home position...")
        robot.go_home(training_aligned=True)

    # Final statistics
    elapsed = time.time() - start_time
    avg_inference_time = total_inference_time / max(1, inference_count) * 1000

    logger.info("\n" + "=" * 70)
    logger.info("SIMPLE INFERENCE RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Duration:           {elapsed:.1f}s")
    logger.info(f"  Inferences:         {inference_count}")
    logger.info(f"  Actions executed:   {action_count}")
    logger.info(f"  Inference rate:     {inference_count/elapsed:.1f} Hz")
    logger.info(f"  Action rate:        {action_count/elapsed:.1f} Hz")
    logger.info(f"  Avg inference time: {avg_inference_time:.0f}ms")
    logger.info(f"  Actions per inf:    {action_count/max(1,inference_count):.1f} (target: {args.action_horizon})")
    logger.info("=" * 70)

    if args.record_imgs:
        logger.info(f"\n[RECORD] Saved {image_count} image pairs to eval_images/")


if __name__ == "__main__":
    main()
