#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SO101 GR00T Async Inference Script
# Producer-Consumer architecture to eliminate stop-go jerkiness
#
# Architecture:
#   Producer Thread (Vision/Model): Captures images, runs inference at ~7Hz, pushes to queue
#   Consumer Thread (Control): Runs at 30Hz, interpolates between predictions, sends to robot
#
# This eliminates the 27% "dead time" where the robot stops during inference.
# Even at 7Hz inference rate, the robot moves smoothly via interpolation.
#
# Reference: custom/jdocs/lora/3_inference_issue_investigation_20251206.md
#
# Usage:
#   python infer_groot_async.py --model-path /path/to/checkpoint
#   python infer_groot_async.py --help
#

import argparse
import json
import logging
import os
import sys
import time
import threading
import queue
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml


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
        config = yaml.safe_load(f)

    return config


def setup_logging(log_dir: str, script_name: str = "infer_groot_async") -> logging.Logger:
    """Setup dual logging to both terminal and log file."""
    # Create log directory if needed
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent duplicate output from root logger
    logger.propagate = False

    logger.info(f"Logging to: {log_file}")

    return logger

# LeRobot imports for robot control (v0.4.1 API)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError

# GR00T imports for inference
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

from tqdm import tqdm


# Module-level logger (initialized in main())
logger: logging.Logger = None


def get_logger() -> logging.Logger:
    """Get the module logger, with fallback to print if not initialized."""
    global logger
    if logger is None:
        # Fallback logger that just prints (before main() runs)
        fallback = logging.getLogger("infer_groot_async_fallback")
        if not fallback.handlers:
            fallback.addHandler(logging.StreamHandler(sys.stdout))
            fallback.setLevel(logging.INFO)
        return fallback
    return logger


#################################################################################
# Data Classes for Thread Communication
#################################################################################


@dataclass
class ActionPrediction:
    """Container for action predictions passed between threads."""
    timestamp: float  # When the observation was captured
    actions: np.ndarray  # Shape: (horizon, 6)
    state_at_capture: np.ndarray  # Robot state when observation was captured
    inference_time_ms: float  # How long inference took
    chunk_idx: int  # For debugging


@dataclass
class AsyncStats:
    """Statistics for async performance monitoring."""
    producer_count: int = 0
    consumer_count: int = 0
    stale_count: int = 0  # Actions executed with old predictions
    interpolation_count: int = 0
    ensemble_count: int = 0  # Actions that used multiple predictions
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.consumer_count)


class TemporalEnsembleBuffer:
    """
    Sliding Window Temporal Ensembling Buffer.

    Maintains overlapping predictions and averages them for smoother actions.
    This is the standard approach used in ACT and Diffusion Policy.

    How it works:
    - Each inference produces 16 actions (horizon)
    - New inferences arrive every ~8 execution steps (with overlap)
    - For each execution step, we average ALL predictions that cover it
    - This reduces variance by sqrt(N) where N = number of overlapping predictions

    Example timeline:
        Step 0:  Inf1 predicts [0-15]
        Step 8:  Inf2 predicts [8-23]  -> Steps 8-15 have 2 predictions
        Step 16: Inf3 predicts [16-31] -> Steps 16-23 have 2 predictions

        Action at step 10 = average(Inf1[10], Inf2[2])
    """

    def __init__(self, action_dim: int = 6, max_predictions: int = 4, use_first_n_actions: int = None):
        """
        Args:
            action_dim: Dimension of action space
            max_predictions: Maximum overlapping predictions to keep
            use_first_n_actions: Only use first N actions from each prediction.
                                 If None, uses full horizon. This helps when the model
                                 predicts trajectories that oscillate back to origin.
        """
        self.action_dim = action_dim
        self.max_predictions = max_predictions
        self.use_first_n_actions = use_first_n_actions  # Truncate horizon if set
        self.predictions = []  # List of dicts with 'start_step', 'actions', 'timestamp'
        self.execution_step = 0  # Global execution step counter
        self.lock = threading.Lock()

    def add_prediction(self, actions: np.ndarray, timestamp: float):
        """
        Add a new prediction to the buffer.

        Args:
            actions: Shape (horizon, action_dim) - predicted action sequence
            timestamp: When the observation was captured
        """
        with self.lock:
            # Optionally truncate to first N actions (avoids oscillating trajectories)
            if self.use_first_n_actions is not None:
                actions = actions[:self.use_first_n_actions]

            self.predictions.append({
                'start_step': self.execution_step,
                'actions': actions.copy(),
                'timestamp': timestamp,
                'horizon': len(actions),
            })

            # Remove old predictions that no longer overlap
            self._cleanup()

            # Keep only most recent predictions
            if len(self.predictions) > self.max_predictions:
                self.predictions = self.predictions[-self.max_predictions:]

    def get_ensembled_action(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Get averaged action for current execution step.

        Returns:
            Tuple of (action, num_predictions_averaged)
            Returns (None, 0) if no predictions available
        """
        with self.lock:
            if not self.predictions:
                return None, 0

            # Collect all predictions that cover current step
            valid_actions = []
            for pred in self.predictions:
                local_idx = self.execution_step - pred['start_step']
                if 0 <= local_idx < pred['horizon']:
                    valid_actions.append(pred['actions'][local_idx])

            if not valid_actions:
                return None, 0

            # Average all valid predictions (temporal ensembling!)
            ensembled = np.mean(valid_actions, axis=0)
            return ensembled, len(valid_actions)

    def advance_step(self):
        """Move to next execution step and cleanup old predictions."""
        with self.lock:
            self.execution_step += 1
            self._cleanup()

    def _cleanup(self):
        """Remove predictions that no longer cover current step."""
        self.predictions = [
            p for p in self.predictions
            if self.execution_step - p['start_step'] < p['horizon']
        ]

    def get_latest_timestamp(self) -> Optional[float]:
        """Get timestamp of most recent prediction."""
        with self.lock:
            if self.predictions:
                return self.predictions[-1]['timestamp']
            return None

    def num_active_predictions(self) -> int:
        """Get number of predictions currently in buffer."""
        with self.lock:
            return len(self.predictions)

    def reset(self):
        """Clear buffer and reset step counter."""
        with self.lock:
            self.predictions = []
            self.execution_step = 0


#################################################################################
# Camera Quality Check (from infer_groot_so101.py)
#################################################################################


def check_camera_quality(frame: np.ndarray, black_threshold: int = 10) -> dict:
    """Check camera frame quality for corruption detection."""
    if frame is None or frame.size == 0:
        return {"is_corrupt": True, "corruption_score": 1.0}

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] == 3 else frame[:, :, 0]
    else:
        gray = frame

    row_means = np.mean(gray, axis=1)
    black_rows = int(np.sum(row_means < black_threshold))
    corruption_score = black_rows / gray.shape[0]

    return {
        "is_corrupt": corruption_score > 0.05,
        "corruption_score": float(corruption_score),
    }


#################################################################################
# LoRA Checkpoint Detection and Loading (from infer_groot_so101.py)
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
# Robot Interface (from infer_groot_so101.py)
#################################################################################


class So101RobotInterface:
    """SO101 robot interface with dual cameras (matches infer_groot_so101.py)."""

    # Training-aligned home position - matches episode START position
    # Episodes START and END at this "ready" position before picking
    # Based on analysis: episodes start at [4.6, -99.3, 100, 50, -1.8, 0.5]
    HOME_POSITION_TRAINING = {
        "shoulder_pan.pos": 0.0,      # episode start: ~0-5°
        "shoulder_lift.pos": -99.0,   # episode start: -99°
        "elbow_flex.pos": 100.0,      # episode start: 100°
        "wrist_flex.pos": 50.0,       # episode start: 50°
        "wrist_roll.pos": -1.0,       # episode start: -1.8°
        "gripper.pos": 0.5,           # episode start: 0.5°
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

        # Camera config (lerobot v0.4.1 format uses index_or_path)
        # Using MJPG (compressed) reduces USB bandwidth: ~3 MB/s vs ~28 MB/s raw per camera
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

        # Robot config (matching infer_groot_so101.py)
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
            get_logger().info("  Using training-aligned home (within training range)")
        else:
            # Default home position (all zeros, gripper open)
            home_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 50.0,
            }
            self.robot.send_action(home_action)
        time.sleep(2)  # Match working script's sleep time

    def get_current_state(self) -> np.ndarray:
        """Get current robot state as (6,) array."""
        obs = self.robot.get_observation()
        # Keys match lerobot v0.4.1 format (direct access, not nested)
        return np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ])

    def get_observation(self) -> dict:
        """Get full observation dict with cameras and state in single call."""
        obs = self.robot.get_observation()
        # Keys match lerobot v0.4.1 format (direct access, not nested)
        # Extract state directly to avoid double robot query
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

    def set_target_state(self, target_state: np.ndarray):
        """Send action to robot."""
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


#################################################################################
# GR00T Model Wrapper (from infer_groot_so101.py)
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
        """Run inference and return action dictionary.

        NOTE: We set a fixed random seed before inference to ensure consistent
        outputs for the same input. The Flow Matching action head uses torch.randn()
        to sample initial noise, which causes different outputs on each call.
        See investigation: custom/jdocs/lora/investigations/6_claude_inference_issue_investigation.md
        """
        obs_dict = {
            "video.front": front_img[np.newaxis, :, :, :],
            "video.wrist": wrist_img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.task],
        }
        # Set fixed seed for deterministic output (prevents arm vibration)
        torch.manual_seed(42)
        return self.policy.get_action(obs_dict)


#################################################################################
# Async Inference Engine
#################################################################################


class AsyncInferenceEngine:
    """
    Producer-Consumer async inference engine with Temporal Ensembling.

    Producer thread: Captures observations and runs model inference (~5-7Hz)
    Consumer thread: Executes actions at 30Hz using ensembled predictions

    Features:
    - Eliminates 27% dead time from blocking inference
    - Temporal ensembling averages overlapping predictions to reduce variance
    - This is the standard approach used in ACT and Diffusion Policy
    """

    def __init__(
        self,
        robot: So101RobotInterface,
        inference: Gr00tLocalInference,
        action_horizon: int = 16,
        action_interval: float = 0.033,  # 30Hz
        queue_size: int = 2,
        record_imgs: bool = False,
        use_first_n_actions: int = None,  # Truncate trajectory to first N actions
        no_ensemble: bool = False,  # Disable temporal ensembling, use only latest prediction
    ):
        self.robot = robot
        self.inference = inference
        self.action_horizon = action_horizon
        self.action_interval = action_interval
        self.record_imgs = record_imgs
        self.use_first_n_actions = use_first_n_actions
        self.no_ensemble = no_ensemble

        # Thread-safe queue for passing predictions
        self.prediction_queue = queue.Queue(maxsize=queue_size)

        # Shared state
        self.running = False
        self.stats = AsyncStats()
        self.lock = threading.Lock()

        # Robot serial port lock - CRITICAL for thread safety
        # Both producer (read state) and consumer (write action) access the robot
        self.robot_lock = threading.Lock()

        # Temporal Ensembling Buffer - averages overlapping predictions
        # This reduces action variance by sqrt(N) where N = number of overlapping predictions
        # use_first_n_actions: Only use first N actions from each prediction (avoids oscillation)
        # no_ensemble: If True, only keep the latest prediction (max_predictions=1)
        self.ensemble_buffer = TemporalEnsembleBuffer(
            action_dim=6,
            max_predictions=1 if no_ensemble else 4,
            use_first_n_actions=use_first_n_actions
        )

        if use_first_n_actions:
            get_logger().info(f"[ASYNC] Using first {use_first_n_actions} actions from each prediction (truncated horizon)")

        if no_ensemble:
            get_logger().info(f"[ASYNC] Temporal ensembling DISABLED - using only latest prediction")

        # Legacy: Keep for fallback if needed
        self.current_prediction: Optional[ActionPrediction] = None
        self.previous_prediction: Optional[ActionPrediction] = None

        # For graceful shutdown
        self.producer_thread: Optional[threading.Thread] = None
        self.consumer_thread: Optional[threading.Thread] = None

        # Modality keys
        self.MODALITY_KEYS = ["single_arm", "gripper"]

    def start(self):
        """Start producer and consumer threads."""
        self.running = True

        self.producer_thread = threading.Thread(target=self._producer_loop, name="Producer")
        self.consumer_thread = threading.Thread(target=self._consumer_loop, name="Consumer")

        self.producer_thread.start()
        self.consumer_thread.start()

        get_logger().info("[ASYNC] Producer and Consumer threads started")
        get_logger().info("[ASYNC] Fixed seed=42 enabled for deterministic model outputs (prevents vibration)")

    def stop(self):
        """Stop all threads gracefully."""
        get_logger().info("[ASYNC] Stopping threads...")
        self.running = False

        if self.producer_thread:
            self.producer_thread.join(timeout=2.0)
        if self.consumer_thread:
            self.consumer_thread.join(timeout=2.0)

        get_logger().info("[ASYNC] Threads stopped")

    def _producer_loop(self):
        """
        Producer thread: Capture observations and run inference.
        Runs as fast as the GPU allows (~7Hz based on diagnosis).
        """
        chunk_idx = 0

        while self.running:
            try:
                # Capture observation (with robot lock to avoid serial port collision)
                capture_start = time.time()
                with self.robot_lock:
                    obs = self.robot.get_observation()
                head_img = obs["head"]
                wrist_img = obs["wrist"]
                state = obs["state"]
                capture_time = time.time()

                # Save images if recording is enabled
                if self.record_imgs:
                    # Save both head and wrist images with chunk index
                    head_bgr = cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
                    wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"eval_images/head_{chunk_idx:05d}.jpg", head_bgr)
                    cv2.imwrite(f"eval_images/wrist_{chunk_idx:05d}.jpg", wrist_bgr)

                # Run inference
                inference_start = time.time()
                action_dict = self.inference.get_action(head_img, wrist_img, state)
                inference_time_ms = (time.time() - inference_start) * 1000

                # Extract action array
                actions = np.zeros((self.action_horizon, 6))
                for i in range(self.action_horizon):
                    actions[i] = np.concatenate(
                        [np.atleast_1d(action_dict[f"action.{key}"][i])
                         for key in self.MODALITY_KEYS],
                        axis=0
                    )

                # Create prediction object
                prediction = ActionPrediction(
                    timestamp=capture_time,
                    actions=actions,
                    state_at_capture=state.copy(),
                    inference_time_ms=inference_time_ms,
                    chunk_idx=chunk_idx,
                )

                # Log first 10 predictions for debugging consistency
                if chunk_idx < 10:
                    delta_0 = actions[0] - state
                    delta_15 = actions[15] - state
                    trajectory_delta = actions[15] - actions[0]  # Full horizon movement
                    get_logger().info(f"[DEBUG] Chunk {chunk_idx}: state={np.round(state, 1)}")
                    get_logger().info(f"[DEBUG]   action[0]={np.round(actions[0], 1)} (delta={np.round(delta_0, 2)})")
                    get_logger().info(f"[DEBUG]   action[15]={np.round(actions[15], 1)} (delta={np.round(delta_15, 2)})")
                    get_logger().info(f"[DEBUG]   horizon_trajectory={np.round(trajectory_delta, 2)} (16-step movement)")
                    # Log action consistency check - with fixed seed, consecutive chunks should be similar
                    if chunk_idx > 0:
                        get_logger().info(f"[DEBUG]   (fixed seed enabled - actions should be consistent for similar inputs)")

                # Try to put in queue (non-blocking to avoid deadlock)
                try:
                    # If queue is full, drop oldest prediction
                    if self.prediction_queue.full():
                        try:
                            self.prediction_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.prediction_queue.put_nowait(prediction)

                    with self.lock:
                        self.stats.producer_count += 1

                except queue.Full:
                    pass  # Consumer is too slow, skip this prediction

                chunk_idx += 1

                # Debug output every 10 chunks
                if chunk_idx % 10 == 0:
                    get_logger().info(f"[PRODUCER] Chunk {chunk_idx}: inference={inference_time_ms:.0f}ms, "
                                      f"rate={1000/inference_time_ms:.1f}Hz")

            except Exception as e:
                get_logger().error(f"[PRODUCER] Error: {e}")
                time.sleep(0.1)

    def _consumer_loop(self):
        """
        Consumer thread: Execute actions at 30Hz with Temporal Ensembling.

        Temporal Ensembling Strategy (ACT/Diffusion Policy standard):
        - Maintain buffer of overlapping predictions
        - For each timestep, AVERAGE all predictions that cover it
        - This reduces variance by sqrt(N) where N = number of overlapping predictions

        Example: At step 10 with 2 overlapping predictions:
          Action = 0.5 * Pred1[10] + 0.5 * Pred2[2]

        This is much more robust than the old "weak interpolation" which only
        blended the first 3 actions on transition.
        """
        last_action = None  # Fallback if no prediction available

        while self.running:
            try:
                loop_start = time.time()

                # Check for new prediction (non-blocking)
                try:
                    new_prediction = self.prediction_queue.get_nowait()

                    # Add to ensemble buffer (NOT replacing - accumulating!)
                    self.ensemble_buffer.add_prediction(
                        actions=new_prediction.actions,
                        timestamp=new_prediction.timestamp,
                    )

                    # Calculate latency (time from capture to now)
                    with self.lock:
                        latency_ms = (time.time() - new_prediction.timestamp) * 1000
                        self.stats.total_latency_ms += latency_ms
                        self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)

                    # Legacy: keep current_prediction for compatibility
                    self.current_prediction = new_prediction

                except queue.Empty:
                    # No new prediction - continue with existing buffer
                    pass

                # Get ensembled action from buffer
                ensembled_action, num_preds = self.ensemble_buffer.get_ensembled_action()

                if ensembled_action is not None:
                    # Track ensemble statistics
                    with self.lock:
                        if num_preds > 1:
                            self.stats.ensemble_count += 1
                        self.stats.consumer_count += 1

                        # Log first 20 consumer actions for debugging
                        should_log = self.stats.consumer_count <= 20
                        if should_log:
                            get_logger().info(f"[CONSUMER] Action {self.stats.consumer_count}: target={np.round(ensembled_action, 1)} (from {num_preds} preds)")

                    # Send action to robot (with robot lock to avoid serial port collision)
                    with self.robot_lock:
                        self.robot.set_target_state(ensembled_action)
                        # Read back state for debugging (first 20 actions only)
                        if should_log:
                            actual_state = self.robot.get_current_state()
                            delta = ensembled_action - actual_state
                            get_logger().info(f"[CONSUMER]   actual={np.round(actual_state, 1)} (error={np.round(delta, 2)})")

                    # Save for fallback
                    last_action = ensembled_action.copy()

                    # Advance to next step in buffer
                    self.ensemble_buffer.advance_step()

                elif last_action is not None:
                    # No ensembled action available - use last known action (stale)
                    with self.lock:
                        self.stats.stale_count += 1
                        self.stats.consumer_count += 1

                    with self.robot_lock:
                        self.robot.set_target_state(last_action)

                # Maintain 30Hz timing
                elapsed = time.time() - loop_start
                sleep_time = self.action_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                get_logger().error(f"[CONSUMER] Error: {e}")
                time.sleep(self.action_interval)

    def get_stats(self) -> AsyncStats:
        """Get current statistics."""
        with self.lock:
            return AsyncStats(
                producer_count=self.stats.producer_count,
                consumer_count=self.stats.consumer_count,
                stale_count=self.stats.stale_count,
                interpolation_count=self.stats.interpolation_count,
                ensemble_count=self.stats.ensemble_count,
                total_latency_ms=self.stats.total_latency_ms,
                max_latency_ms=self.stats.max_latency_ms,
            )


#################################################################################
# Main
#################################################################################


def main():
    # Load hardware config first to use as defaults
    hw_config = load_hardware_config()

    # Extract defaults from config (with fallbacks)
    robot_cfg = hw_config.get("robot_arms", {}).get("left", {})
    cam_cfg = hw_config.get("cameras", {})
    infer_cfg = hw_config.get("inference", {})
    log_cfg = hw_config.get("logging", {})

    default_port = robot_cfg.get("port", "/dev/ttyACM0")
    default_head_cam = cam_cfg.get("head", {}).get("device_index", 8)
    default_wrist_cam = cam_cfg.get("wrist", {}).get("device_index", 4)
    default_cam_fps = cam_cfg.get("head", {}).get("fps", 30)  # 30 FPS with MJPG compression
    # Resolution - GR00T model requires 640x480, cannot reduce
    head_res = cam_cfg.get("head", {}).get("resolution", {})
    default_cam_width = head_res.get("width", 640)
    default_cam_height = head_res.get("height", 480)
    default_data_config = infer_cfg.get("data_config", "so100_dualcam")
    default_embodiment = infer_cfg.get("embodiment_tag", "new_embodiment")
    default_denoising = infer_cfg.get("denoising_steps", 4)
    default_horizon = infer_cfg.get("action_horizon", 16)
    default_interval = infer_cfg.get("action_interval", 0.033)
    default_task = infer_cfg.get("default_task", "pick the red cube from the table")
    default_log_dir = log_cfg.get("log_dir", "/home/jrobot/project/Isaac-GR00T/custom/logs")

    parser = argparse.ArgumentParser(
        description="SO101 GR00T Async Inference - Producer-Consumer architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Hardware config loaded from: {DEFAULT_CONFIG_PATH}
Edit this file to change default ports/cameras without command-line args.
        """
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Hardware config YAML file (default: {DEFAULT_CONFIG_PATH})",
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the finetuned GR00T checkpoint",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=default_data_config,
        help=f"Data configuration name (default from config: {default_data_config})",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default=default_embodiment,
        help=f"Embodiment tag for the model (default from config: {default_embodiment})",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=default_denoising,
        help=f"Number of denoising steps (default from config: {default_denoising})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=default_task,
        help="Task description for the model",
    )

    # Robot configuration (defaults from YAML config)
    parser.add_argument(
        "--port",
        type=str,
        default=default_port,
        help=f"Robot serial port (default from config: {default_port})",
    )
    parser.add_argument(
        "--head-cam-idx",
        type=int,
        default=default_head_cam,
        help=f"Head/central camera index (default from config: {default_head_cam})",
    )
    parser.add_argument(
        "--wrist-cam-idx",
        type=int,
        default=default_wrist_cam,
        help=f"Wrist/left-arm camera index (default from config: {default_wrist_cam})",
    )
    parser.add_argument(
        "--cam-fps",
        type=int,
        default=default_cam_fps,
        help=f"Camera FPS (default from config: {default_cam_fps})",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=default_cam_width,
        help=f"Camera width - GR00T requires 640 (default: {default_cam_width})",
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=default_cam_height,
        help=f"Camera height - GR00T requires 480 (default: {default_cam_height})",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="Camera FOURCC format: MJPG (compressed, ~3MB/s) or YUYV (raw, ~28MB/s). Default: MJPG",
    )

    # Execution configuration
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=default_horizon,
        help=f"Number of actions to predict (default from config: {default_horizon})",
    )
    parser.add_argument(
        "--action-interval",
        type=float,
        default=default_interval,
        help=f"Time between actions in seconds (default from config: {default_interval})",
    )
    parser.add_argument(
        "--use-first-n-actions",
        type=int,
        default=None,
        help="Only use first N actions from each prediction. Helps when model predicts "
             "oscillating trajectories (e.g., --use-first-n-actions 4 uses only first 4 of 16 actions)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable temporal ensembling - use only the latest prediction. "
             "Helps when ensembling predictions from different states causes oscillation.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration to run in seconds (default: 60)",
    )
    parser.add_argument(
        "--go-home-first",
        action="store_true",
        help="Move to training-aligned home position before inference",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="async_inference_results.json",
        help="Output file for statistics",
    )
    parser.add_argument(
        "--record-imgs",
        action="store_true",
        help="Record inference images to eval_images/ folder for verification",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=default_log_dir,
        help="Directory for log files (default: custom/logs)",
    )

    args = parser.parse_args()

    # Setup logging (dual output to terminal and file)
    global logger
    logger = setup_logging(args.log_dir, "infer_groot_async")

    # Setup image recording if enabled
    if args.record_imgs:
        os.makedirs("eval_images", exist_ok=True)
        # Clear existing images
        for file in os.listdir("eval_images"):
            if file.endswith(('.jpg', '.png')):
                os.remove(os.path.join("eval_images", file))
        logger.info("[RECORD] Image recording enabled - saving to eval_images/")

    logger.info("=" * 60)
    logger.info("GR00T Async Inference - Producer-Consumer Architecture")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Action interval: {args.action_interval}s ({1/args.action_interval:.0f}Hz)")
    logger.info(f"Duration: {args.duration}s")
    logger.info("=" * 60)

    # Initialize robot
    logger.info("\n[INIT] Creating robot interface...")
    logger.info(f"  Camera: {args.cam_width}x{args.cam_height} @ {args.cam_fps}fps ({args.fourcc})")
    if args.fourcc == "MJPG":
        # MJPG compression: ~3 MB/s per camera vs ~28 MB/s raw
        bandwidth_mb = 3.0 * 2  # ~6 MB/s total for 2 cameras
        logger.info(f"  Est. USB bandwidth: ~{bandwidth_mb:.0f} MB/s (MJPG compressed, limit ~35 MB/s)")
    else:
        bandwidth_mb = (args.cam_width * args.cam_height * 3 * args.cam_fps * 2) / 1_000_000
        logger.info(f"  Est. USB bandwidth: {bandwidth_mb:.1f} MB/s (raw, limit ~35 MB/s)")
    robot = So101RobotInterface(
        serial_port=args.port,
        head_cam_idx=args.head_cam_idx,
        wrist_cam_idx=args.wrist_cam_idx,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        fourcc=args.fourcc,
    )

    # Initialize model
    logger.info("\n[INIT] Loading GR00T model...")
    inference = Gr00tLocalInference(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        task=args.task,
        denoising_steps=args.denoising_steps,
    )

    # Warmup model
    logger.info("\n[WARMUP] Running warmup inference...")
    dummy_img = np.zeros((args.cam_height, args.cam_width, 3), dtype=np.uint8)
    dummy_state = np.zeros(6)
    for _ in range(5):
        inference.get_action(dummy_img, dummy_img, dummy_state)
    logger.info("[WARMUP] Complete!")

    # Note: Camera pre-warming removed - causes race condition with robot.connect()
    # LeRobot handles camera connection internally

    # Create async engine
    engine = AsyncInferenceEngine(
        robot=robot,
        inference=inference,
        action_horizon=args.action_horizon,
        action_interval=args.action_interval,
        record_imgs=args.record_imgs,
        use_first_n_actions=args.use_first_n_actions,
        no_ensemble=args.no_ensemble,
    )

    # Run inference
    with robot.activate():
        # Go home if requested
        if args.go_home_first:
            logger.info("\n[RESET] Moving to training-aligned home position...")
            robot.go_home(training_aligned=True)
            time.sleep(1.0)
            state = robot.get_current_state()
            logger.info(f"  Home state: {np.round(state, 2)}")

        logger.info(f"\n[RUN] Starting async inference for {args.duration}s...")
        logger.info("  Press Ctrl+C to stop early\n")

        # Start async engine
        engine.start()

        try:
            start_time = time.time()
            last_stats_time = start_time

            while time.time() - start_time < args.duration:
                time.sleep(1.0)

                # Print stats every 5 seconds
                if time.time() - last_stats_time >= 5.0:
                    stats = engine.get_stats()
                    elapsed = time.time() - start_time

                    producer_rate = stats.producer_count / elapsed
                    consumer_rate = stats.consumer_count / elapsed
                    stale_pct = 100 * stats.stale_count / max(1, stats.consumer_count)
                    ensemble_pct = 100 * stats.ensemble_count / max(1, stats.consumer_count)

                    logger.info(f"[STATS] t={elapsed:.0f}s | "
                               f"Producer: {producer_rate:.1f}Hz | "
                               f"Consumer: {consumer_rate:.1f}Hz | "
                               f"Ensembled: {ensemble_pct:.0f}% | "
                               f"Latency: {stats.avg_latency_ms():.0f}ms")

                    last_stats_time = time.time()

        except KeyboardInterrupt:
            logger.info("\n\n[STOP] Interrupted by user")

        # Stop engine
        engine.stop()

        # Return home
        logger.info("\n[RESET] Returning to home position...")
        robot.go_home(training_aligned=True)

    # Final statistics
    stats = engine.get_stats()
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("ASYNC INFERENCE RESULTS (with Temporal Ensembling)")
    logger.info("=" * 60)
    logger.info(f"  Duration:           {elapsed:.1f}s")
    logger.info(f"  Producer count:     {stats.producer_count}")
    logger.info(f"  Consumer count:     {stats.consumer_count}")
    logger.info(f"  Producer rate:      {stats.producer_count/elapsed:.1f} Hz")
    logger.info(f"  Consumer rate:      {stats.consumer_count/elapsed:.1f} Hz")
    logger.info(f"  Stale actions:      {stats.stale_count} ({100*stats.stale_count/max(1,stats.consumer_count):.1f}%)")
    ensemble_pct = 100 * stats.ensemble_count / max(1, stats.consumer_count)
    logger.info(f"  Ensembled actions:  {stats.ensemble_count} ({ensemble_pct:.1f}%) <- actions with multiple predictions averaged")
    logger.info(f"  Avg latency:        {stats.avg_latency_ms():.0f}ms")
    logger.info(f"  Max latency:        {stats.max_latency_ms:.0f}ms")
    logger.info("=" * 60)

    # Save results
    results = {
        "duration_s": elapsed,
        "producer_count": stats.producer_count,
        "consumer_count": stats.consumer_count,
        "producer_rate_hz": stats.producer_count / elapsed,
        "consumer_rate_hz": stats.consumer_count / elapsed,
        "stale_count": stats.stale_count,
        "stale_rate": stats.stale_count / max(1, stats.consumer_count),
        "ensemble_count": stats.ensemble_count,
        "ensemble_rate": stats.ensemble_count / max(1, stats.consumer_count),
        "avg_latency_ms": stats.avg_latency_ms(),
        "max_latency_ms": stats.max_latency_ms,
        "config": {
            "model_path": args.model_path,
            "action_horizon": args.action_horizon,
            "action_interval": args.action_interval,
            "denoising_steps": args.denoising_steps,
        }
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n[SAVE] Results saved to: {args.output}")

    # Report recorded images
    if args.record_imgs:
        head_count = len([f for f in os.listdir("eval_images") if f.startswith("head_")])
        wrist_count = len([f for f in os.listdir("eval_images") if f.startswith("wrist_")])
        logger.info(f"\n[RECORD] Saved {head_count} head images and {wrist_count} wrist images to eval_images/")
        logger.info(f"         View with: ls eval_images/ | head -20")


if __name__ == "__main__":
    main()
