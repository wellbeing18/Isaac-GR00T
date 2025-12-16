#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SO101 GR00T Stable Inference Script (Plan Locking)
# Implements "Plan Locking" to fix high-frequency vibration issues.
#
# Usage:
#   python custom/scripts/infer_groot_stable.py --model-path ... --execution-steps 8

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
import math

# --- One-Euro Filter ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        One Euro Filter for smoothing noisy signals.
        min_cutoff: Min cutoff frequency in Hz (lower = smoother, more lag)
        beta: Speed coefficient (higher = faster response to high speed changes)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(x0, dtype=float)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t = float(t)
        x = np.array(x, dtype=float)
        t_e = t - self.t_prev

        # Avoid division by zero or negative time
        if t_e <= 0: return self.x_prev

        # Estimate derivative (velocity)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Update cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        # Filter signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "cfgs" / "so101_hardware.yaml"

def load_hardware_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: str, script_name: str = "infer_groot_stable") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    # Force global logger update
    global logger_instance
    logger_instance = logger
    
    logger.info(f"Logging to: {log_file}")
    return logger

# LeRobot & GR00T imports
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

logger_instance: logging.Logger = None
def get_logger() -> logging.Logger:
    global logger_instance
    if logger_instance is None:
        fallback = logging.getLogger("fallback")
        if not fallback.handlers: 
            fallback.addHandler(logging.StreamHandler(sys.stdout))
            fallback.setLevel(logging.INFO)
        return fallback
    return logger_instance

@dataclass
class ActionPrediction:
    timestamp: float
    actions: np.ndarray
    state_at_capture: np.ndarray
    inference_time_ms: float
    chunk_idx: int

@dataclass
class AsyncStats:
    producer_count: int = 0
    consumer_count: int = 0
    stale_count: int = 0
    execution_switches: int = 0 # How many times we switched plans
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.consumer_count)

# --- Robot Interface (Duplicated for standalone) ---
class So101RobotInterface:
    HOME_POSITION_TRAINING = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -99.0,
        "elbow_flex.pos": 100.0,
        "wrist_flex.pos": 50.0,
        "wrist_roll.pos": -1.0,
        "gripper.pos": 0.5,
    }

    def __init__(self, serial_port, robot_id, head_cam_idx, wrist_cam_idx, fps, width, height, fourcc, use_degrees=True):
        cameras_cfg = {
            "head": OpenCVCameraConfig(index_or_path=head_cam_idx, fps=fps, width=width, height=height, fourcc=fourcc),
            "wrist": OpenCVCameraConfig(index_or_path=wrist_cam_idx, fps=fps, width=width, height=height, fourcc=fourcc),
        }
        robot_cfg = SO101FollowerConfig(port=serial_port, id=robot_id, cameras=cameras_cfg, use_degrees=use_degrees)
        self.robot = SO101Follower(robot_cfg)

    @contextmanager
    def activate(self):
        try:
            if not self.robot.is_connected: self.robot.connect()
        except DeviceAlreadyConnectedError: pass
        try: yield
        finally: self.robot.disconnect()

    def go_home(self, training_aligned=True):
        if training_aligned: self.robot.send_action(self.HOME_POSITION_TRAINING)
        else: pass # Simplified
        time.sleep(2)

    def get_current_state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"], obs["elbow_flex.pos"],
                         obs["wrist_flex.pos"], obs["wrist_roll.pos"], obs["gripper.pos"]])

    def get_observation(self) -> dict:
        obs = self.robot.get_observation()
        state = np.array([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"], obs["elbow_flex.pos"],
                          obs["wrist_flex.pos"], obs["wrist_roll.pos"], obs["gripper.pos"]])
        return {"head": obs["head"], "wrist": obs["wrist"], "state": state}

    def set_target_state(self, target_state: np.ndarray):
        if isinstance(target_state, torch.Tensor): target_state = target_state.numpy()
        action = {
            "shoulder_pan.pos": float(target_state[0]), "shoulder_lift.pos": float(target_state[1]),
            "elbow_flex.pos": float(target_state[2]), "wrist_flex.pos": float(target_state[3]),
            "wrist_roll.pos": float(target_state[4]), "gripper.pos": float(target_state[5]),
        }
        self.robot.send_action(action)

# --- LoRA Loading (Duplicated) ---
def is_lora_checkpoint(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()

def get_lora_config(model_path: str) -> dict:
    with open(Path(model_path) / "adapter_config.json", "r") as f: return json.load(f)

def load_groot_with_lora(model_path, embodiment_tag, modality_config, modality_transform, denoising_steps=4, merge_weights=True):
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.dataset import DatasetMetadata
    from gr00t.model.policy import EmbodimentTag
    
    lora_config = get_lora_config(model_path)
    base_model_path = lora_config.get("base_model_name_or_path")
    get_logger().info(f"[LoRA] Loading base: {base_model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=compute_dtype)
    base_model.eval()
    
    peft_model = PeftModel.from_pretrained(base_model, str(model_path))
    if merge_weights: merged_model = peft_model.merge_and_unload()
    else: merged_model = peft_model
    merged_model.to(device=device)
    
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.device = device
    policy._modality_config = modality_config
    policy._modality_transform = modality_transform
    policy.model = merged_model
    policy.model.action_head.num_inference_timesteps = denoising_steps
    
    exp_cfg_dir = Path(model_path) / "experiment_cfg"
    if exp_cfg_dir.exists():
        with open(exp_cfg_dir / "metadata.json", "r") as f: metadatas = json.load(f)
        embodiment_tag_enum = EmbodimentTag(embodiment_tag)
        policy.embodiment_tag = embodiment_tag_enum
        metadata = DatasetMetadata.model_validate(metadatas.get(embodiment_tag_enum.value))
        policy._modality_transform.set_metadata(metadata)
        policy.metadata = metadata

    policy._video_delta_indices = np.array(modality_config["video"].delta_indices)
    policy._video_horizon = len(policy._video_delta_indices)
    if "state" in modality_config:
        policy._state_delta_indices = np.array(modality_config["state"].delta_indices)
        policy._state_horizon = len(policy._state_delta_indices)
    policy._action_delta_indices = np.array(modality_config["action"].delta_indices)
    
    return policy

class Gr00tLocalInference:
    def __init__(self, model_path, data_config, embodiment_tag, task, denoising_steps):
        self.task = task
        data_cfg = load_data_config(data_config)
        modality_config = data_cfg.modality_config()
        modality_transform = data_cfg.transform()
        
        if is_lora_checkpoint(model_path):
            self.policy = load_groot_with_lora(model_path, embodiment_tag, modality_config, modality_transform, denoising_steps)
        else:
            self.policy = Gr00tPolicy(model_path, embodiment_tag, modality_config, modality_transform, denoising_steps)
            
    def get_action(self, front_img, wrist_img, state):
        obs_dict = {
            "video.front": front_img[np.newaxis, :, :, :],
            "video.wrist": wrist_img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.task],
        }
        return self.policy.get_action(obs_dict)

# --- Stable Inference Engine ---
class StableInferenceEngine:
    """
    Producer-Consumer engine with PLAN LOCKING and ONE-EURO FILTERING.
    """
    def __init__(self, robot, inference, action_horizon=16, action_interval=0.033, execution_steps=8, record_imgs=False, use_filter=True):
        self.robot = robot
        self.inference = inference
        self.action_horizon = action_horizon
        self.action_interval = action_interval
        self.execution_steps = execution_steps
        self.record_imgs = record_imgs
        self.use_filter = use_filter
        
        self.prediction_queue = queue.Queue(maxsize=2)
        self.running = False
        self.stats = AsyncStats()
        self.lock = threading.Lock()
        self.robot_lock = threading.Lock()
        self.producer_thread = None
        self.consumer_thread = None
        self.MODALITY_KEYS = ["single_arm", "gripper"]
        
        # One-Euro Filter (Initialized in consumer loop to get first state)
        self.filter = None

    def start(self):
        self.running = True
        self.producer_thread = threading.Thread(target=self._producer_loop, name="Producer")
        self.consumer_thread = threading.Thread(target=self._consumer_loop, name="Consumer")
        self.producer_thread.start()
        self.consumer_thread.start()
        get_logger().info(f"[STABLE] Threads started. Execution steps: {self.execution_steps}")
        # Add explicit flush for debugging
        sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.producer_thread: self.producer_thread.join(timeout=2.0)
        if self.consumer_thread: self.consumer_thread.join(timeout=2.0)

    def _producer_loop(self):
        chunk_idx = 0
        while self.running:
            try:
                # Capture
                with self.robot_lock:
                    obs = self.robot.get_observation()
                
                # Inference
                inference_start = time.time()
                action_dict = self.inference.get_action(obs["head"], obs["wrist"], obs["state"])
                inference_time_ms = (time.time() - inference_start) * 1000
                
                # Format
                actions = np.zeros((self.action_horizon, 6))
                for i in range(self.action_horizon):
                    actions[i] = np.concatenate([np.atleast_1d(action_dict[f"action.{k}"][i]) for k in self.MODALITY_KEYS], axis=0)
                
                prediction = ActionPrediction(
                    timestamp=time.time(), # Capture time approx
                    actions=actions,
                    state_at_capture=obs["state"].copy(),
                    inference_time_ms=inference_time_ms,
                    chunk_idx=chunk_idx
                )
                
                # Push to queue (Drop old if full to keep fresh)
                if self.prediction_queue.full():
                    try: self.prediction_queue.get_nowait()
                    except queue.Empty: pass
                self.prediction_queue.put(prediction)
                
                with self.lock: self.stats.producer_count += 1
                chunk_idx += 1
                
                if chunk_idx % 10 == 0:
                    get_logger().info(f"[PRODUCER] Chunk {chunk_idx}: {inference_time_ms:.0f}ms")
                    
            except Exception as e:
                get_logger().error(f"[PRODUCER] Error: {e}")
                time.sleep(0.1)

    def _consumer_loop(self):
        current_plan = None
        plan_step_idx = 0
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Check if we need a new plan
                # 1. No plan yet
                # 2. Finished current plan (reached execution_steps limit)
                # 3. Finished current plan (reached horizon limit)
                need_new_plan = (current_plan is None) or \
                                (plan_step_idx >= self.execution_steps) or \
                                (plan_step_idx >= len(current_plan))
                
                if need_new_plan:
                    # Try to get NEWEST plan
                    new_pred = None
                    try:
                        # Drain queue to get the absolute latest
                        while not self.prediction_queue.empty():
                            new_pred = self.prediction_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    if new_pred:
                        current_plan = new_pred.actions
                        plan_step_idx = 0
                        with self.lock:
                            self.stats.execution_switches += 1
                            # Latency is time from capture until we START executing it
                            latency = (time.time() - new_pred.timestamp) * 1000
                            self.stats.total_latency_ms += latency
                            self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency)
                        
                        # LOGGING: New plan accepted
                        # Compare first step of new plan with current robot state
                        with self.robot_lock:
                            current_state = self.robot.get_current_state()
                        
                        # Initialize filter on first plan
                        if self.use_filter and self.filter is None:
                            # min_cutoff=0.1 (Very smooth), beta=0.01 (Low latency response)
                            self.filter = OneEuroFilter(time.time(), current_state, min_cutoff=0.1, beta=0.01)
                            get_logger().info("[FILTER] One-Euro Filter Initialized (min_cutoff=0.1)")
                        
                        target_start = current_plan[0]
                        diff = target_start - current_state
                        diff_norm = np.linalg.norm(diff)
                        
                        get_logger().info(f"[PLAN] New plan accepted (Latency: {latency:.0f}ms)")
                        get_logger().info(f"       Current State: {np.round(current_state, 2)}")
                        get_logger().info(f"       Plan Start:    {np.round(target_start, 2)}")
                        get_logger().info(f"       Delta:         {np.round(diff, 2)} (Norm: {diff_norm:.2f})")
                        
                        if diff_norm < 1.0:
                            get_logger().warning("[PLAN] WARNING: Plan is very close to current state (Stuck?)")

                # Execute
                target_action = None
                if current_plan is not None and plan_step_idx < len(current_plan):
                    target_action = current_plan[plan_step_idx]
                    plan_step_idx += 1
                
                if target_action is not None:
                    # Apply Filter
                    final_action = target_action
                    if self.use_filter and self.filter is not None:
                        final_action = self.filter(time.time(), target_action)
                        
                    with self.robot_lock:
                        self.robot.set_target_state(final_action)
                        # Optional: Log execution every step? Maybe too noisy.
                        # Let's log if it's the first step of the plan
                        if plan_step_idx == 1:
                            get_logger().info(f"[EXEC] Executing step 0 of new plan (Filtered: {self.use_filter})")

                    with self.lock: self.stats.consumer_count += 1
                else:
                    with self.lock: self.stats.stale_count += 1
                
                # Sleep
                elapsed = time.time() - loop_start
                sleep_time = self.action_interval - elapsed
                if sleep_time > 0: time.sleep(sleep_time)
                
            except Exception as e:
                get_logger().error(f"[CONSUMER] Error: {e}")
                time.sleep(0.033)
                
    def get_stats(self):
        with self.lock:
            return AsyncStats(self.stats.producer_count, self.stats.consumer_count, 
                              self.stats.stale_count, self.stats.execution_switches, 
                              self.stats.total_latency_ms, self.stats.max_latency_ms)

def main():
    parser = argparse.ArgumentParser(description="GR00T Stable Inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", default="pick the red cube from the table")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--execution-steps", type=int, default=8, help="Steps to execute per plan (Plan Locking)")
    parser.add_argument("--go-home-first", action="store_true")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--no-filter", action="store_true", help="Disable One-Euro Filter")
    
    # Defaults from config
    hw = load_hardware_config()
    cam = hw.get("cameras", {})
    
    args = parser.parse_args()
    
    global logger_instance
    logger_instance = setup_logging("custom/logs", "infer_groot_stable")
    
    robot = So101RobotInterface(
        serial_port=hw.get("robot_arms", {}).get("left", {}).get("port", "/dev/ttyACM0"),
        robot_id="xlerobot_left_arm",
        head_cam_idx=cam.get("head", {}).get("device_index", 8),
        wrist_cam_idx=cam.get("wrist", {}).get("device_index", 4),
        fps=30, width=640, height=480, fourcc="MJPG"
    )
    
    inference = Gr00tLocalInference(args.model_path, "so100_dualcam", "new_embodiment", args.task, args.denoising_steps)
    
    # Warmup
    inference.get_action(np.zeros((480,640,3),dtype=np.uint8), np.zeros((480,640,3),dtype=np.uint8), np.zeros(6))
    
    engine = StableInferenceEngine(robot, inference, execution_steps=args.execution_steps, use_filter=not args.no_filter)
    
    with robot.activate():
        if args.go_home_first: robot.go_home()
        engine.start()
        try:
            start = time.time()
            while time.time() - start < args.duration:
                time.sleep(5)
                stats = engine.get_stats()
                get_logger().info(f"[STATS] Switches: {stats.execution_switches} | Latency: {stats.avg_latency_ms():.0f}ms")
        except KeyboardInterrupt: pass
        engine.stop()
        robot.go_home()

if __name__ == "__main__":
    main()

