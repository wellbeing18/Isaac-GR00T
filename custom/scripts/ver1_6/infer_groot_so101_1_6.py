#!/usr/bin/env python3
"""
GR00T 1.6 Robot Inference for SO-101.

This script runs a finetuned GR00T model on the real SO-101 robot arm.
It captures images from dual cameras, reads robot state, and executes predicted actions.

Execution Flow:
1. Load modality config (registers NEW_EMBODIMENT)
2. Load hardware config from YAML
3. Initialize robot connection (serial port)
4. Initialize cameras (head + wrist)
5. Load Gr00tPolicy with checkpoint
6. Main loop:
   a. Capture images from both cameras
   b. Read robot state (6 DOF)
   c. Format observation dict (batch dim, temporal dim)
   d. Run policy.get_action(observation)
   e. Execute action chunk at 30Hz
   f. Repeat until duration exceeded or user stops

Observation Format (critical for correct inference):
    observation = {
        "video": {
            "head": np.array(...),   # (B=1, T=1, H=480, W=640, C=3) uint8
            "wrist": np.array(...),  # (B=1, T=1, H=480, W=640, C=3) uint8
        },
        "state": {
            "single_arm": np.array(...),  # (B=1, T=1, D=5) float32
            "gripper": np.array(...),      # (B=1, T=1, D=1) float32
        },
        "language": {
            "annotation.human.action.task_description": [["pick the red cube"]]
        }
    }

Usage:
    python infer_groot_so101_1_6.py --checkpoint outputs/groot_1_6_so101/checkpoint-10000

Reference: getting_started/policy.md, custom/scripts/ver1_5/infer_groot_simple.py
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Model
DEFAULT_CHECKPOINT = "outputs/groot_1_6_so101/checkpoint-10000"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Inference Settings
ACTION_HORIZON = 16       # Actions per inference (must match training)
ACTION_INTERVAL = 0.033   # 30Hz execution rate (1/30 seconds)
NUM_DENOISING_STEPS = 4   # DiT denoising iterations (NVIDIA default)

# Hardware Config (external file)
HARDWARE_CONFIG = "custom/cfgs/so101_hardware.yaml"

# Task
DEFAULT_TASK = "pick the red cube from the table"

# Recording
RECORD_IMAGES = False     # Save images to eval_images/
MAX_DURATION = 60.0       # Maximum run duration in seconds

# State/Action dimensions for SO-101
ARM_DIM = 5               # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DIM = 1           # gripper position
# ============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for clean shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    logger.info("\nShutdown requested...")
    running = False


def load_hardware_config(config_path: str) -> dict:
    """Load hardware configuration from YAML file."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {full_path}")

    with open(full_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded hardware config from: {config_path}")
    return config


def import_modality_config(config_path: str):
    """Import the modality config module to register NEW_EMBODIMENT."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Modality config not found: {full_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Modality config loaded and registered")


class CameraManager:
    """Manage dual camera capture."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
        cam_config = hw_config.get("cameras", {})

        # Initialize head camera
        head_cfg = cam_config.get("head", {})
        self.cameras["head"] = self._init_camera(
            head_cfg.get("device_index", 4),
            head_cfg.get("resolution", {}).get("width", 640),
            head_cfg.get("resolution", {}).get("height", 480),
            "head"
        )

        # Initialize wrist camera
        wrist_cfg = cam_config.get("wrist", {})
        self.cameras["wrist"] = self._init_camera(
            wrist_cfg.get("device_index", 6),
            wrist_cfg.get("resolution", {}).get("width", 640),
            wrist_cfg.get("resolution", {}).get("height", 480),
            "wrist"
        )

    def _init_camera(self, device_index: int, width: int, height: int, name: str) -> cv2.VideoCapture:
        """Initialize a single camera."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera at index {device_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Verify resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  {name} camera initialized: {actual_w}x{actual_h}")

        return cap

    def capture(self) -> dict:
        """Capture frames from all cameras."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")
            # Convert BGR to RGB
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()


class RobotController:
    """Interface to SO-101 robot arm via serial."""

    def __init__(self, hw_config: dict):
        self.port = hw_config.get("robot_arms", {}).get("left", {}).get("port", "/dev/ttyACM1")
        self.use_degrees = hw_config.get("robot_arms", {}).get("left", {}).get("use_degrees", True)

        # Initialize robot connection
        self._init_robot()

    def _init_robot(self):
        """Initialize robot connection."""
        try:
            from lerobot.common.robot_devices.robots.configs import So100RobotConfig
            from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

            robot_config = So100RobotConfig()
            # Update port if needed
            for name in robot_config.follower_arms:
                robot_config.follower_arms[name].port = self.port

            self.robot = ManipulatorRobot(robot_config)
            self.robot.connect()

            logger.info(f"  Robot connected on {self.port}")

        except ImportError:
            logger.warning("lerobot not available, using mock robot")
            self.robot = None

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            logger.warning("Using mock robot for testing")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (6 DOF)."""
        if self.robot is None:
            # Mock state for testing
            return np.zeros(ARM_DIM + GRIPPER_DIM, dtype=np.float32)

        # Read state from robot
        obs = self.robot.get_observation()
        state = obs.get("observation.state", np.zeros(ARM_DIM + GRIPPER_DIM))

        return state.astype(np.float32)

    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        if self.robot is None:
            return  # Skip for mock robot

        # Send action to robot
        action_dict = {"action": action}
        self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()


def format_observation(
    images: dict,
    state: np.ndarray,
    task: str,
) -> dict:
    """Format observation for GR00T policy input.

    Expected format:
        video: (B=1, T=1, H, W, C) uint8
        state: (B=1, T=1, D) float32
        language: [[str]]
    """
    observation = {
        "video": {},
        "state": {},
        "language": {},
    }

    # Video: add batch and temporal dimensions
    for name, frame in images.items():
        # frame is (H, W, C), we need (B=1, T=1, H, W, C)
        observation["video"][name] = frame[np.newaxis, np.newaxis, :, :, :]

    # State: split into arm and gripper
    arm_state = state[:ARM_DIM]  # First 5 values
    gripper_state = state[ARM_DIM:ARM_DIM + GRIPPER_DIM]  # 6th value

    # Add batch and temporal dimensions: (B=1, T=1, D)
    observation["state"]["single_arm"] = arm_state[np.newaxis, np.newaxis, :]
    observation["state"]["gripper"] = gripper_state[np.newaxis, np.newaxis, :]

    # Language: nested list format
    observation["language"]["annotation.human.action.task_description"] = [[task]]

    return observation


def run_inference_loop(
    policy,
    cameras: CameraManager,
    robot: RobotController,
    task: str,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    record_images: bool = False,
):
    """Main inference loop."""
    global running

    logger.info(f"\nStarting inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Action interval: {action_interval*1000:.1f}ms ({1/action_interval:.1f}Hz)")
    logger.info("Press Ctrl+C to stop\n")

    start_time = time.time()
    step_count = 0
    inference_times = []

    # Create record directory if needed
    if record_images:
        record_dir = PROJECT_ROOT / "eval_images" / f"inference_{int(start_time)}"
        record_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Recording images to: {record_dir}")

    action_buffer = []
    action_idx = 0

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        # Check if we need new actions
        if len(action_buffer) == 0 or action_idx >= len(action_buffer):
            # Capture images
            images = cameras.capture()

            # Get robot state
            state = robot.get_state()

            # Format observation
            observation = format_observation(images, state, task)

            # Run inference
            inf_start = time.time()
            action_dict, info = policy.get_action(observation)
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)

            # Extract action buffer (horizon x dim for each joint group)
            # Concatenate arm and gripper actions
            arm_actions = action_dict["single_arm"][0]  # (horizon, arm_dim)
            gripper_actions = action_dict["gripper"][0]  # (horizon, gripper_dim)
            action_buffer = np.concatenate([arm_actions, gripper_actions], axis=1)
            action_idx = 0

            if step_count % 10 == 0:
                logger.info(f"Step {step_count}: inference={inf_time*1000:.1f}ms, "
                           f"buffer_size={len(action_buffer)}")

            # Record images if enabled
            if record_images:
                for name, frame in images.items():
                    img_path = record_dir / f"step_{step_count:04d}_{name}.jpg"
                    cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Execute action from buffer
        action = action_buffer[action_idx]
        robot.send_action(action)
        action_idx += 1
        step_count += 1

        # Maintain action rate
        elapsed = time.time() - loop_start
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"Inference loop complete")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Effective rate: {step_count/total_time:.1f}Hz")

    if inference_times:
        logger.info(f"  Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
        logger.info(f"  Max inference time: {np.max(inference_times)*1000:.1f}ms")

    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="GR00T 1.6 robot inference for SO-101"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=DEFAULT_TASK,
        help=f"Task description (default: {DEFAULT_TASK})"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=MAX_DURATION,
        help=f"Max duration in seconds (default: {MAX_DURATION})"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record images during inference"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending actions to robot (for testing)"
    )
    parser.add_argument(
        "--hw-config",
        type=str,
        default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
    )
    args = parser.parse_args()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("GR00T 1.6 Robot Inference")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task:       {args.task}")
    print(f"Duration:   {args.duration}s")
    print(f"Device:     {DEVICE}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 70)

    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        # Check if it's a HuggingFace model ID
        if not args.checkpoint.startswith("nvidia/"):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    cameras = None
    robot = None

    try:
        # Load configs
        import_modality_config(MODALITY_CONFIG_PATH)
        hw_config = load_hardware_config(args.hw_config)

        # Load policy
        logger.info("\nLoading policy...")
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=DEVICE,
        )
        logger.info(f"  Policy loaded on {DEVICE}")

        # Initialize hardware
        logger.info("\nInitializing hardware...")
        cameras = CameraManager(hw_config)
        robot = RobotController(hw_config)

        # Override robot with dry-run mock if requested
        if args.dry_run:
            logger.info("  DRY RUN mode - actions will not be sent to robot")
            robot.robot = None

        # Run inference
        run_inference_loop(
            policy=policy,
            cameras=cameras,
            robot=robot,
            task=args.task,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            record_images=args.record,
        )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up
        logger.info("\nCleaning up...")
        if cameras is not None:
            cameras.release()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
