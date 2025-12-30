#!/usr/bin/env python3
"""
GR00T 1.6 Robot Inference with Camera Ablation Testing.

This script extends infer_groot_so101_trace.py to test whether the model
uses head camera, wrist camera, or both for navigation and grasping behavior.

Ablation Modes:
- Normal: No ablation (baseline)
- Ablate HEAD: Black out head camera to test if it drives navigation
- Ablate WRIST: Black out wrist camera to test if it drives grasping
- Ablate BOTH: Black out both cameras (sanity check)
- SWAP: Swap head/wrist inputs to test camera ordering

Expected Results:
- If behavior SAME with head ablated -> head camera NOT used for navigation
- If behavior DIFFERENT with head ablated -> head camera IS used
- Similar logic for wrist ablation

Usage:
    # Baseline run
    python infer_groot_ablation_test.py --checkpoint <path> --task "pick the blocks"

    # Ablate head camera (test navigation blindness)
    python infer_groot_ablation_test.py --checkpoint <path> --ablate-head

    # Ablate wrist camera (test grasp blindness)
    python infer_groot_ablation_test.py --checkpoint <path> --ablate-wrist

    # Swap cameras (test ordering issue)
    python infer_groot_ablation_test.py --checkpoint <path> --swap-cameras
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
DEFAULT_CHECKPOINT = "outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ACTION_HORIZON = 8  # Use 8 as default (NVIDIA recommended)
ACTION_INTERVAL = 0.033  # 30Hz
NUM_DENOISING_STEPS = 4

HARDWARE_CONFIG = "custom/cfgs/so101_hardware.yaml"
DEFAULT_TASK = "pick up the blocks and place them on the plate"
MAX_DURATION = 60.0

ARM_DIM = 5
GRIPPER_DIM = 1
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TraceEntry:
    """Single trace entry for one control loop iteration."""
    step: int
    inference_triggered: bool
    action_idx_in_buffer: int
    t_loop_start: float
    t_capture_start: float = 0.0
    t_capture_end: float = 0.0
    t_state_read: float = 0.0
    t_inference_start: float = 0.0
    t_inference_end: float = 0.0
    t_action_sent: float = 0.0
    t_loop_end: float = 0.0
    joint_states: list = field(default_factory=list)
    task_description: str = ""
    action_buffer: list = field(default_factory=list)
    action_executed: list = field(default_factory=list)
    action_delta: list = field(default_factory=list)
    loop_duration_ms: float = 0.0
    inference_duration_ms: float = 0.0
    capture_duration_ms: float = 0.0
    ablation_mode: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def setup_logging(log_file: Path = None):
    """Set up logging."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


logger = logging.getLogger(__name__)
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
    """Import modality config to register NEW_EMBODIMENT."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Modality config not found: {full_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Modality config loaded and registered")


class CameraManagerWithAblation:
    """Manage dual camera capture with ablation support."""

    def __init__(self, hw_config: dict, ablate_head: bool = False,
                 ablate_wrist: bool = False, swap_cameras: bool = False):
        self.cameras = {}
        self.ablate_head = ablate_head
        self.ablate_wrist = ablate_wrist
        self.swap_cameras = swap_cameras

        cam_config = hw_config.get("cameras", {})

        head_cfg = cam_config.get("head", {})
        self.cameras["head"] = self._init_camera(
            head_cfg.get("device_index", 4),
            head_cfg.get("resolution", {}).get("width", 640),
            head_cfg.get("resolution", {}).get("height", 480),
            "head"
        )

        wrist_cfg = cam_config.get("wrist", {})
        self.cameras["wrist"] = self._init_camera(
            wrist_cfg.get("device_index", 6),
            wrist_cfg.get("resolution", {}).get("width", 640),
            wrist_cfg.get("resolution", {}).get("height", 480),
            "wrist"
        )

        # Log ablation mode
        if self.ablate_head:
            logger.warning(">>> ABLATION MODE: HEAD camera will be BLACKED OUT <<<")
        if self.ablate_wrist:
            logger.warning(">>> ABLATION MODE: WRIST camera will be BLACKED OUT <<<")
        if self.swap_cameras:
            logger.warning(">>> ABLATION MODE: HEAD and WRIST cameras will be SWAPPED <<<")

    def _init_camera(self, device_index: int, width: int, height: int, name: str) -> cv2.VideoCapture:
        """Initialize a single camera."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {name} camera at index {device_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  {name} camera initialized: {actual_w}x{actual_h}")
        return cap

    def capture(self) -> dict:
        """Capture frames from all cameras with ablation applied."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store original frames for saving (before ablation)
        original_frames = {k: v.copy() for k, v in frames.items()}

        # Apply ablations to frames sent to model
        if self.swap_cameras:
            # Swap head and wrist
            frames["head"], frames["wrist"] = frames["wrist"].copy(), frames["head"].copy()

        if self.ablate_head:
            frames["head"] = np.zeros_like(frames["head"])

        if self.ablate_wrist:
            frames["wrist"] = np.zeros_like(frames["wrist"])

        # Return both original (for saving) and ablated (for model)
        return frames, original_frames

    def get_ablation_mode(self) -> str:
        """Return string describing current ablation mode."""
        modes = []
        if self.ablate_head:
            modes.append("HEAD_BLACK")
        if self.ablate_wrist:
            modes.append("WRIST_BLACK")
        if self.swap_cameras:
            modes.append("SWAPPED")
        return "_".join(modes) if modes else "NORMAL"

    def release(self):
        """Release all cameras."""
        for cap in self.cameras.values():
            cap.release()


class RobotController:
    """Interface to SO-101 robot arm."""

    def __init__(self, hw_config: dict):
        self.port = hw_config.get("robot_arms", {}).get("left", {}).get("port", "/dev/ttyACM1")
        self.use_degrees = hw_config.get("robot_arms", {}).get("left", {}).get("use_degrees", True)
        self.robot_id = hw_config.get("robot_arms", {}).get("left", {}).get("robot_id", "xlerobot_left_arm")
        self._init_robot()

    def _init_robot(self):
        """Initialize robot connection."""
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            robot_config = SO101FollowerConfig(
                port=self.port,
                id=self.robot_id,
                use_degrees=self.use_degrees,
            )
            self.robot = SO101Follower(robot_config)
            self.robot.connect()
            logger.info(f"  Robot connected on {self.port}")
        except ImportError as e:
            logger.warning(f"lerobot import failed: {e}")
            self.robot = None
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self.robot = None

    def get_state(self) -> np.ndarray:
        """Get current robot state (6 DOF)."""
        if self.robot is None:
            return np.zeros(ARM_DIM + GRIPPER_DIM, dtype=np.float32)
        obs = self.robot.get_observation()
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        state = np.array([obs[f"{name}.pos"] for name in motor_names], dtype=np.float32)
        return state

    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        if self.robot is None:
            return
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(motor_names)}
        self.robot.send_action(action_dict)

    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()


def format_observation(images: dict, state: np.ndarray, task: str) -> dict:
    """Format observation for GR00T policy input."""
    observation = {"video": {}, "state": {}, "language": {}}

    for name, frame in images.items():
        observation["video"][name] = frame[np.newaxis, np.newaxis, :, :, :]

    arm_state = state[:ARM_DIM]
    gripper_state = state[ARM_DIM:ARM_DIM + GRIPPER_DIM]
    observation["state"]["single_arm"] = arm_state[np.newaxis, np.newaxis, :]
    observation["state"]["gripper"] = gripper_state[np.newaxis, np.newaxis, :]
    observation["language"]["annotation.human.action.task_description"] = [[task]]

    return observation


def run_ablation_inference(
    policy,
    cameras: CameraManagerWithAblation,
    robot: RobotController,
    task: str,
    trace_dir: Path,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    action_horizon: int = 8,
):
    """Main inference loop with ablation support."""
    global running

    ablation_mode = cameras.get_ablation_mode()

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION TEST: {ablation_mode}")
    logger.info(f"{'='*70}")
    logger.info(f"Task: {task}")
    logger.info(f"Duration: {max_duration}s")
    logger.info(f"Action horizon: {action_horizon}")
    logger.info(f"Trace output: {trace_dir}")
    logger.info("Press Ctrl+C to stop\n")

    # Create trace output directories
    images_dir = trace_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Open trace file
    trace_file = trace_dir / "trace.jsonl"
    trace_fp = open(trace_file, "w")

    # Track timing
    trace_start = time.perf_counter()
    wall_start = time.time()
    step_count = 0

    action_buffer = []
    action_idx = 0

    inference_times = []
    loop_times = []

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    while running and (time.time() - wall_start) < max_duration:
        t_loop_start = time.perf_counter() - trace_start

        trace_entry = TraceEntry(
            step=step_count,
            inference_triggered=False,
            action_idx_in_buffer=action_idx,
            t_loop_start=t_loop_start,
            task_description=task,
            ablation_mode=ablation_mode,
        )

        # Check if we need new actions
        if len(action_buffer) == 0 or action_idx >= len(action_buffer):
            trace_entry.inference_triggered = True

            # Capture images (with ablation applied)
            trace_entry.t_capture_start = time.perf_counter() - trace_start
            ablated_images, original_images = cameras.capture()
            trace_entry.t_capture_end = time.perf_counter() - trace_start

            # Get robot state
            state = robot.get_state()
            trace_entry.t_state_read = time.perf_counter() - trace_start
            trace_entry.joint_states = state.tolist()

            # Format observation (using ABLATED images for model)
            observation = format_observation(ablated_images, state, task)

            # Run inference
            trace_entry.t_inference_start = time.perf_counter() - trace_start
            action_dict, info = policy.get_action(observation)
            trace_entry.t_inference_end = time.perf_counter() - trace_start

            # Extract action buffer
            arm_actions = action_dict["single_arm"][0]
            gripper_actions = action_dict["gripper"][0]
            full_buffer = np.concatenate([arm_actions, gripper_actions], axis=1)
            action_buffer = full_buffer[:action_horizon]
            action_idx = 0

            trace_entry.action_buffer = action_buffer.tolist()

            inf_duration = trace_entry.t_inference_end - trace_entry.t_inference_start
            trace_entry.inference_duration_ms = inf_duration * 1000
            inference_times.append(inf_duration)

            trace_entry.capture_duration_ms = (trace_entry.t_capture_end - trace_entry.t_capture_start) * 1000

            # Save ORIGINAL images (so we can see what was actually in front of camera)
            # Add ablation indicator to filename
            for name, frame in original_images.items():
                img_path = images_dir / f"step_{step_count:04d}_{name}_ORIG.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Also save what model SAW (ablated)
            for name, frame in ablated_images.items():
                img_path = images_dir / f"step_{step_count:04d}_{name}_MODEL.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Log inference details
            state_str = ", ".join([f"{x:6.1f}" for x in state])
            action0 = action_buffer[0]
            delta = action0 - state

            logger.info(f"")
            logger.info(f"[INF {step_count:4d}] t={trace_entry.t_inference_end:.2f}s, inf={trace_entry.inference_duration_ms:.0f}ms | {ablation_mode}")
            logger.info(f"  State:  [{state_str}]")
            logger.info(f"  Delta:  {np.round(delta, 1).tolist()}")
            logger.info(f"  Gripper: {state[5]:.1f} -> {action0[5]:.1f}")

        else:
            state = robot.get_state()
            trace_entry.t_state_read = time.perf_counter() - trace_start
            trace_entry.joint_states = state.tolist()

        # Execute action
        action = action_buffer[action_idx]
        trace_entry.action_executed = action.tolist()

        state = np.array(trace_entry.joint_states)
        delta = action - state
        trace_entry.action_delta = delta.tolist()

        robot.send_action(action)
        trace_entry.t_action_sent = time.perf_counter() - trace_start

        action_idx += 1
        trace_entry.action_idx_in_buffer = action_idx - 1

        t_loop_end = time.perf_counter() - trace_start
        trace_entry.t_loop_end = t_loop_end
        trace_entry.loop_duration_ms = (t_loop_end - t_loop_start) * 1000
        loop_times.append(trace_entry.loop_duration_ms)

        trace_fp.write(json.dumps(trace_entry.to_dict()) + "\n")

        elapsed = time.perf_counter() - (t_loop_start + trace_start)
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        step_count += 1

    trace_fp.close()

    # Generate summary
    total_time = time.time() - wall_start
    summary = {
        "ablation_mode": ablation_mode,
        "total_steps": step_count,
        "total_time_s": total_time,
        "effective_rate_hz": step_count / total_time if total_time > 0 else 0,
        "num_inferences": len(inference_times),
        "timing": {
            "loop_ms_mean": float(np.mean(loop_times)) if loop_times else 0,
            "inference_ms_mean": float(np.mean(inference_times) * 1000) if inference_times else 0,
        }
    }

    with open(trace_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION TEST COMPLETE: {ablation_mode}")
    logger.info(f"{'='*70}")
    logger.info(f"Total steps: {step_count}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Inferences: {len(inference_times)}")
    logger.info(f"Trace saved to: {trace_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="GR00T Inference with Camera Ablation Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Test Protocol:
  1. Run baseline (no flags)
  2. Run with --ablate-head to test if head camera drives navigation
  3. Run with --ablate-wrist to test if wrist camera drives grasping
  4. Run with --swap-cameras to test camera ordering

Expected Results:
  - If behavior SAME with --ablate-head -> Model ignores head camera
  - If behavior DIFFERENT with --ablate-head -> Model uses head camera
        """
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--task", type=str, default=DEFAULT_TASK,
        help=f"Task description (default: {DEFAULT_TASK})"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Max duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--action-horizon", type=int, default=ACTION_HORIZON,
        help=f"Action horizon (default: {ACTION_HORIZON})"
    )
    parser.add_argument(
        "--hw-config", type=str, default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/inference_traces",
        help="Output directory for traces"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without sending actions to robot"
    )

    # Ablation options
    parser.add_argument(
        "--ablate-head", action="store_true",
        help="Black out HEAD camera (test navigation blindness)"
    )
    parser.add_argument(
        "--ablate-wrist", action="store_true",
        help="Black out WRIST camera (test grasp blindness)"
    )
    parser.add_argument(
        "--swap-cameras", action="store_true",
        help="Swap HEAD and WRIST camera inputs"
    )

    args = parser.parse_args()

    # Determine ablation mode for directory naming
    ablation_suffix = "normal"
    if args.ablate_head and args.ablate_wrist:
        ablation_suffix = "ablate_both"
    elif args.ablate_head:
        ablation_suffix = "ablate_head"
    elif args.ablate_wrist:
        ablation_suffix = "ablate_wrist"
    elif args.swap_cameras:
        ablation_suffix = "swap_cameras"

    # Create trace directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = Path(args.output_dir) / f"ablation_{ablation_suffix}_{timestamp}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    global logger
    log_file = trace_dir / "inference.log"
    logger = setup_logging(log_file)

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 70)
    logger.info("GR00T Camera Ablation Test")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:     {args.checkpoint}")
    logger.info(f"Task:           {args.task}")
    logger.info(f"Duration:       {args.duration}s")
    logger.info(f"Action Horizon: {args.action_horizon}")
    logger.info(f"Ablate HEAD:    {args.ablate_head}")
    logger.info(f"Ablate WRIST:   {args.ablate_wrist}")
    logger.info(f"Swap Cameras:   {args.swap_cameras}")
    logger.info(f"Dry run:        {args.dry_run}")
    logger.info(f"Trace dir:      {trace_dir}")
    logger.info("=" * 70)

    # Save config
    config = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "duration": args.duration,
        "action_horizon": args.action_horizon,
        "ablate_head": args.ablate_head,
        "ablate_wrist": args.ablate_wrist,
        "swap_cameras": args.swap_cameras,
        "dry_run": args.dry_run,
        "timestamp": timestamp,
    }
    with open(trace_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        if not args.checkpoint.startswith("nvidia/"):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    cameras = None
    robot = None

    try:
        # Load hardware config
        hw_config = load_hardware_config(args.hw_config)

        # Initialize cameras with ablation
        logger.info("\nInitializing cameras with ablation settings...")
        cameras = CameraManagerWithAblation(
            hw_config,
            ablate_head=args.ablate_head,
            ablate_wrist=args.ablate_wrist,
            swap_cameras=args.swap_cameras
        )

        # Initialize robot
        logger.info("\nInitializing robot...")
        robot = RobotController(hw_config)

        if args.dry_run:
            logger.info("  DRY RUN mode - actions will not be sent to robot")
            robot.robot = None

        # Load model
        logger.info("\nLoading GR00T model...")
        import_modality_config(MODALITY_CONFIG_PATH)

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=DEVICE,
        )
        logger.info(f"  Model loaded on {DEVICE}")

        # Run ablation test
        run_ablation_inference(
            policy=policy,
            cameras=cameras,
            robot=robot,
            task=args.task,
            trace_dir=trace_dir,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            action_horizon=args.action_horizon,
        )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("\nCleaning up...")
        if cameras is not None:
            cameras.release()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
