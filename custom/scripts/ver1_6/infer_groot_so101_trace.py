#!/usr/bin/env python3
"""
GR00T 1.6 Robot Inference with Detailed Tracing for SO-101.

This script extends infer_groot_so101_1_6.py with comprehensive tracing
to investigate the gap between open-loop and closed-loop performance.

Trace Data Captured:
- High-resolution timestamps for each component (capture, state, inference, execute)
- Full action buffer on each inference
- Joint states at each step
- Images saved on inference steps

Output:
    outputs/inference_traces/trace_YYYYMMDD_HHMMSS/
    ├── trace.jsonl           # All trace entries (JSON lines)
    ├── summary.json          # Statistics
    ├── config.json           # Run configuration
    └── images/               # Images on inference steps
        ├── step_0000_head.jpg
        ├── step_0000_wrist.jpg
        └── ...

Usage:
    python infer_groot_so101_trace.py --checkpoint <path> --task "pick the red cube" --duration 30

Reference: custom/scripts/ver1_6/infer_groot_so101_1_6.py
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
from typing import Any, Optional

import cv2
import numpy as np
import torch
import yaml

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
DEFAULT_CHECKPOINT = "outputs/groot_1_6_so101/checkpoint-10000"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ACTION_HORIZON = 8  # Changed from 16 to reduce image staleness (NVIDIA recommended)
ACTION_INTERVAL = 0.033  # 30Hz
NUM_DENOISING_STEPS = 4

HARDWARE_CONFIG = "custom/cfgs/so101_hardware.yaml"
DEFAULT_TASK = "pick the red cube from the table"
MAX_DURATION = 60.0

ARM_DIM = 5
GRIPPER_DIM = 1
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TraceEntry:
    """Single trace entry for one control loop iteration."""
    # Step identification
    step: int
    inference_triggered: bool
    action_idx_in_buffer: int

    # High-resolution timestamps (seconds since trace start)
    t_loop_start: float
    t_capture_start: float = 0.0
    t_capture_end: float = 0.0
    t_state_read: float = 0.0
    t_inference_start: float = 0.0
    t_inference_end: float = 0.0
    t_action_sent: float = 0.0
    t_loop_end: float = 0.0

    # Input data
    joint_states: list = field(default_factory=list)
    task_description: str = ""

    # Inference data (only when triggered)
    action_buffer: list = field(default_factory=list)  # Full 16-step prediction

    # Execution data
    action_executed: list = field(default_factory=list)
    action_delta: list = field(default_factory=list)

    # Derived metrics (computed at save time)
    loop_duration_ms: float = 0.0
    inference_duration_ms: float = 0.0
    capture_duration_ms: float = 0.0

    # NEW: State after inference completes (before first action)
    state_after_inference: list = field(default_factory=list)

    # NEW: How much state changed during inference time
    state_drift_during_inference: list = field(default_factory=list)

    # NEW: Error between current state and expected action
    prediction_error: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
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


class CameraManager:
    """Manage dual camera capture."""

    def __init__(self, hw_config: dict):
        self.cameras = {}
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
        """Capture frames from all cameras."""
        frames = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

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


def run_inference_loop_with_trace(
    policy,
    cameras: CameraManager,
    robot: RobotController,
    task: str,
    trace_dir: Path,
    max_duration: float = 60.0,
    action_interval: float = 0.033,
    action_horizon: int = 8,
):
    """Main inference loop with comprehensive tracing."""
    global running

    logger.info(f"\nStarting traced inference loop (max {max_duration}s)...")
    logger.info(f"Task: {task}")
    logger.info(f"Trace output: {trace_dir}")
    logger.info("Press Ctrl+C to stop\n")

    # Create trace output directories
    images_dir = trace_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Open trace file for streaming writes
    trace_file = trace_dir / "trace.jsonl"
    trace_fp = open(trace_file, "w")

    # Track timing
    trace_start = time.perf_counter()
    wall_start = time.time()
    step_count = 0

    action_buffer = []
    action_idx = 0
    last_inference_time = trace_start

    # Statistics collectors
    inference_times = []
    loop_times = []
    capture_times = []

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Current images holder (for saving on inference)
    current_images = None

    while running and (time.time() - wall_start) < max_duration:
        t_loop_start = time.perf_counter() - trace_start

        # Create trace entry for this step
        trace_entry = TraceEntry(
            step=step_count,
            inference_triggered=False,
            action_idx_in_buffer=action_idx,
            t_loop_start=t_loop_start,
            task_description=task,
        )

        # Check if we need new actions (inference step)
        if len(action_buffer) == 0 or action_idx >= len(action_buffer):
            trace_entry.inference_triggered = True

            # Capture images
            trace_entry.t_capture_start = time.perf_counter() - trace_start
            current_images = cameras.capture()
            trace_entry.t_capture_end = time.perf_counter() - trace_start

            # Get robot state
            state = robot.get_state()
            trace_entry.t_state_read = time.perf_counter() - trace_start
            trace_entry.joint_states = state.tolist()

            # Format observation
            observation = format_observation(current_images, state, task)

            # Run inference
            trace_entry.t_inference_start = time.perf_counter() - trace_start
            action_dict, info = policy.get_action(observation)
            trace_entry.t_inference_end = time.perf_counter() - trace_start

            # Extract action buffer and slice to action_horizon
            arm_actions = action_dict["single_arm"][0]
            gripper_actions = action_dict["gripper"][0]
            full_buffer = np.concatenate([arm_actions, gripper_actions], axis=1)
            action_buffer = full_buffer[:action_horizon]  # Actually use the horizon setting!
            action_idx = 0

            # Store full action buffer in trace
            trace_entry.action_buffer = action_buffer.tolist()

            # NEW: Capture state AFTER inference (before first action)
            # This measures how much arm moved during inference time
            state_after_inf = robot.get_state()
            trace_entry.state_after_inference = state_after_inf.tolist()
            trace_entry.state_drift_during_inference = (state_after_inf - state).tolist()

            # Compute timing metrics
            inf_duration = trace_entry.t_inference_end - trace_entry.t_inference_start
            trace_entry.inference_duration_ms = inf_duration * 1000
            inference_times.append(inf_duration)

            cap_duration = trace_entry.t_capture_end - trace_entry.t_capture_start
            trace_entry.capture_duration_ms = cap_duration * 1000
            capture_times.append(cap_duration)

            # Save images on inference steps
            for name, frame in current_images.items():
                img_path = images_dir / f"step_{step_count:04d}_{name}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Log progress
            time_since_last_inf = (trace_entry.t_inference_end + trace_start) - last_inference_time
            last_inference_time = trace_entry.t_inference_end + trace_start

            # Always log inference details for diagnosis
            state_str = ", ".join([f"{x:6.1f}" for x in state])
            action0 = action_buffer[0]
            action_last = action_buffer[-1]
            action0_str = ", ".join([f"{x:6.1f}" for x in action0])
            action_last_str = ", ".join([f"{x:6.1f}" for x in action_last])
            trajectory = action_buffer[-1] - action_buffer[0]
            traj_str = ", ".join([f"{x:+6.1f}" for x in trajectory])
            horizon_len = len(action_buffer)

            logger.info(f"")
            logger.info(f"[INF {step_count:4d}] t={trace_entry.t_inference_end:.2f}s, inf={trace_entry.inference_duration_ms:.0f}ms, horizon={horizon_len}")
            logger.info(f"  State now:   [{state_str}]")
            logger.info(f"  Action[0]:   [{action0_str}]  (delta: {np.round(action0 - state, 1).tolist()})")
            logger.info(f"  Action[{horizon_len-1}]:  [{action_last_str}]")
            logger.info(f"  Trajectory:  [{traj_str}]  ({horizon_len}-step motion)")
            logger.info(f"  Gripper:     {state[5]:.1f} -> {action0[5]:.1f} -> {action_last[5]:.1f}")

            # Warn if falling behind
            if trace_entry.inference_duration_ms > 100:
                logger.warning(f"  ^ Slow inference!")

        else:
            # Non-inference step: just read state for tracking
            state = robot.get_state()
            trace_entry.t_state_read = time.perf_counter() - trace_start
            trace_entry.joint_states = state.tolist()

        # Execute action from buffer
        action = action_buffer[action_idx]
        trace_entry.action_executed = action.tolist()

        # Compute delta (action - state: how far to move)
        state = np.array(trace_entry.joint_states)
        delta = action - state
        trace_entry.action_delta = delta.tolist()

        # NEW: Compute prediction error (state - action: where am I vs where model expected)
        # This is the inverse of action_delta and helps track if errors accumulate
        trace_entry.prediction_error = (-delta).tolist()

        # Send to robot
        robot.send_action(action)
        trace_entry.t_action_sent = time.perf_counter() - trace_start

        action_idx += 1
        trace_entry.action_idx_in_buffer = action_idx - 1  # The index we just used

        # Maintain action rate
        t_loop_end = time.perf_counter() - trace_start
        trace_entry.t_loop_end = t_loop_end
        trace_entry.loop_duration_ms = (t_loop_end - t_loop_start) * 1000
        loop_times.append(trace_entry.loop_duration_ms)

        # Warn if loop took too long
        if trace_entry.loop_duration_ms > 40 and not trace_entry.inference_triggered:
            logger.warning(f"Step {step_count}: Slow loop {trace_entry.loop_duration_ms:.1f}ms (no inference)")

        # Write trace entry
        trace_fp.write(json.dumps(trace_entry.to_dict()) + "\n")

        # Sleep to maintain rate
        elapsed = time.perf_counter() - (t_loop_start + trace_start)
        sleep_time = action_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        step_count += 1

    # Close trace file
    trace_fp.close()

    # Generate summary
    total_time = time.time() - wall_start
    summary = {
        "total_steps": step_count,
        "total_time_s": total_time,
        "effective_rate_hz": step_count / total_time if total_time > 0 else 0,
        "num_inferences": len(inference_times),
        "timing": {
            "loop_ms": {
                "mean": float(np.mean(loop_times)) if loop_times else 0,
                "std": float(np.std(loop_times)) if loop_times else 0,
                "min": float(np.min(loop_times)) if loop_times else 0,
                "max": float(np.max(loop_times)) if loop_times else 0,
                "p50": float(np.percentile(loop_times, 50)) if loop_times else 0,
                "p95": float(np.percentile(loop_times, 95)) if loop_times else 0,
                "p99": float(np.percentile(loop_times, 99)) if loop_times else 0,
            },
            "inference_ms": {
                "mean": float(np.mean(inference_times) * 1000) if inference_times else 0,
                "std": float(np.std(inference_times) * 1000) if inference_times else 0,
                "min": float(np.min(inference_times) * 1000) if inference_times else 0,
                "max": float(np.max(inference_times) * 1000) if inference_times else 0,
            },
            "capture_ms": {
                "mean": float(np.mean(capture_times) * 1000) if capture_times else 0,
                "std": float(np.std(capture_times) * 1000) if capture_times else 0,
                "min": float(np.min(capture_times) * 1000) if capture_times else 0,
                "max": float(np.max(capture_times) * 1000) if capture_times else 0,
            },
        },
        "warnings": {
            "slow_loops_gt_40ms": sum(1 for t in loop_times if t > 40),
            "slow_inferences_gt_100ms": sum(1 for t in inference_times if t * 1000 > 100),
        },
    }

    # Save summary
    summary_file = trace_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRACE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total steps:        {step_count}")
    logger.info(f"Total time:         {total_time:.1f}s")
    logger.info(f"Effective rate:     {summary['effective_rate_hz']:.1f}Hz")
    logger.info(f"Inferences:         {len(inference_times)}")
    logger.info(f"\nLoop timing (ms):")
    logger.info(f"  Mean: {summary['timing']['loop_ms']['mean']:.1f}")
    logger.info(f"  Std:  {summary['timing']['loop_ms']['std']:.1f}")
    logger.info(f"  P95:  {summary['timing']['loop_ms']['p95']:.1f}")
    logger.info(f"  P99:  {summary['timing']['loop_ms']['p99']:.1f}")
    logger.info(f"  Max:  {summary['timing']['loop_ms']['max']:.1f}")
    logger.info(f"\nInference timing (ms):")
    logger.info(f"  Mean: {summary['timing']['inference_ms']['mean']:.1f}")
    logger.info(f"  Max:  {summary['timing']['inference_ms']['max']:.1f}")
    logger.info(f"\nCapture timing (ms):")
    logger.info(f"  Mean: {summary['timing']['capture_ms']['mean']:.1f}")
    logger.info(f"  Max:  {summary['timing']['capture_ms']['max']:.1f}")
    logger.info(f"\nWarnings:")
    logger.info(f"  Slow loops (>40ms):      {summary['warnings']['slow_loops_gt_40ms']}")
    logger.info(f"  Slow inferences (>100ms): {summary['warnings']['slow_inferences_gt_100ms']}")
    logger.info(f"\nTrace saved to: {trace_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="GR00T 1.6 robot inference with tracing for SO-101"
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
        "--output-dir",
        type=str,
        default="outputs/inference_traces",
        help="Base output directory for traces"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending actions to robot"
    )
    parser.add_argument(
        "--hw-config",
        type=str,
        default=HARDWARE_CONFIG,
        help=f"Hardware config path (default: {HARDWARE_CONFIG})"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=ACTION_HORIZON,
        help=f"Number of actions to execute before re-inference (default: {ACTION_HORIZON})"
    )
    args = parser.parse_args()

    # Use the command-line argument value (stored locally since we can't use global after reference)
    action_horizon = args.action_horizon

    # Create trace directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = Path(args.output_dir) / f"trace_{timestamp}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    global logger
    log_file = trace_dir / "inference.log"
    logger = setup_logging(log_file)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 70)
    logger.info("GR00T 1.6 Robot Inference with Tracing")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Task:            {args.task}")
    logger.info(f"Duration:        {args.duration}s")
    logger.info(f"Action Horizon:  {action_horizon} (re-inference every {action_horizon} steps)")
    logger.info(f"Device:          {DEVICE}")
    logger.info(f"Dry run:         {args.dry_run}")
    logger.info(f"Trace dir:       {trace_dir}")
    logger.info("=" * 70)

    # Save config
    config = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "duration": args.duration,
        "dry_run": args.dry_run,
        "hw_config": args.hw_config,
        "device": DEVICE,
        "action_horizon": action_horizon,
        "action_interval": ACTION_INTERVAL,
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

        if args.dry_run:
            logger.info("  DRY RUN mode - actions will not be sent to robot")
            robot.robot = None

        # Run traced inference
        run_inference_loop_with_trace(
            policy=policy,
            cameras=cameras,
            robot=robot,
            task=args.task,
            trace_dir=trace_dir,
            max_duration=args.duration,
            action_interval=ACTION_INTERVAL,
            action_horizon=action_horizon,
        )

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        logger.info("\nCleaning up...")
        if cameras is not None:
            cameras.release()
        if robot is not None:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
