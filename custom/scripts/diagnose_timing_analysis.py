#!/usr/bin/env python3
"""
GR00T Inference Pipeline Timing Analysis

This script instruments the entire inference pipeline to identify timing bottlenecks:
1. Camera capture timing
2. State read timing
3. Model inference timing (breakdown by component)
4. Action execution timing
5. Full loop timing

Usage:
    # Analyze timing with dummy inputs (no robot)
    python diagnose_timing_analysis.py --model-path /path/to/checkpoint --num-iterations 50

    # Analyze with real robot
    python diagnose_timing_analysis.py --model-path /path/to/checkpoint --with-robot

    # Test different configurations
    python diagnose_timing_analysis.py --model-path /path/to/checkpoint \
        --action-horizon 8 --action-interval 0.02 --denoising-steps 16
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
import statistics

import numpy as np

# Try imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TimingBreakdown:
    """Timing for a single inference iteration."""
    iteration: int
    camera_capture_ms: float = 0.0
    state_read_ms: float = 0.0
    obs_dict_build_ms: float = 0.0
    model_inference_ms: float = 0.0
    action_extraction_ms: float = 0.0
    action_execution_ms: float = 0.0
    total_loop_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingStats:
    """Statistics for a timing metric."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def calc_stats(values: List[float]) -> TimingStats:
    """Calculate statistics for a list of timing values."""
    if not values:
        return TimingStats(0, 0, 0, 0, 0, 0, 0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return TimingStats(
        mean_ms=statistics.mean(values),
        std_ms=statistics.stdev(values) if n > 1 else 0,
        min_ms=min(values),
        max_ms=max(values),
        p50_ms=sorted_vals[int(n * 0.50)],
        p95_ms=sorted_vals[int(n * 0.95)] if n >= 20 else max(values),
        p99_ms=sorted_vals[int(n * 0.99)] if n >= 100 else max(values),
    )


class TimingAnalyzer:
    """Analyzes timing of GR00T inference pipeline."""

    def __init__(
        self,
        model_path: str,
        data_config: str = "so100_dualcam",
        embodiment_tag: str = "new_embodiment",
        denoising_steps: int = 4,
        action_horizon: int = 12,
        action_interval: float = 0.033,
    ):
        self.model_path = model_path
        self.data_config = data_config
        self.embodiment_tag = embodiment_tag
        self.denoising_steps = denoising_steps
        self.action_horizon = action_horizon
        self.action_interval = action_interval

        self.policy = None
        self.robot = None
        self.timings: List[TimingBreakdown] = []

    def load_model(self):
        """Load the GR00T model."""
        print(f"[MODEL] Loading model from: {self.model_path}")
        print(f"[MODEL] Data config: {self.data_config}")
        print(f"[MODEL] Denoising steps: {self.denoising_steps}")

        # Import model utilities
        sys.path.insert(0, str(Path(__file__).parent))
        from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora

        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy

        data_cfg = load_data_config(self.data_config)
        modality_config = data_cfg.modality_config()
        modality_transform = data_cfg.transform()

        if is_lora_checkpoint(self.model_path):
            print("[MODEL] Detected LoRA checkpoint")
            self.policy = load_groot_with_lora(
                model_path=self.model_path,
                embodiment_tag=self.embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=self.denoising_steps,
                merge_weights=True,
            )
        else:
            print("[MODEL] Loading full checkpoint")
            self.policy = Gr00tPolicy(
                model_path=self.model_path,
                embodiment_tag=self.embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=self.denoising_steps,
            )

        print("[MODEL] Model loaded successfully")

        # Warmup
        print("[MODEL] Running warmup (10 iterations)...")
        dummy_obs = self._create_dummy_obs()
        for _ in range(10):
            with torch.no_grad():
                _ = self.policy.get_action(dummy_obs)
        print("[MODEL] Warmup complete")

    def connect_robot(self, port: str, head_cam_idx: int, wrist_cam_idx: int):
        """Connect to the SO101 robot."""
        from infer_groot_so101 import SO101Robot

        print(f"[ROBOT] Connecting to robot on {port}...")
        self.robot = SO101Robot(
            port=port,
            head_cam_idx=head_cam_idx,
            wrist_cam_idx=wrist_cam_idx,
        )
        self.robot.connect()
        print("[ROBOT] Robot connected")

    def disconnect_robot(self):
        """Disconnect from robot."""
        if self.robot:
            self.robot.disconnect()
            print("[ROBOT] Robot disconnected")

    def _create_dummy_obs(self) -> Dict:
        """Create dummy observation dict for testing."""
        return {
            "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5), dtype=np.float64),
            "state.gripper": np.zeros((1, 1), dtype=np.float64),
            "annotation.human.task_description": ["pick the red cube from the table"],
        }

    def run_iteration_dummy(self, iteration: int) -> TimingBreakdown:
        """Run a single iteration with dummy inputs (no robot)."""
        timing = TimingBreakdown(iteration=iteration)
        loop_start = time.perf_counter()

        # Simulate camera capture
        t0 = time.perf_counter()
        dummy_head = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        dummy_wrist = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        timing.camera_capture_ms = (time.perf_counter() - t0) * 1000

        # Simulate state read
        t0 = time.perf_counter()
        dummy_state = np.random.randn(6).astype(np.float64)
        timing.state_read_ms = (time.perf_counter() - t0) * 1000

        # Build observation dict
        t0 = time.perf_counter()
        obs_dict = {
            "video.front": dummy_head[np.newaxis, :, :, :],
            "video.wrist": dummy_wrist[np.newaxis, :, :, :],
            "state.single_arm": dummy_state[:5][np.newaxis, :],
            "state.gripper": dummy_state[5:6][np.newaxis, :],
            "annotation.human.task_description": ["pick the red cube from the table"],
        }
        timing.obs_dict_build_ms = (time.perf_counter() - t0) * 1000

        # Model inference
        t0 = time.perf_counter()
        with torch.no_grad():
            action = self.policy.get_action(obs_dict)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing.model_inference_ms = (time.perf_counter() - t0) * 1000

        # Action extraction
        t0 = time.perf_counter()
        action_chunk = np.zeros((self.action_horizon, 6))
        for i in range(self.action_horizon):
            action_chunk[i] = np.concatenate([
                np.atleast_1d(action["action.single_arm"][i]),
                np.atleast_1d(action["action.gripper"][i])
            ])
        timing.action_extraction_ms = (time.perf_counter() - t0) * 1000

        # Simulate action execution (sleep)
        t0 = time.perf_counter()
        for _ in range(self.action_horizon):
            time.sleep(self.action_interval)
        timing.action_execution_ms = (time.perf_counter() - t0) * 1000

        timing.total_loop_ms = (time.perf_counter() - loop_start) * 1000
        return timing

    def run_iteration_robot(self, iteration: int) -> TimingBreakdown:
        """Run a single iteration with real robot."""
        if not self.robot:
            raise RuntimeError("Robot not connected")

        timing = TimingBreakdown(iteration=iteration)
        loop_start = time.perf_counter()

        # Camera capture
        t0 = time.perf_counter()
        head_img, wrist_img = self.robot.get_dual_images()
        timing.camera_capture_ms = (time.perf_counter() - t0) * 1000

        # State read
        t0 = time.perf_counter()
        state = self.robot.get_current_state()
        timing.state_read_ms = (time.perf_counter() - t0) * 1000

        # Build observation dict
        t0 = time.perf_counter()
        obs_dict = {
            "video.front": head_img[np.newaxis, :, :, :],
            "video.wrist": wrist_img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": ["pick the red cube from the table"],
        }
        timing.obs_dict_build_ms = (time.perf_counter() - t0) * 1000

        # Model inference
        t0 = time.perf_counter()
        with torch.no_grad():
            action = self.policy.get_action(obs_dict)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing.model_inference_ms = (time.perf_counter() - t0) * 1000

        # Action extraction
        t0 = time.perf_counter()
        action_chunk = np.zeros((self.action_horizon, 6))
        for i in range(self.action_horizon):
            action_chunk[i] = np.concatenate([
                np.atleast_1d(action["action.single_arm"][i]),
                np.atleast_1d(action["action.gripper"][i])
            ])
        timing.action_extraction_ms = (time.perf_counter() - t0) * 1000

        # Action execution (actually send to robot)
        t0 = time.perf_counter()
        for i in range(self.action_horizon):
            self.robot.set_target_state(action_chunk[i])
            time.sleep(self.action_interval)
        timing.action_execution_ms = (time.perf_counter() - t0) * 1000

        timing.total_loop_ms = (time.perf_counter() - loop_start) * 1000
        return timing

    def run_analysis(
        self,
        num_iterations: int = 50,
        use_robot: bool = False
    ) -> Dict:
        """
        Run timing analysis.

        Returns:
            Dictionary with timing statistics
        """
        self.timings = []

        print(f"\n[ANALYSIS] Running {num_iterations} iterations...")
        print(f"[ANALYSIS] Action horizon: {self.action_horizon}")
        print(f"[ANALYSIS] Action interval: {self.action_interval*1000:.1f}ms")
        print(f"[ANALYSIS] Expected execution: {self.action_horizon * self.action_interval * 1000:.1f}ms")

        for i in range(num_iterations):
            if use_robot:
                timing = self.run_iteration_robot(i)
            else:
                timing = self.run_iteration_dummy(i)

            self.timings.append(timing)

            if (i + 1) % 10 == 0:
                print(f"[ANALYSIS] Progress: {i+1}/{num_iterations}")

        # Calculate statistics
        results = self._calculate_statistics()
        return results

    def _calculate_statistics(self) -> Dict:
        """Calculate statistics from collected timings."""
        camera = [t.camera_capture_ms for t in self.timings]
        state = [t.state_read_ms for t in self.timings]
        obs_build = [t.obs_dict_build_ms for t in self.timings]
        inference = [t.model_inference_ms for t in self.timings]
        extraction = [t.action_extraction_ms for t in self.timings]
        execution = [t.action_execution_ms for t in self.timings]
        total = [t.total_loop_ms for t in self.timings]

        # Calculate overhead (total - execution - inference)
        overhead = [t.total_loop_ms - t.action_execution_ms - t.model_inference_ms for t in self.timings]

        return {
            "camera_capture": asdict(calc_stats(camera)),
            "state_read": asdict(calc_stats(state)),
            "obs_dict_build": asdict(calc_stats(obs_build)),
            "model_inference": asdict(calc_stats(inference)),
            "action_extraction": asdict(calc_stats(extraction)),
            "action_execution": asdict(calc_stats(execution)),
            "overhead": asdict(calc_stats(overhead)),
            "total_loop": asdict(calc_stats(total)),
            "config": {
                "action_horizon": self.action_horizon,
                "action_interval_ms": self.action_interval * 1000,
                "denoising_steps": self.denoising_steps,
                "num_iterations": len(self.timings),
            },
            "raw_timings": [asdict(t) for t in self.timings],
        }


def print_results(results: Dict):
    """Print timing analysis results."""
    print("\n" + "="*70)
    print("TIMING ANALYSIS RESULTS")
    print("="*70)

    config = results["config"]
    print(f"\nConfiguration:")
    print(f"  Action horizon:    {config['action_horizon']}")
    print(f"  Action interval:   {config['action_interval_ms']:.1f}ms")
    print(f"  Denoising steps:   {config['denoising_steps']}")
    print(f"  Iterations:        {config['num_iterations']}")

    print(f"\n{'Stage':<25} {'Mean':>10} {'Std':>10} {'P95':>10} {'Max':>10}")
    print("-"*70)

    stages = [
        ("Camera capture", "camera_capture"),
        ("State read", "state_read"),
        ("Obs dict build", "obs_dict_build"),
        ("Model inference", "model_inference"),
        ("Action extraction", "action_extraction"),
        ("Action execution", "action_execution"),
        ("Overhead", "overhead"),
        ("TOTAL LOOP", "total_loop"),
    ]

    for name, key in stages:
        s = results[key]
        prefix = ">>> " if key == "total_loop" else "    "
        print(f"{prefix}{name:<21} {s['mean_ms']:>9.1f}ms {s['std_ms']:>9.1f}ms {s['p95_ms']:>9.1f}ms {s['max_ms']:>9.1f}ms")

    print("="*70)

    # Calculate percentages
    total_mean = results["total_loop"]["mean_ms"]
    inference_mean = results["model_inference"]["mean_ms"]
    execution_mean = results["action_execution"]["mean_ms"]
    overhead_mean = results["overhead"]["mean_ms"]

    print(f"\nTime Breakdown:")
    print(f"  Model inference:  {inference_mean:>6.1f}ms ({inference_mean/total_mean*100:>5.1f}%)")
    print(f"  Action execution: {execution_mean:>6.1f}ms ({execution_mean/total_mean*100:>5.1f}%)")
    print(f"  Overhead:         {overhead_mean:>6.1f}ms ({overhead_mean/total_mean*100:>5.1f}%)")

    # Control frequency
    control_freq = 1000.0 / total_mean
    print(f"\nEffective control frequency: {control_freq:.2f} Hz")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if inference_mean > 200:
        print("  [HIGH] Model inference is slow (>200ms)")
        print("         Consider: TensorRT optimization, fewer denoising steps")

    if results["camera_capture"]["max_ms"] > 100:
        print("  [HIGH] Camera capture has high variance")
        print("         Consider: Frame buffering, USB controller isolation")

    if control_freq < 2.0:
        print("  [WARN] Control frequency is low (<2 Hz)")
        print("         Consider: Reduce action_horizon, reduce action_interval")

    if overhead_mean > 50:
        print("  [WARN] High overhead detected")
        print("         Consider: Profiling for memory allocations, optimize data copies")

    nvidia_expected = 160  # 8 * 20ms
    if total_mean > nvidia_expected * 2:
        print(f"  [INFO] Loop time ({total_mean:.0f}ms) > NVIDIA reference ({nvidia_expected}ms)")
        print("         Consider: action_horizon=8, action_interval=0.02")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="GR00T Inference Pipeline Timing Analysis"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to GR00T checkpoint"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=50,
        help="Number of iterations to run"
    )
    parser.add_argument(
        "--with-robot", action="store_true",
        help="Use real robot instead of dummy inputs"
    )
    parser.add_argument(
        "--port", type=str, default="/dev/ttyACM2",
        help="Robot serial port"
    )
    parser.add_argument(
        "--head-cam-idx", type=int, default=4,
        help="Head camera index"
    )
    parser.add_argument(
        "--wrist-cam-idx", type=int, default=6,
        help="Wrist camera index"
    )
    parser.add_argument(
        "--action-horizon", type=int, default=12,
        help="Number of actions to execute per inference"
    )
    parser.add_argument(
        "--action-interval", type=float, default=0.033,
        help="Time between actions in seconds"
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--output", type=str, default="timing_analysis.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    print("="*70)
    print("GR00T INFERENCE TIMING ANALYSIS")
    print("="*70)

    analyzer = TimingAnalyzer(
        model_path=args.model_path,
        denoising_steps=args.denoising_steps,
        action_horizon=args.action_horizon,
        action_interval=args.action_interval,
    )

    try:
        # Load model
        analyzer.load_model()

        # Connect robot if needed
        if args.with_robot:
            analyzer.connect_robot(
                port=args.port,
                head_cam_idx=args.head_cam_idx,
                wrist_cam_idx=args.wrist_cam_idx,
            )

        # Run analysis
        results = analyzer.run_analysis(
            num_iterations=args.num_iterations,
            use_robot=args.with_robot,
        )

        # Print results
        print_results(results)

        # Save results
        # Remove raw timings for smaller file
        results_summary = {k: v for k, v in results.items() if k != "raw_timings"}
        with open(args.output, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n[INFO] Results saved to: {args.output}")

        # Also save full results with raw timings
        full_output = args.output.replace(".json", "_full.json")
        with open(full_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Full results saved to: {full_output}")

    finally:
        if args.with_robot:
            analyzer.disconnect_robot()

    print("\nTiming analysis complete!")


if __name__ == "__main__":
    main()
