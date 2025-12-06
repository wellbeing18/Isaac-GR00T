#!/usr/bin/env python3
"""
Async Inference Throughput Test

This script tests whether the GPU can sustain inference at a rate that would
support async (pipelined) architecture. If we can achieve >15 Hz inference
throughput, we can decouple inference from execution.

The current blocking architecture causes "stop-and-go" motion:
- Robot moves for ~400ms (12 actions @ 33ms)
- Robot STOPS for ~150ms (inference time)
- This creates 27% "dead time" duty cycle

This test determines if async architecture is feasible by measuring:
1. Pure inference throughput (no robot, continuous inference)
2. Inference + camera capture throughput
3. Sustainable inference rate with realistic workload

Usage:
    # Test inference throughput only
    python diagnose_async_throughput.py --model-path /path/to/checkpoint

    # Test with camera capture
    python diagnose_async_throughput.py --model-path /path/to/checkpoint --with-cameras

    # Full async simulation
    python diagnose_async_throughput.py --model-path /path/to/checkpoint --simulate-async --duration 30
"""

import argparse
import os
import sys
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ThroughputResult:
    """Results from throughput test."""
    condition: str
    duration_sec: float
    total_inferences: int
    throughput_hz: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p95_ms: float


def load_policy(model_path: str, denoising_steps: int = 4):
    """Load GR00T policy."""
    sys.path.insert(0, str(Path(__file__).parent))
    from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora

    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    data_cfg = load_data_config("so100_dualcam")
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    if is_lora_checkpoint(model_path):
        print("[MODEL] Loading LoRA checkpoint...")
        policy = load_groot_with_lora(
            model_path=model_path,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            merge_weights=True,
        )
    else:
        print("[MODEL] Loading full checkpoint...")
        policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
        )

    return policy


def create_dummy_obs():
    """Create dummy observation for testing."""
    return {
        "video.front": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
        "video.wrist": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
        "state.single_arm": np.random.randn(1, 5).astype(np.float64),
        "state.gripper": np.random.randn(1, 1).astype(np.float64),
        "annotation.human.task_description": ["pick the red cube from the table"],
    }


def test_pure_inference_throughput(
    policy,
    duration_sec: float = 10.0,
) -> ThroughputResult:
    """
    Test pure inference throughput (no camera, dummy data).
    This measures the maximum possible inference rate.
    """
    print(f"\n[TEST] Pure inference throughput ({duration_sec}s)...")

    latencies = []
    start_time = time.time()
    count = 0

    while time.time() - start_time < duration_sec:
        obs = create_dummy_obs()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = policy.get_action(obs)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        latencies.append(latency)
        count += 1

        if count % 20 == 0:
            print(f"  {count} inferences, avg latency: {np.mean(latencies[-20:]):.1f}ms")

    elapsed = time.time() - start_time
    latencies = np.array(latencies)

    return ThroughputResult(
        condition="pure_inference",
        duration_sec=elapsed,
        total_inferences=count,
        throughput_hz=count / elapsed,
        latency_mean_ms=float(np.mean(latencies)),
        latency_std_ms=float(np.std(latencies)),
        latency_min_ms=float(np.min(latencies)),
        latency_max_ms=float(np.max(latencies)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
    )


def test_with_camera_throughput(
    policy,
    head_cam_idx: int,
    wrist_cam_idx: int,
    duration_sec: float = 10.0,
) -> ThroughputResult:
    """
    Test inference throughput with real camera capture.
    This measures realistic throughput with I/O.
    """
    import cv2

    print(f"\n[TEST] Inference + camera throughput ({duration_sec}s)...")

    # Connect cameras
    head_cap = cv2.VideoCapture(head_cam_idx)
    wrist_cap = cv2.VideoCapture(wrist_cam_idx)

    if not head_cap.isOpened() or not wrist_cap.isOpened():
        print("[ERROR] Could not open cameras")
        return None

    # Warm up
    for _ in range(10):
        head_cap.read()
        wrist_cap.read()

    latencies = []
    start_time = time.time()
    count = 0

    try:
        while time.time() - start_time < duration_sec:
            t0 = time.perf_counter()

            # Capture
            ret1, head_frame = head_cap.read()
            ret2, wrist_frame = wrist_cap.read()

            if not ret1 or not ret2:
                continue

            # Convert BGR to RGB
            head_rgb = cv2.cvtColor(head_frame, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)

            # Build observation
            obs = {
                "video.front": head_rgb[np.newaxis, :, :, :],
                "video.wrist": wrist_rgb[np.newaxis, :, :, :],
                "state.single_arm": np.random.randn(1, 5).astype(np.float64),
                "state.gripper": np.random.randn(1, 1).astype(np.float64),
                "annotation.human.task_description": ["pick the red cube from the table"],
            }

            # Inference
            with torch.no_grad():
                _ = policy.get_action(obs)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            count += 1

            if count % 20 == 0:
                print(f"  {count} inferences, avg latency: {np.mean(latencies[-20:]):.1f}ms")

    finally:
        head_cap.release()
        wrist_cap.release()

    elapsed = time.time() - start_time
    latencies = np.array(latencies)

    return ThroughputResult(
        condition="inference_with_cameras",
        duration_sec=elapsed,
        total_inferences=count,
        throughput_hz=count / elapsed,
        latency_mean_ms=float(np.mean(latencies)),
        latency_std_ms=float(np.std(latencies)),
        latency_min_ms=float(np.min(latencies)),
        latency_max_ms=float(np.max(latencies)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
    )


def simulate_async_architecture(
    policy,
    duration_sec: float = 30.0,
    execution_hz: float = 30.0,
) -> dict:
    """
    Simulate async producer-consumer architecture.

    Producer thread: Captures images and runs inference as fast as possible
    Consumer thread: Executes actions at fixed rate (simulated)

    This tests if async architecture can provide fresh predictions
    for 30Hz execution.
    """
    print(f"\n[TEST] Async architecture simulation ({duration_sec}s)...")

    action_queue = queue.Queue(maxsize=5)
    stats = {
        "producer_count": 0,
        "consumer_count": 0,
        "queue_drops": 0,
        "stale_actions": 0,
        "producer_latencies": [],
    }
    stop_event = threading.Event()

    def producer():
        """Inference producer thread."""
        while not stop_event.is_set():
            obs = create_dummy_obs()

            t0 = time.perf_counter()
            with torch.no_grad():
                action = policy.get_action(obs)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) * 1000

            stats["producer_latencies"].append(latency)
            stats["producer_count"] += 1

            # Try to put in queue (non-blocking)
            try:
                action_queue.put_nowait({
                    "action": action,
                    "timestamp": time.time(),
                })
            except queue.Full:
                stats["queue_drops"] += 1

    def consumer():
        """Action consumer thread (simulated execution)."""
        interval = 1.0 / execution_hz

        while not stop_event.is_set():
            try:
                item = action_queue.get(timeout=0.1)
                age = time.time() - item["timestamp"]

                if age > 0.1:  # Action older than 100ms is "stale"
                    stats["stale_actions"] += 1

                stats["consumer_count"] += 1

            except queue.Empty:
                pass

            time.sleep(interval)

    # Start threads
    producer_thread = threading.Thread(target=producer, daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)

    producer_thread.start()
    consumer_thread.start()

    # Run for duration
    time.sleep(duration_sec)
    stop_event.set()

    producer_thread.join(timeout=2)
    consumer_thread.join(timeout=2)

    # Calculate results
    latencies = np.array(stats["producer_latencies"])

    return {
        "duration_sec": duration_sec,
        "producer_count": stats["producer_count"],
        "producer_hz": stats["producer_count"] / duration_sec,
        "consumer_count": stats["consumer_count"],
        "consumer_hz": stats["consumer_count"] / duration_sec,
        "queue_drops": stats["queue_drops"],
        "stale_actions": stats["stale_actions"],
        "stale_rate": stats["stale_actions"] / max(stats["consumer_count"], 1),
        "latency_mean_ms": float(np.mean(latencies)) if len(latencies) > 0 else 0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0,
    }


def print_result(result: ThroughputResult):
    """Print throughput result."""
    print(f"\n{'='*60}")
    print(f"RESULT: {result.condition}")
    print(f"{'='*60}")
    print(f"  Duration:      {result.duration_sec:.1f}s")
    print(f"  Inferences:    {result.total_inferences}")
    print(f"  Throughput:    {result.throughput_hz:.2f} Hz")
    print(f"  ")
    print(f"  Latency (ms):")
    print(f"    Mean:        {result.latency_mean_ms:.1f}")
    print(f"    Std:         {result.latency_std_ms:.1f}")
    print(f"    Min:         {result.latency_min_ms:.1f}")
    print(f"    Max:         {result.latency_max_ms:.1f}")
    print(f"    P95:         {result.latency_p95_ms:.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Async Inference Throughput Test"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to GR00T checkpoint"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Test duration in seconds"
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--with-cameras", action="store_true",
        help="Test with real camera capture"
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
        "--simulate-async", action="store_true",
        help="Simulate async producer-consumer architecture"
    )
    parser.add_argument(
        "--output", type=str, default="async_throughput_results.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    print("="*60)
    print("ASYNC INFERENCE THROUGHPUT TEST")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Duration: {args.duration}s")
    print(f"Denoising steps: {args.denoising_steps}")
    print("="*60)

    # Load model
    print("\n[LOAD] Loading model...")
    policy = load_policy(args.model_path, args.denoising_steps)

    # Warmup
    print("[WARMUP] Running 10 warmup inferences...")
    for _ in range(10):
        obs = create_dummy_obs()
        with torch.no_grad():
            _ = policy.get_action(obs)

    results = {}

    # Test 1: Pure inference throughput
    result1 = test_pure_inference_throughput(policy, args.duration)
    print_result(result1)
    results["pure_inference"] = {
        "throughput_hz": result1.throughput_hz,
        "latency_mean_ms": result1.latency_mean_ms,
        "latency_p95_ms": result1.latency_p95_ms,
    }

    # Test 2: With cameras (if requested)
    if args.with_cameras:
        result2 = test_with_camera_throughput(
            policy, args.head_cam_idx, args.wrist_cam_idx, args.duration
        )
        if result2:
            print_result(result2)
            results["with_cameras"] = {
                "throughput_hz": result2.throughput_hz,
                "latency_mean_ms": result2.latency_mean_ms,
                "latency_p95_ms": result2.latency_p95_ms,
            }

    # Test 3: Async simulation (if requested)
    if args.simulate_async:
        async_result = simulate_async_architecture(policy, args.duration)
        results["async_simulation"] = async_result

        print(f"\n{'='*60}")
        print("ASYNC SIMULATION RESULT")
        print(f"{'='*60}")
        print(f"  Producer rate:   {async_result['producer_hz']:.2f} Hz")
        print(f"  Consumer rate:   {async_result['consumer_hz']:.2f} Hz")
        print(f"  Queue drops:     {async_result['queue_drops']}")
        print(f"  Stale actions:   {async_result['stale_actions']} ({async_result['stale_rate']*100:.1f}%)")
        print(f"  Latency (mean):  {async_result['latency_mean_ms']:.1f}ms")
        print(f"{'='*60}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*60)

    pure_hz = results["pure_inference"]["throughput_hz"]

    if pure_hz >= 15:
        print(f"  [GOOD] Pure inference rate ({pure_hz:.1f} Hz) supports async architecture")
        print(f"         Async can provide fresh predictions for 30Hz execution")
        print(f"         Recommended: Implement async producer-consumer pattern")
    elif pure_hz >= 8:
        print(f"  [OK] Pure inference rate ({pure_hz:.1f} Hz) is marginal")
        print(f"       Async may work with 8-action chunks (160ms execution)")
        print(f"       Consider: TensorRT optimization to improve throughput")
    else:
        print(f"  [SLOW] Pure inference rate ({pure_hz:.1f} Hz) is too slow")
        print(f"         Async architecture may not help significantly")
        print(f"         Required: Model optimization (TensorRT, quantization)")

    # Calculate current dead time
    inference_ms = results["pure_inference"]["latency_mean_ms"]
    current_execution_ms = 12 * 33  # 12 actions @ 33ms
    dead_time_pct = inference_ms / (inference_ms + current_execution_ms) * 100

    print(f"\n  Current blocking architecture:")
    print(f"    Inference time:  {inference_ms:.0f}ms")
    print(f"    Execution time:  {current_execution_ms}ms")
    print(f"    Dead time:       {dead_time_pct:.1f}% (robot idle during inference)")

    if args.simulate_async and "async_simulation" in results:
        async_res = results["async_simulation"]
        if async_res["stale_rate"] < 0.1:
            print(f"\n  Async simulation shows <10% stale actions")
            print(f"  Async architecture is VIABLE for this model")
        else:
            print(f"\n  Async simulation shows {async_res['stale_rate']*100:.0f}% stale actions")
            print(f"  Need faster inference or smaller execution chunks")

    print("="*60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
