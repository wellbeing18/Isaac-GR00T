#!/usr/bin/env python3
"""
Camera Synchronization and Corruption Diagnosis Tool

This script diagnoses camera capture issues during GR00T inference by:
1. Testing camera capture WITHOUT GPU load (baseline)
2. Testing camera capture WITH GPU load (simulating inference)
3. Detecting frame corruption patterns
4. Measuring capture timing variance

Usage:
    # Basic diagnosis (no robot needed)
    python diagnose_camera_sync.py --head-cam-idx 4 --wrist-cam-idx 6

    # With GPU load simulation
    python diagnose_camera_sync.py --head-cam-idx 4 --wrist-cam-idx 6 --with-gpu-load

    # Full diagnosis with model loading
    python diagnose_camera_sync.py --with-model --model-path /path/to/checkpoint
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import threading
import queue

import cv2
import numpy as np

# Try to import torch for GPU load testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class FrameStats:
    """Statistics for a captured frame."""
    timestamp: float
    capture_time_ms: float
    is_corrupt: bool
    corruption_score: float
    black_row_count: int
    mean_brightness: float
    frame_shape: Tuple[int, int, int]


@dataclass
class DiagnosisResult:
    """Results from a diagnosis run."""
    condition: str
    frames_captured: int
    corrupt_frames: int
    corruption_rate: float
    capture_latency_mean_ms: float
    capture_latency_std_ms: float
    capture_latency_max_ms: float
    black_rows_mean: float
    brightness_mean: float
    brightness_std: float
    frame_stats: List[dict]


def detect_frame_corruption(frame: np.ndarray, black_threshold: int = 10) -> Tuple[bool, float, int]:
    """
    Detect if a frame is corrupted (has horizontal black bands).

    Args:
        frame: RGB image (H, W, 3)
        black_threshold: Pixel value below which is considered black

    Returns:
        (is_corrupt, corruption_score, black_row_count)
    """
    if frame is None or frame.size == 0:
        return True, 1.0, frame.shape[0] if frame is not None else 0

    # Convert to grayscale for analysis
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Find rows that are mostly black (potential corruption)
    row_means = np.mean(gray, axis=1)
    black_rows = np.sum(row_means < black_threshold)

    # Calculate corruption score (0-1)
    # Normal images might have a few dark rows, but corruption shows many consecutive black rows
    total_rows = gray.shape[0]
    corruption_score = black_rows / total_rows

    # Consider corrupt if more than 5% of rows are black
    is_corrupt = corruption_score > 0.05

    return is_corrupt, corruption_score, int(black_rows)


def create_gpu_load(duration_sec: float = 10.0, memory_gb: float = 8.0):
    """
    Create GPU load to simulate inference conditions.

    Args:
        duration_sec: How long to run the load
        memory_gb: How much GPU memory to allocate
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        print("[WARN] CUDA not available, skipping GPU load")
        return

    device = torch.device("cuda")
    print(f"[GPU] Allocating {memory_gb}GB on GPU...")

    # Allocate memory
    elements = int(memory_gb * 1024 * 1024 * 1024 / 4)  # float32 = 4 bytes
    tensor = torch.randn(elements, device=device)

    print(f"[GPU] Running matrix operations for {duration_sec}s...")
    start = time.time()
    while time.time() - start < duration_sec:
        # Simulate inference-like operations
        _ = tensor * 2.0 + 1.0
        torch.cuda.synchronize()
        time.sleep(0.01)  # Small delay to not completely block

    del tensor
    torch.cuda.empty_cache()
    print("[GPU] Load complete")


class CameraDiagnostics:
    """Camera diagnostics with corruption detection."""

    def __init__(self, head_cam_idx: int, wrist_cam_idx: int):
        self.head_cam_idx = head_cam_idx
        self.wrist_cam_idx = wrist_cam_idx
        self.head_cap = None
        self.wrist_cap = None

    def connect(self):
        """Connect to cameras."""
        print(f"[CAM] Connecting to head camera (idx={self.head_cam_idx})...")
        self.head_cap = cv2.VideoCapture(self.head_cam_idx)
        if not self.head_cap.isOpened():
            raise RuntimeError(f"Failed to open head camera at index {self.head_cam_idx}")

        # Set camera properties
        self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.head_cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"[CAM] Connecting to wrist camera (idx={self.wrist_cam_idx})...")
        self.wrist_cap = cv2.VideoCapture(self.wrist_cam_idx)
        if not self.wrist_cap.isOpened():
            raise RuntimeError(f"Failed to open wrist camera at index {self.wrist_cam_idx}")

        self.wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.wrist_cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm up cameras
        print("[CAM] Warming up cameras...")
        for _ in range(30):
            self.head_cap.read()
            self.wrist_cap.read()

        print("[CAM] Cameras connected and ready")

    def disconnect(self):
        """Disconnect cameras."""
        if self.head_cap:
            self.head_cap.release()
        if self.wrist_cap:
            self.wrist_cap.release()
        print("[CAM] Cameras disconnected")

    def capture_frame(self, camera: str = "head") -> Tuple[np.ndarray, float]:
        """
        Capture a single frame and measure timing.

        Returns:
            (frame, capture_time_ms)
        """
        cap = self.head_cap if camera == "head" else self.wrist_cap

        start = time.perf_counter()
        ret, frame = cap.read()
        capture_time = (time.perf_counter() - start) * 1000  # ms

        if not ret or frame is None:
            return None, capture_time

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, capture_time

    def run_diagnosis(
        self,
        num_frames: int = 100,
        condition: str = "baseline",
        save_corrupt_frames: bool = True,
        output_dir: str = "diagnose_camera_output"
    ) -> DiagnosisResult:
        """
        Run camera diagnosis and collect statistics.

        Args:
            num_frames: Number of frames to capture
            condition: Description of test condition
            save_corrupt_frames: Save corrupt frames to disk
            output_dir: Directory for saving corrupt frames

        Returns:
            DiagnosisResult with statistics
        """
        if save_corrupt_frames:
            os.makedirs(output_dir, exist_ok=True)

        frame_stats = []
        corrupt_count = 0

        print(f"\n[DIAG] Running diagnosis: {condition}")
        print(f"[DIAG] Capturing {num_frames} frames...")

        for i in range(num_frames):
            # Capture from both cameras
            head_frame, head_time = self.capture_frame("head")
            wrist_frame, wrist_time = self.capture_frame("wrist")

            # Analyze head frame
            if head_frame is not None:
                is_corrupt, score, black_rows = detect_frame_corruption(head_frame)
                brightness = np.mean(head_frame)

                stats = FrameStats(
                    timestamp=time.time(),
                    capture_time_ms=head_time,
                    is_corrupt=is_corrupt,
                    corruption_score=score,
                    black_row_count=black_rows,
                    mean_brightness=float(brightness),
                    frame_shape=head_frame.shape
                )
                frame_stats.append(stats)

                if is_corrupt:
                    corrupt_count += 1
                    if save_corrupt_frames:
                        path = os.path.join(output_dir, f"corrupt_head_{i:04d}.jpg")
                        cv2.imwrite(path, cv2.cvtColor(head_frame, cv2.COLOR_RGB2BGR))

            # Brief delay between frames
            time.sleep(0.033)  # 30 Hz

            if (i + 1) % 20 == 0:
                print(f"[DIAG] Progress: {i+1}/{num_frames} frames, {corrupt_count} corrupt")

        # Calculate statistics
        capture_times = [s.capture_time_ms for s in frame_stats]
        black_rows = [s.black_row_count for s in frame_stats]
        brightness = [s.mean_brightness for s in frame_stats]

        result = DiagnosisResult(
            condition=condition,
            frames_captured=len(frame_stats),
            corrupt_frames=corrupt_count,
            corruption_rate=corrupt_count / len(frame_stats) if frame_stats else 0,
            capture_latency_mean_ms=np.mean(capture_times),
            capture_latency_std_ms=np.std(capture_times),
            capture_latency_max_ms=np.max(capture_times),
            black_rows_mean=np.mean(black_rows),
            brightness_mean=np.mean(brightness),
            brightness_std=np.std(brightness),
            frame_stats=[asdict(s) for s in frame_stats]
        )

        return result


def run_with_gpu_load(
    diagnostics: CameraDiagnostics,
    num_frames: int,
    gpu_memory_gb: float,
    output_dir: str
) -> DiagnosisResult:
    """
    Run diagnosis while GPU is under load.
    """
    result_queue = queue.Queue()

    def capture_task():
        result = diagnostics.run_diagnosis(
            num_frames=num_frames,
            condition=f"with_gpu_load_{gpu_memory_gb}gb",
            save_corrupt_frames=True,
            output_dir=output_dir
        )
        result_queue.put(result)

    def gpu_task():
        create_gpu_load(duration_sec=num_frames * 0.05, memory_gb=gpu_memory_gb)

    # Start both tasks
    capture_thread = threading.Thread(target=capture_task)
    gpu_thread = threading.Thread(target=gpu_task)

    gpu_thread.start()
    time.sleep(1.0)  # Let GPU load stabilize
    capture_thread.start()

    capture_thread.join()
    gpu_thread.join()

    return result_queue.get()


def print_result(result: DiagnosisResult):
    """Print diagnosis result in formatted table."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS RESULT: {result.condition}")
    print(f"{'='*60}")
    print(f"  Frames captured:      {result.frames_captured}")
    print(f"  Corrupt frames:       {result.corrupt_frames}")
    print(f"  Corruption rate:      {result.corruption_rate*100:.1f}%")
    print(f"  ")
    print(f"  Capture latency (ms):")
    print(f"    Mean:               {result.capture_latency_mean_ms:.2f}")
    print(f"    Std:                {result.capture_latency_std_ms:.2f}")
    print(f"    Max:                {result.capture_latency_max_ms:.2f}")
    print(f"  ")
    print(f"  Black rows (mean):    {result.black_rows_mean:.1f}")
    print(f"  Brightness (mean):    {result.brightness_mean:.1f}")
    print(f"  Brightness (std):     {result.brightness_std:.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Camera Synchronization and Corruption Diagnosis"
    )
    parser.add_argument(
        "--head-cam-idx", type=int, default=4,
        help="Head camera device index"
    )
    parser.add_argument(
        "--wrist-cam-idx", type=int, default=6,
        help="Wrist camera device index"
    )
    parser.add_argument(
        "--num-frames", type=int, default=100,
        help="Number of frames to capture per test"
    )
    parser.add_argument(
        "--with-gpu-load", action="store_true",
        help="Test with simulated GPU load"
    )
    parser.add_argument(
        "--gpu-memory-gb", type=float, default=8.0,
        help="GPU memory to allocate for load test (GB)"
    )
    parser.add_argument(
        "--with-model", action="store_true",
        help="Test with actual model inference"
    )
    parser.add_argument(
        "--model-path", type=str,
        help="Path to GR00T checkpoint for model inference test"
    )
    parser.add_argument(
        "--output-dir", type=str, default="diagnose_camera_output",
        help="Directory to save corrupt frames and results"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("CAMERA SYNCHRONIZATION DIAGNOSIS")
    print("="*60)
    print(f"Head camera index:  {args.head_cam_idx}")
    print(f"Wrist camera index: {args.wrist_cam_idx}")
    print(f"Frames per test:    {args.num_frames}")
    print(f"Output directory:   {args.output_dir}")
    print("="*60)

    # Initialize diagnostics
    diagnostics = CameraDiagnostics(args.head_cam_idx, args.wrist_cam_idx)

    all_results = []

    try:
        diagnostics.connect()

        # Test 1: Baseline (no GPU load)
        print("\n[TEST 1] Baseline (no GPU load)")
        result_baseline = diagnostics.run_diagnosis(
            num_frames=args.num_frames,
            condition="baseline_no_gpu",
            save_corrupt_frames=True,
            output_dir=args.output_dir
        )
        print_result(result_baseline)
        all_results.append(asdict(result_baseline))

        # Test 2: With GPU load (if requested)
        if args.with_gpu_load and HAS_TORCH and torch.cuda.is_available():
            print(f"\n[TEST 2] With GPU load ({args.gpu_memory_gb}GB)")
            result_gpu = run_with_gpu_load(
                diagnostics,
                args.num_frames,
                args.gpu_memory_gb,
                args.output_dir
            )
            print_result(result_gpu)
            all_results.append(asdict(result_gpu))

        # Test 3: With actual model inference (if requested)
        if args.with_model and args.model_path:
            print("\n[TEST 3] With model inference")
            print("[INFO] Loading model for inference test...")

            # Import model loading utilities
            sys.path.insert(0, str(Path(__file__).parent))
            from infer_groot_so101 import is_lora_checkpoint, load_groot_with_lora, Gr00tLocalInference

            # Load model
            inference = Gr00tLocalInference(
                model_path=args.model_path,
                denoising_steps=4
            )

            # Run capture with inference
            frame_stats = []
            corrupt_count = 0

            dummy_state = np.zeros(6)

            for i in range(args.num_frames):
                # Capture frame
                head_frame, head_time = diagnostics.capture_frame("head")
                wrist_frame, _ = diagnostics.capture_frame("wrist")

                if head_frame is not None:
                    # Run inference
                    _ = inference.get_action(head_frame, wrist_frame, dummy_state)

                    # Analyze for corruption
                    is_corrupt, score, black_rows = detect_frame_corruption(head_frame)
                    brightness = np.mean(head_frame)

                    stats = FrameStats(
                        timestamp=time.time(),
                        capture_time_ms=head_time,
                        is_corrupt=is_corrupt,
                        corruption_score=score,
                        black_row_count=black_rows,
                        mean_brightness=float(brightness),
                        frame_shape=head_frame.shape
                    )
                    frame_stats.append(stats)

                    if is_corrupt:
                        corrupt_count += 1
                        path = os.path.join(args.output_dir, f"corrupt_model_{i:04d}.jpg")
                        cv2.imwrite(path, cv2.cvtColor(head_frame, cv2.COLOR_RGB2BGR))

                if (i + 1) % 20 == 0:
                    print(f"[DIAG] Progress: {i+1}/{args.num_frames} frames")

            # Calculate statistics
            capture_times = [s.capture_time_ms for s in frame_stats]
            black_rows = [s.black_row_count for s in frame_stats]
            brightness = [s.mean_brightness for s in frame_stats]

            result_model = DiagnosisResult(
                condition="with_model_inference",
                frames_captured=len(frame_stats),
                corrupt_frames=corrupt_count,
                corruption_rate=corrupt_count / len(frame_stats) if frame_stats else 0,
                capture_latency_mean_ms=np.mean(capture_times),
                capture_latency_std_ms=np.std(capture_times),
                capture_latency_max_ms=np.max(capture_times),
                black_rows_mean=np.mean(black_rows),
                brightness_mean=np.mean(brightness),
                brightness_std=np.std(brightness),
                frame_stats=[asdict(s) for s in frame_stats]
            )
            print_result(result_model)
            all_results.append(asdict(result_model))

        # Save all results to JSON
        results_path = os.path.join(args.output_dir, "diagnosis_results.json")
        with open(results_path, "w") as f:
            # Don't include detailed frame_stats in summary
            summary_results = []
            for r in all_results:
                r_copy = r.copy()
                r_copy.pop("frame_stats", None)
                summary_results.append(r_copy)
            json.dump(summary_results, f, indent=2)
        print(f"\n[INFO] Results saved to: {results_path}")

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Condition':<30} {'Corrupt%':>10} {'Latency(ms)':>12} {'Max(ms)':>10}")
        print("-"*62)
        for r in all_results:
            print(f"{r['condition']:<30} {r['corruption_rate']*100:>9.1f}% {r['capture_latency_mean_ms']:>11.1f} {r['capture_latency_max_ms']:>10.1f}")
        print("="*60)

        # Verdict
        print("\n[VERDICT]")
        if len(all_results) >= 2:
            baseline = all_results[0]
            test = all_results[1]

            if test['corruption_rate'] > baseline['corruption_rate'] + 0.05:
                print("  GPU load INCREASES camera corruption!")
                print("  Recommendation: Use separate USB controllers for cameras")
            elif test['capture_latency_max_ms'] > baseline['capture_latency_max_ms'] * 2:
                print("  GPU load INCREASES capture latency variance!")
                print("  Recommendation: Implement frame buffering")
            else:
                print("  No significant difference detected.")
                print("  Camera corruption may have other causes (hardware, cables)")

    finally:
        diagnostics.disconnect()

    print("\nDiagnosis complete!")


if __name__ == "__main__":
    main()
