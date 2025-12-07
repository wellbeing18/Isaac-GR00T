# GR00T SO101 Inference Diagnosis Results

**Date**: 2025-12-06
**Status**: Analysis of Phase 1 Diagnostics
**Source**: `custom/scripts/diagnose_async_throughput.py` logs

---

## 1. Diagnostic Findings (Async Throughput)

The "Phase 1: Latency Profiling" has been completed. The results reveal a **performance bottleneck** that complicates the path to smooth control.

### Key Metrics
*   **Pure Inference Throughput**: **6.86 Hz** (avg 145ms/step)
*   **Async Simulation Rate**: **6.47 Hz**
*   **Latency Stability**: Highly unstable (Std: 198ms, Max: 729ms)

### Analysis
1.  **The Speed Limit**: The GPU can only process ~6.9 frames per second.
    *   NVIDIA's reference target is **50Hz** control (20ms). We are at **~7Hz** (145ms).
    *   Even with perfect async pipelining, we cannot generate actions faster than 7Hz.

2.  **The "Dead Time" Validation**:
    *   Current blocking setup: 145ms inference + 396ms execution = 541ms cycle.
    *   Dead time = 145/541 = **26.8%**.
    *   This confirms our hypothesis: The robot is stopped for >1/4 of the time.

3.  **Async Viability**:
    *   The logs say "Async architecture is VIABLE" (because stale actions < 10%), but also "[SLOW] Pure inference rate... is too slow".
    *   **Interpretation**: Moving to Async will eliminate the 27% dead time, so the robot will move continuously. However, the control updates will still only arrive at ~7Hz. This is better than "stop-and-go", but still far from the 30Hz/50Hz ideal.

---

## 2. Updated Recommendations

Given that the model speed (6.9 Hz) is the hard limit, we need to adjust our strategy.

### Immediate Plan (Software Fix)
We **MUST** implement the **Async Architecture** (`infer_groot_async.py`).
*   **Why**: Even at 7Hz, a continuous 7Hz update is better than a "stop-go-stop-go" 2Hz cycle.
*   **How**:
    *   Producer Thread: Runs model at max speed (~7Hz).
    *   Consumer Thread: Interpolates between the last two received trajectories to generate smooth 30Hz commands for the robot.
    *   This decoupling hides the low inference rate from the robot motors.

### Parallel Plan (Performance Optimization)
The 145ms inference time is quite high for an RTX 5090 (assuming high-end GPU).
1.  **TensorRT**: We need to explore compiling the model to TensorRT (as suggested in `deployment_scripts/`). This could bring us from 7Hz -> 20Hz+.
2.  **Precision**: Ensure we are using `bfloat16` (which seems to be enabled in logs) and not accidentally falling back to float32 in critical paths.

### Next Step
Proceed to implement `custom/scripts/infer_groot_async.py` to solve the "jerkiness" (dead time), accepting that the reaction time will be limited to ~150ms until TensorRT is applied.

