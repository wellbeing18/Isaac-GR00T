# GR00T SO101 Inference Diagnosis Results - Phase 4 (Camera Fixed)

**Date**: 2025-12-06
**Status**: Analysis of Inference with MJPEG Cameras
**Source**: User report (Camera issue fixed, motion smooth but trajectory wrong)

---

## 1. Current Status

*   **Hardware**: ✅ Cameras fixed (MJPEG used). ✅ Port contention fixed (Thread lock).
*   **Architecture**: ✅ Async Producer-Consumer working at ~30Hz control loop.
*   **Symptom**: "Arm still didn't finish the task... movement is still not right... not random but... back and forth."

### Interpretation
The transition from "random/panic" (due to black images) to "smooth but wrong" (due to valid images but wrong actions) confirms that the **control loop is healthy**. The problem is now isolated to the **policy's decisions**.

Why does a trained policy fail to complete the task?
1.  **Open-Loop vs Closed-Loop Gap**: The model was trained on perfect expert trajectories. It has never seen "recovery" states. If it drifts slightly off the expert path, it might not know how to correct, leading to oscillation.
2.  **Coordinate/Extrinsic Mismatch**: Did moving the cameras change their position/angle? Even a 1cm shift can break a policy trained without heavy augmentation.
3.  **Task Mismatch**: Is the "red cube" in the same starting distribution as training?

---

## 2. Next Steps (Step 3-5 from Plan)

Yes, we should proceed with the remaining diagnostic steps, as they are designed specifically for this "policy quality" phase.

### Step 3: Closed-Loop Simulation (`diagnose_closed_loop_sim.py`)
*   **Goal**: Determine if the model is inherently unstable or if reality is the problem.
*   **Logic**:
    *   If the model fails in *simulation* (feeding its own predictions back), then the **model itself is weak** (needs more training/data).
    *   If the model succeeds in simulation but fails on the robot, then **Sim-to-Real gap** (camera angle, lighting, calibration) is the cause.

### Step 4: Timing Analysis (`diagnose_timing_analysis.py`)
*   **Status**: Low priority. We already know from async logs that we are hitting 30Hz. We can skip this or run it just to confirm latency is stable.

### Step 5: Real Robot Diagnostic Run
*   **Modification**: We need to record the *actual state trajectory* vs the *commanded trajectory*.
*   **Why**: "Back and forth" can be caused by the robot PID controller fighting the model commands (latency induced oscillation).

---

## 3. Recommendation

**Execute Step 3 (Closed-Loop Simulation) immediately.** This is the fastest way to differentiate between "Bad Model" and "Bad Setup".

```bash
python custom/scripts/diagnose_closed_loop_sim.py \
    --checkpoint /path/to/checkpoint \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot \
    --steps 100 \
    --save-plot closedloop_sim_check.png
```

If simulation shows high drift/error, we need to **train longer** or **add noise augmentation**.
If simulation is perfect, we need to **re-calibrate cameras** or **check robot state normalization**.

