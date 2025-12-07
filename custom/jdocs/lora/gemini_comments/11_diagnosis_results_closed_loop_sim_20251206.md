# Diagnosis Result: Closed-Loop Simulation (Error Accumulation)

**Date**: 2025-12-06
**Status**: ✅ Completed
**Tool**: `custom/scripts/diagnose_closed_loop_sim.py`
**Log File**: `custom/logs/closedloop_sim.json`

## 1. Executive Summary

The closed-loop simulation confirms **severe error accumulation** in the policy. While the model performs acceptably in open-loop evaluation (where it is constantly "reset" to the ground truth state), it fails catastrophically when its own predictions are fed back as state inputs.

**Key Metric**: The Closed-Loop MSE is **19.6x higher** than Open-Loop MSE. A ratio > 5x is considered critical.

This explains the "back and forth" and "random" behavior observed on the real robot:
1. The robot makes a small error.
2. It ends up in a slightly drifted state.
3. The policy (brittle/overfitted) doesn't know how to recover from this drifted state and predicts a wildly wrong action.
4. The error compounds exponentially, leading to 99° of drift in just 50 steps.

---

## 2. Quantitative Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Open-Loop MSE** | **19.5** | Acceptable (Baseline) |
| **Closed-Loop MSE** | **382.0** | **CRITICAL** (High Error) |
| **Error Ratio** | **19.6x** | **Unstable Policy** |
| **Total State Drift** | **99.0°** | ~2° drift per step (Avg) |

### Per-Joint Breakdown (MSE)

| Joint | Open-Loop | Closed-Loop | Degradation |
|-------|-----------|-------------|-------------|
| `shoulder_pan` | 3.0 | **714.7** | **238x** (Worst) |
| `wrist_roll` | 17.3 | **642.7** | 37x |
| `gripper` | 4.4 | **524.0** | 119x |
| `shoulder_lift` | 16.9 | **346.0** | 20x |
| `elbow_flex` | 65.5 | 58.8 | 0.9x (Stable?) |
| `wrist_flex` | 9.7 | 5.9 | 0.6x (Stable?) |

**Analysis**:
- The `shoulder_pan` (base rotation) is the most unstable. A small error here swings the entire arm, changing the visual perspective significantly.
- `wrist_roll` and `gripper` are also highly unstable.

---

## 3. Root Cause Analysis

Since we have already fixed the hardware issues (Camera Corruption, Async Latency), this result isolates the remaining issue to **Policy Robustness**.

The policy is **Open-Loop Overfitted**. It has learned to memorize the trajectory from the exact training states but has not learned the underlying vector field to correct deviations.

**Why does this happen?**
1.  **Insufficient Data/Diversity**: 5K steps might be too few for the model to generalize around the trajectory tube.
2.  **Lack of Temporal Ensembling**: The current inference script executes 12 steps from a single inference. If that one inference is slightly off, the robot executes 12 wrong steps in a row, driving it far off-course before the next inference can try to correct it.
3.  **State-Image Mismatch**: In reality, if the arm drifts, the camera image changes. The policy needs to be robust to this.

---

## 4. Recommended Next Steps

We should attempt **software mitigation** strategies before resorting to long retraining.

### Step 1: Implement Sliding Window Temporal Ensembling (High Priority)
The current `infer_groot_async.py` uses "Chunk Execution" (predict 16, execute 12, discard rest). This is brittle.
We should switch to **Sliding Window Ensembling** (standard for Diffusion Policy/ACT):
- **Horizon**: 16 steps
- **Inference Interval**: Every 8 steps (overlapping)
- **Aggregation**: Average the overlapping predictions.
- **Benefit**: This acts as a low-pass filter on the policy's decisions, smoothing out the "wild" predictions that cause drift.

### Step 2: Increase Denoising Steps
- **Current**: 4 steps (Fast but noisy)
- **Proposed**: 8 or 16 steps.
- **Benefit**: Higher quality action predictions, less variance.

### Step 3: Resume Training (If SW fails)
- If software mitigation doesn't fix the drift, the policy simply needs more training (25k+ steps) or more data.

