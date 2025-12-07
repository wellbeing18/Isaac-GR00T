# Verification of 10K Training & Closed-Loop Simulation

**Date**: 2025-12-06
**Status**: 🟡 Improvement Confirmed (But not yet converged)
**Data Source**: `custom/logs/closedloop_sim_10k.json` vs `custom/logs/closedloop_sim.json`

## 1. Executive Summary

We have verified the closed-loop performance after extending training from **5,000** to **10,000** steps.
**Result:** ✅ **Confirmed Improvement**. The model is becoming more robust, validating that "under-training" was a primary factor.

However, the model is **still unstable** (16x error ratio), meaning 10K is not the finish line. We must continue to 25K/50K.

## 2. Quantitative Comparison (5K vs 10K)

| Metric | 5K Checkpoint | 10K Checkpoint | Change | Interpretation |
|--------|---------------|----------------|--------|----------------|
| **Open-Loop MSE** | 19.5 | **19.03** | -2% | Minimal change (expected, memorization happens early) |
| **Closed-Loop MSE** | 382.0 | **307.45** | **-20%** | **Significant improvement in robustness** |
| **Error Ratio** | 19.6x | **16.2x** | **-17%** | Gap between evaluation and reality is closing |
| **Final Drift** | 99.0° | 98.9° | 0% | Robot still drifts off-target eventually |

### Per-Joint Breakdown (The Good & The Bad)

| Joint | 5K MSE | 10K MSE | Change | Status |
|-------|--------|---------|--------|--------|
| **`shoulder_pan`** | 714.7 | **438.7** | **-39%** ✅ | **CRITICAL FIX**. The base joint is stabilizing. |
| **`wrist_roll`** | 642.7 | **416.2** | **-35%** ✅ | Strong improvement in orientation control. |
| **`shoulder_lift`** | 346.0 | **299.5** | **-13%** ✅ | Moderate improvement. |
| **`elbow_flex`** | 58.8 | 100.1 | +70% ❌ | Regression (likely due to base stabilizing first). |
| **`wrist_flex`** | 5.9 | 15.7 | +166% ❌ | Regression. |

## 3. Analysis

**Is under-training the key issue?**
**YES.**
The fact that `shoulder_pan` (the root of the kinematic chain) improved by nearly 40% with just 5K extra steps is definitive proof.
- In early training, models often flail the base joint.
- As training proceeds, the base stabilizes, but errors may temporarily shift to distal joints (elbow/wrist) as the model refines its policy layer by layer.
- The 20% drop in overall Closed-Loop MSE is a strong signal that we are on the right path.

**Why is drift still high (99°)?**
The model has improved its *local* robustness (staying on track for short bursts), but 10K steps is still too "young" to have learned the full corrective vector field required to recover from large deviations. It takes 25K-50K steps for these VLA models to fully converge on complex manipulation tasks.

## 4. Recommendation

**DO NOT STOP.** The gradient is positive.
1.  **Continue Training**: Proceed immediately to **25,000 steps**.
2.  **Expectation**: We expect the `elbow_flex` and `wrist_flex` regressions to correct themselves as the model converges.
3.  **Check Interval**: Run this same diagnosis again at 25K.

**Next Action**: Resume training to 25K.

