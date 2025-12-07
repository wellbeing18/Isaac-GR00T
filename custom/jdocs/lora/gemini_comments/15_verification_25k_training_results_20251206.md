# Verification of 25K Training Results: The "Wrist Instability" Problem

**Date**: 2025-12-06
**Status**: 🔴 **Regression Detected** (Critical Issue Identified)
**Data Source**: `closedloop_sim_25k.json` vs `closedloop_sim_10k.json`

## 1. Executive Summary

We have verified the 25K checkpoint. While the **Total MSE** decreased (improved), the user's observation of "no clear improvement" is **correct**.

**Diagnosis**: The model has suffered from **Catastrophic Forgetting** on the fine-motor joints (Wrist/Gripper).
-   It significantly improved the "Arm Body" (Shoulder/Elbow) stability.
-   It **broke** the "Wrist Pitch" (Wrist Flex), causing the end-effector to point in the wrong direction.

This explains why the task fails: The robot moves the arm to the correct location (thanks to better shoulder control) but arrives with the gripper at the wrong angle (due to broken wrist control), making grasping impossible.

## 2. Quantitative Analysis: The Trade-off

| Joint | 10K MSE | 25K MSE | Change | Status |
|-------|---------|---------|--------|--------|
| **`shoulder_pan`** | 438.7 | **15.3** | **-96%** 🚀 | **Perfect**. The base is locked in. |
| **`shoulder_lift`** | 299.5 | **14.6** | **-95%** 🚀 | **Perfect**. |
| **`elbow_flex`** | 100.1 | **16.2** | **-84%** 🚀 | **Perfect**. |
| **`wrist_flex`** | 15.7 | **306.1** | **+1850%** 💀 | **CATASTROPHIC FAILURE**. |
| **`wrist_roll`** | 416.2 | 491.1 | +18% ❌ | Still bad, slightly worse. |
| **`gripper`** | 574.5 | 619.4 | +8% ❌ | Still bad. |

### Interpretation
1.  **The "Body" is Fixed**: The model has mastered the transport phase. It knows exactly how to move the arm from A to B.
2.  **The "Hand" is Broken**: The `wrist_flex` (pitch) error explosion (15.7 → 306.1) is the killer. The gripper pitch is fluctuating wildly (~17° error squared), likely causing the gripper to point down/up randomly instead of aligning with the object.

## 3. Why did this happen?
This is a common issue in end-to-end Visuomotor Policy learning:
-   **Loss Domination**: The "large" joints (Shoulder/Elbow) sweep through large arcs, generating large loss values. The optimizer prioritizes reducing these errors first.
-   **Forgetting**: As the model aggressively optimized the shoulder/elbow (reducing error by 95%), it shifted its internal weights away from the features needed for fine wrist control.

## 4. Recommended Action Plan

We cannot deploy the 25K model as-is. We need to recover the wrist performance.

### Option A: Checkpoint Surgery (Quickest)
Since 10K had good `wrist_flex` (15.7) but bad body, and 25K has good body but bad wrist, the "sweet spot" might be in between.
-   **Action**: Check if a checkpoint exists around **15K - 18K steps**.
-   **Hypothesis**: There might be a point where the body improved *enough* but the wrist hadn't broken yet.

### Option B: Resume Training with Focus
We need to force the model to pay attention to the wrist.
-   **Action**: Continue training to **50K steps**.
-   **Reasoning**: Often, after the "Body" converges (loss ~0.02), the optimizer finally switches focus to the "Details" (Wrist). The regression might be temporary.

### Option C: Loss Reweighting (Requires Code Change)
-   **Action**: Modify the training loss function to multiply wrist errors by 10x.
-   **Effort**: High (requires modifying `gr00t` training code).

### Decision
**Proceed with Option B (Train to 50K)**.
The loss curve is still decreasing (0.0229). The model has not fully converged. It is likely in a phase where it prioritized the high-variance joints. Training longer usually allows it to settle the fine joints once the coarse joints are solved.

**Next Step**: Resume training to 50K steps.

