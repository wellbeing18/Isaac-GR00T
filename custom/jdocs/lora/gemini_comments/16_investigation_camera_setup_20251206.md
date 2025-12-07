# Investigation: Camera Setup & Data Quality

**Date**: 2025-12-06
**Topic**: Impact of Camera Placement on Wrist Instability
**Reference**: User Query regarding "Same Side" vs "Opposite Side" setup

## 1. The "Camera Angle" Hypothesis

The user pointed out a critical discrepancy:
-   **Standard GR00T/SO101 Setup**: Camera is **opposite** the arm (facing the front of the robot).
-   **Current Setup**: Camera is on the **same side** as the arm.

### Why this kills performance (The Occlusion Problem)
If the camera is on the same side as the arm (e.g., both on the right):
1.  **Approach Phase**: As the arm reaches for the object, the **forearm and gripper likely block the camera's view** of the object.
2.  **Blind Grasp**: At the most critical moment (just before grasping), the model loses visual lock on the target.
3.  **Result**: The "Body" (shoulder/elbow) gets the hand *near* the target (memorized trajectory), but the "Wrist" flails because it has no visual feedback to correct the final alignment.

This perfectly matches the **25K results**:
-   `shoulder`/`elbow` errors are low (trajectory is correct).
-   `wrist_flex` error is massive (final alignment fails).

## 2. Base Model Bias (The "GR00T Prior")

You are using `GR00T-N1.5-3B` as a base model.
-   This model was pre-trained on terabytes of robot data.
-   If 99% of that data used "Opposite View" or "Center View", the model's visual encoder is highly optimized for those angles.
-   **LoRA Finetuning** can adapt to new views, but it fights against the pre-trained priors. If the view is "Same Side", the features extracted by the vision tower might be confusing the policy head, requiring **much more data** (50k+ steps) or **more demonstration data** to overwrite the prior.

## 3. Revised Action Plan

We should **PAUSE** the blind march to 50K and verify the data first. If the data is fundamentally flawed (occluded), 50K steps won't fix it.

### Step A: Verify Training Data (Immediate)
You need to watch the videos in your dataset (`datasets_groot`).
-   **Check**: In the last 1 second of the video (the grasp), **can you see the object?**
-   **Fail Condition**: If the robot arm completely covers the red cube, the data is invalid. The model cannot learn to grasp what it cannot see.

### Step B: Camera Re-alignment (Recommended)
If Step A shows occlusion:
1.  **Move the Camera**: Place it opposite the arm (Standard Setup).
2.  **Re-collect Data**: Record 50 episodes with the new view.
3.  **Re-train**: Start fresh (or from 5K). This will likely converge much faster than fighting a bad view.

### Step C: 50K Training (Conditional)
Only proceed to 50K if:
1.  You verify that the training data is **NOT occluded**.
2.  The camera view provides a clear line of sight to the gripper jaws and the object throughout the entire trajectory.

