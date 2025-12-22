# Inference Mismatch Investigation Report

## 1. Executive Summary

The investigation into the poor inference performance of the GR00T 1.6 model (checkpoint-45000) identifies **Inference Staleness (Low Control Frequency)** as the primary root cause.

The inference scripts are currently executing the full 16-step prediction horizon (~0.53 seconds) from the model without replanning, despite the intention to reduce this horizon. This causes the robot to execute outdated actions ("open-loop" behavior) for half a second, leading to overshooting ("swinging to far left"), poor recovery from errors, and missed grasps.

## 2. Root Cause Analysis

### A. Inference Control Frequency (Primary Issue)

**Observation:**
The model is configured to predict 16 steps (`ACTION_HORIZON = 16` in `so101_config_1_6.py`).
In `infer_groot_so101_trace.py`, there is a variable `ACTION_HORIZON = 1`, and in `infer_groot_so101_1_6.py`, `ACTION_HORIZON = 8`.
**However, this variable is unused in the action execution loop.**

**Code Evidence (`infer_groot_so101_trace.py`):**
```python
# Lines 370-373
arm_actions = action_dict["single_arm"][0]     # Shape: (16, 5)
gripper_actions = action_dict["gripper"][0]    # Shape: (16, 1)
action_buffer = np.concatenate([arm_actions, gripper_actions], axis=1) # Length: 16
```

The loop continues until `action_idx >= len(action_buffer)` (Line 349).
This means the robot executes **all 16 steps** captured at time $t$ until time $t + 0.53s$.

**Impact:**
- **Overshooting:** If the robot starts moving towards a block, it commits to that trajectory for 0.5s. If it passes the block or needs micro-correction, it cannot react until the buffer finishes.
- **"Swing to far left":** If the initial prediction is slightly biased to the left (or correct but requires stopping), the robot will continue moving left for the full duration, ending up "far left".
- **Grip Timing:** Gripper actions are planned 0.5s in advance. If the arm moves faster/slower than expected, the grip happens at the wrong physical location.

### B. Dataset Augmentation (Secondary Suspect)

The "swing to far left" symptom (moving to where there is no block) strongly suggests a potential issue with the horizontal flip augmentation or a bias in the dataset that the augmentation failed to correct (or exacerbated).

**Code Verification (`augment_dataset_horizontal.py`):**
- **Logic:** Negates `shoulder_pan` and `wrist_roll`. Flips `head`/`wrist` images.
- **Assessment:** This logic is mathematically correct for a standard robot setup where `shoulder_pan` rotates around the vertical Z-axis and the camera is upright.
- **Risk:** If the physical robot's "left" (+Pan) does not correspond to the image's "left" (pixel column < width/2) in the way the flip expects (e.g., if cameras are rotated), this augmentation generates adversarial data (e.g., "See block on right -> Move Left").

However, since the behavior is "swinging" (motion) rather than just "moving wrong direction from start", the **staleness** is the more immediate multiplier of any small directional error.

## 3. Immediate Recommendations

### Fix 1: Implement Closed-Loop Inference (High Priority)

Modify the inference scripts to respect a shorter execution horizon. This forces the model to replan more frequently using fresh observations.

**Changes to `infer_groot_so101_1_6.py` and `infer_groot_so101_trace.py`:**

```python
# In run_inference_loop...

# 1. Define Execution Horizon (e.g., 8 steps = ~240ms, or 1 step = ~30ms for full closed-loop)
EXECUTION_HORIZON = 8  # Set this based on desired latency vs smoothness

# 2. Slice the buffer
action_dict, info = policy.get_action(observation)
arm_actions = action_dict["single_arm"][0]
gripper_actions = action_dict["gripper"][0]
full_buffer = np.concatenate([arm_actions, gripper_actions], axis=1)

# Take only the first N steps
action_buffer = full_buffer[:EXECUTION_HORIZON] 
```

**Recommendation:** Start with `EXECUTION_HORIZON = 4` (approx 133ms latency) to drastically improve reactivity while maintaining some temporal coherence.

### Fix 2: Verify Augmentation via Visualization

Before retraining, verify the augmented data is physically correct.
1. Visualize an augmented episode.
2. Check: When the block is on the **Right** (flipped image), does the arm move **Right**?
   - If the arm moves **Left** in the augmented visualization, the `shoulder_pan` negation is incorrect (or sign convention is opposite).

### Fix 3: Recovery Behavior

If "swinging to far left" persists after fixing staleness, it indicates the model is entering Out-Of-Distribution (OOD) states (far left workspace) where it has no training data to recover.
- **Action:** Add "recovery" data (as suggested in previous reports) or ensure the augmentation covers these edge cases correctly.

## 4. Next Steps

1. **Apply Fix 1** to `infer_groot_so101_trace.py`.
2. **Re-run Inference** with `EXECUTION_HORIZON = 4`.
3. **Analyze New Trace**: Check if the "swing" is dampened and if the robot corrects its path towards the block.


