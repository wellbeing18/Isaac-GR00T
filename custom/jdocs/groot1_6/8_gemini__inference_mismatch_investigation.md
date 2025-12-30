# Inference Mismatch Investigation Report: Global Blindness & Camera Dominance

## 1. Executive Summary

Analysis of the latest inference traces (`trace_20251222_113045` and `trace_20251222_115129`) confirms that reducing the **Action Horizon to 4** significantly improved the local grasping performance (50% success rate), resolving the "overshoot/swing" issues caused by open-loop staleness.

However, the remaining failure modes—"sitting back" after a task or "doing nothing" when blocks are to the side—indicate a secondary critical issue: **Camera Modality Imbalance**. The model appears to have learned to rely almost exclusively on the **Wrist Camera** for decision making, effectively ignoring the **Head Camera** for global scene understanding and navigation.

## 2. Trace Analysis Findings

### Trace 1: `trace_20251222_113045` ("Sitting Back")
- **Behavior:** Robot picks/places one block, then retracts and stops, despite other blocks being visible in the scene.
- **Data:**
  - **Gripper:** Successfully cycled (0.42 → 29.07), confirming a grasp occurred.
  - **Wrist Roll:** Reached extreme values (`-166.95°`).
- **Diagnosis:** During the "place" or "retract" motion, the wrist camera often ends up pointing away from the table (or is inverted/occluded). In this state, the Wrist Camera sees "nothing" (no blocks).
- **Failure:** Although the Head Camera clearly sees the remaining blocks, the model predicts "Stop/Hover" because its primary sensor (Wrist) sees no target. It fails to switch attention to the Head Camera to initiate a new approach.

### Trace 2: `trace_20251222_115129` ("Doing Nothing")
- **Behavior:** Robot stays centered and does not move towards blocks located to the right.
- **Data:**
  - **Shoulder Pan Range:** Extremely narrow (`-10°` to `22°`). The robot barely moved laterally.
  - **Wrist Roll Range:** Stable (`-3°` to `27°`), camera upright.
- **Diagnosis:** The blocks were likely outside the **Field of View (FOV)** of the Wrist Camera (which has a narrower, focused view).
- **Failure:** The Head Camera (wide global view) definitely saw the blocks to the right. The fact that the robot did not rotate the shoulder (`Pan`) to center the blocks in the Wrist view proves the Head Camera input is not driving the "Approach" policy.

## 3. Root Cause: The "Wrist Dominance" Problem

In Vision-Language-Action (VLA) models, it is common for the model to "latch on" to the sensor that provides the highest correlation with the reward (successful grasp).
- **Wrist Camera:** Provides high-fidelity, occlusion-free views during the critical final centimeters of a grasp.
- **Head Camera:** Provides context but is less useful for the fine manipulation that dominates the dataset's "success" signal.

**The Issue:** The training dataset likely lacks sufficient examples where **only** the Head Camera sees the target and the Wrist Camera does not.
- If most training data starts with the arm already somewhat pointing at the block, the Wrist Camera always has the target in view.
- The model learns: *"If Wrist sees block, move to it. If Wrist sees nothing, stay."*
- It never learned: *"If Wrist sees nothing, check Head. If Head sees block on Right, move Right until Wrist sees it."*

## 4. Recommendations

Since we cannot change code, the solution must be data-driven.

### Fix 1: Collect "Blind Approach" Data (High Priority)
We need to explicitly teach the model to trust the Head Camera for navigation.

**Data Collection Protocol:**
1. **Start Position:** Move the arm to a "neutral" or "far" position where the **Wrist Camera cannot see the blocks** (e.g., pointed up, or far left/right), but the **Head Camera can**.
2. **Action:** Demonstrate moving the arm from this "blind" state towards the block until it enters the Wrist Camera's view.
3. **Transition:** Continue to pick up the block.
4. **Volume:** Add 20-30 such episodes (or ~20% of dataset).

### Fix 2: Verify Augmentation Consistency
Ensure the horizontal flip augmentation is not confusing the Head Camera signal.
- **Check:** Does `flip_video_horizontal(head_cam)` geometrically align with `negate(shoulder_pan)`?
  - If the Head Camera is mounted facing the robot, a block on the robot's Left appears on the image's Right.
  - Flipping the image puts the block on the image's Left.
  - Negating `shoulder_pan` moves the arm to the Right.
  - **Risk:** Ensure the coordinate frames align. If the augmentation teaches "Block on Left of Image -> Move Right", but real life is "Block on Left of Image -> Move Left", the model will learn to ignore the Head Camera to avoid the conflict.
- **Action:** Visualize one augmented episode to confirm: "Augmented Head Image shows block on Right" -> "Augmented Action moves Arm to Right".

### Fix 3: Random Camera Dropout (Training Config)
(If data collection is difficult)
You can force the model to use the Head Camera by randomly "blinding" the Wrist Camera during training.
- **Method:** In the data loader transform, with 10-20% probability, replace the Wrist Camera image with a black frame (or noise) *for the early parts of the trajectory*.
- This forces the gradient to flow through the Head Camera branch to predict the approach velocity.

## 5. Conclusion
The "Inference Mismatch" is no longer a code bug (staleness is fixed). It is now a **Behavioral Blindness** due to dataset distribution. The model is over-fitted to the Wrist Camera. Diversifying the "Approach" phase in the dataset is the required fix.


