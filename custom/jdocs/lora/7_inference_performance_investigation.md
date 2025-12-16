# GR00T Inference Performance Investigation

**Date:** 2025-12-13
**Training:** `groot_mvp_lora_20251213_155740026`
**Dataset:** `datasets_copy/left/pick_and_place` (70 episodes, 99,738 frames)

## Problem Statement

After training with good metrics (MAE 3.63°, Acc@10 93.5%), real robot inference shows poor performance - the arm barely moves.

## Evidence from Inference Log

```
Home state: [0, -99, 97.5, 50.5, -1.3, 0.7]

Chunk 0 predictions:
  action[0]:  [0.9, -98, 97.3, 42.4, -3.4, 0.8]   (delta = [0.9, 1, -0.2, -8.1, -2.1, 0.1])
  action[15]: [0.6, -99.8, 98.5, 44.9, -4.5, 0.7] (delta = [0.6, -0.8, 1, -5.6, -3.2, 0])
```

The model is producing very small movements (1-8° per step) instead of the large movements needed.

## Root Cause Analysis

### 1. Distribution Mismatch Between Home Position and Training Data

```
                         shoulder_pan  shoulder_lift  elbow_flex  wrist_flex  wrist_roll  gripper
Inference Home:               0.0        -99.0         97.5        50.5        -1.3       0.7
Training Mean State:         10.6        -42.9         57.4        36.5       -37.8       6.5
Training Mean Action:        10.7        -43.9         55.8        36.2       -37.8       5.5
Episode Start Mean:           7.1        -99.3         99.7        54.2        -1.8       0.7
```

**Key Insight:** The inference home position `[0, -99, 97.5, 50.5, -1.3, 0.7]` is similar to the **episode starting positions** (mean `[7.1, -99.3, 99.7, 54.2, -1.8, 0.7]`), but the **training data mean** position is vastly different.

### 2. Training Data Action Deltas Are Small

```
Training Action Deltas (action - state):
  Mean: [0.08, -0.97, -1.57, -0.34, 0, -1]
  Std:  [1.0,   2.97,  2.87,  1.79, 1.32, 2.29]
```

The training data has mostly **small deltas** (mean ~1° per joint). This is because:
- Human demonstrations are smooth and gradual
- Actions are recorded at 30Hz, so per-step movements are tiny
- Large movements (>5°) only occur in 31.1% of frames

### 3. Model Learned to Predict Small Adjustments

The model correctly learned the training distribution:
- Given a state, predict an action that is ~1-2° different
- This matches the training data where most deltas are small

**BUT** this is problematic because:
- At the home position, the robot needs to move **~40-60°** to reach the working area
- The model only predicts **~1-8°** movements because that's what it learned

### 4. Why Evaluation (MAE) Looked Good

Evaluation uses the **training distribution** where states are already in the working area:
- State: near `[10.6, -42.9, 57.4, 36.5, -37.8, 6.5]`
- Action: near `[10.7, -43.9, 55.8, 36.2, -37.8, 5.5]`
- Delta: ~1° (small, easy to predict)

Real inference starts from a **different distribution** (home position):
- State: `[0, -99, 97.5, 50.5, -1.3, 0.7]`
- Needs to move to: `[10.6, -42.9, 57.4, 36.5, -37.8, 6.5]`
- Required delta: 40-60° total

## The Core Problem: Closed-Loop vs Open-Loop

```
Training (Open-Loop):
  Frame N:   state=[10, -50, 60, 40, -35, 5] → action=[10.5, -51, 58, 39, -36, 4]
  Frame N+1: state=[10.5, -51, 58, 39, -36, 4] → action=[11, -52, 56, 38, -37, 3]
  ...
  The state naturally progresses through the trajectory

Inference (Closed-Loop):
  Step 1: state=[0, -99, 97, 50, -1, 1] → action=[1, -98, 96, 48, -3, 1]
  Step 2: state=[1, -98, 96, 48, -3, 1] → action=[2, -97, 95, 46, -5, 1]  (robot moved slightly)
  ...
  The robot slowly drifts but never reaches the working area
```

The model predicts actions that are **relative to the current state**, but the magnitude is too small to make progress toward the goal.

## Why The Model Predicts Small Movements

1. **Training Signal:** The MSE loss penalizes large deviations from the training action
2. **Training Distribution:** 69% of frames have deltas < 5° on all joints
3. **Normalization:** Actions are normalized to ~[-1, 1] range, so small physical movements = small loss
4. **Task Conditioning:** The task description doesn't encode absolute target positions

## Potential Solutions

### Option A: Use Episode-Start Home Position (Quick Fix)
Set inference home closer to training episode starts:
```python
HOME_POSITION = [7.1, -99.3, 99.7, 54.2, -1.8, 0.7]  # Training episode mean
```
This starts closer to training distribution but doesn't solve the core issue.

### Option B: Train with Diverse Starting Positions
Record demonstrations that start from various positions, including the current home position.

### Option C: Use Action Chunking with Larger Horizon
Currently using 16-step horizon. The model predicts small per-step movements. Could:
- Execute only first action, re-predict more frequently
- Or train with larger action magnitudes

### Option D: Add Goal Conditioning
Instead of just task description, provide explicit goal positions that the model should reach.

### Option E: Train with Delta Actions (Not Absolute)
Train the model to predict `action_delta = action - state` instead of absolute actions. This makes the output magnitude independent of the state.

### Option F: Closed-Loop Training
Fine-tune with actual robot execution, where the model experiences the distribution shift.

## Recommended Next Steps

1. **Quick Test:** Try starting from a position inside the training distribution (shoulder_lift around -50° instead of -99°)

2. **Verify Hypothesis:** Run inference starting from training data state (not home position) to confirm the model works when in-distribution

3. **Long-term Fix:** Collect new training data with varied starting positions OR use delta-action formulation

## Verification Script

```bash
# Run inference from a training-like starting position
python custom/scripts/infer_groot_async.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_155740026/checkpoint-8000 \
    --task "pick up the red cube and place it on the white plate" \
    --denoising-steps 4 \
    --duration 30
    # Manually position arm to [10, -50, 60, 40, -35, 5] before running
```

## Appendix: Training Data Statistics

```
Total frames: 99,738
Total episodes: 70

State Range:
  Min:  [-27.6, -100, -58.9, -74.5, -61.2, 0.5]
  Max:  [48.2, 64.9, 100, 81.1, 4.3, 35.7]
  Mean: [10.6, -42.9, 57.4, 36.5, -37.8, 6.5]

Action Range:
  Min:  [-27.9, -100, -61.8, -75.8, -61.5, 0]
  Max:  [48.3, 64.3, 100, 81.3, 4.5, 35.9]
  Mean: [10.7, -43.9, 55.8, 36.2, -37.8, 5.5]

Delta (action - state):
  Mean: [0.08, -0.97, -1.57, -0.34, 0, -1]
  Std:  [1.0, 2.97, 2.87, 1.79, 1.32, 2.29]
```
