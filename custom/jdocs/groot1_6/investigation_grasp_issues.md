# Investigation Report: Grasp Timing and Vision-Arm Mismatch Issues

**Date:** 2024-12-19
**Related to:** Investigation of inference issues beyond "swing left" bias
**Log analyzed:** `custom/logs/inference_20251219_132217.log`

## Problem Statement

User observed during inference:
1. Robot arm approaches objects but has grasp timing issues
2. Arm appears to grasp in empty space next to the block (vision-arm mismatch)
3. If grasp fails, following actions become random/erratic

## Finding 1: Both Cameras ARE Being Used Properly

**Verified configuration chain:**

| Component | Camera Keys | Status |
|-----------|-------------|--------|
| Inference script | `head`, `wrist` | ✓ Both initialized |
| so101_config_1_6.py | `modality_keys=["head", "wrist"]` | ✓ Both configured |
| Dataset modality.json | `head`, `wrist` | ✓ Both mapped |
| Hardware YAML | head (index 4), wrist (index 6) | ✓ Both connected |

**Evidence from log:**
```
2025-12-19 13:22:22,489 - INFO -   head camera initialized: 640x480
2025-12-19 13:22:22,587 - INFO -   wrist camera initialized: 640x480
```

**Conclusion:** Camera configuration is NOT the issue.

---

## Finding 2: Grasp Positions Are OUT OF TRAINING DISTRIBUTION

### Training Data Grasp Positions (gripper < 5°):

| Joint | Training Range | Training Mean |
|-------|----------------|---------------|
| shoulder_pan | [-45.7°, 31.5°] | -2.6° |
| shoulder_lift | [-100.0°, 58.1°] | -18.6° |
| elbow_flex | [-49.7°, 99.8°] | 28.6° |

### Inference Grasp Attempt Positions:

| Step | shoulder_pan | shoulder_lift | elbow_flex | Issue |
|------|--------------|---------------|------------|-------|
| 720 | -13.1° | **64.8°** | **-70.2°** | shoulder_lift OUT of range, elbow_flex OUT |
| 800 | **56.2°** | -41.4° | 42.1° | shoulder_pan OUT of range |
| 880 | **60.1°** | 23.3° | 12.9° | shoulder_pan OUT of range |
| 960 | **33.5°** | 38.6° | -32.8° | shoulder_pan BARELY out |

**Critical finding:** Most grasp attempts happen when arm is in positions NEVER seen during training grasps!
- shoulder_pan at 56-60° vs training max 31.5°
- shoulder_lift at 64.8° vs training max 58.1°
- elbow_flex at -70.2° vs training min -49.7°

---

## Finding 3: Arm Continues Moving DURING Grasp (Wrong Timing)

### Analysis of Arm Motion During Grasp Attempts:

| Step | Gripper Action | Total Arm Delta | Issue |
|------|----------------|-----------------|-------|
| 0 | 0.9° (close) | **29.0°** | Arm moving during grasp! |
| 720 | 0.2° (close) | **14.3°** | Arm moving during grasp! |
| 800 | 0.8° (close) | **19.4°** | Arm moving during grasp! |
| 880 | 1.7° (close) | **18.3°** | Arm moving during grasp! |
| 960 | 0.3° (close) | **16.3°** | Arm moving during grasp! |

**Training data comparison:**
- Training: 85% of grasps have >30° arm motion during 16-step grasp window
- This means the model learned "grasp while moving" pattern
- BUT during inference, the arm is often in wrong position when grasping

**Root cause:** Model learned to close gripper while arm is still in motion (from training data), but the visual-proprioceptive alignment is poor, causing grasps to miss.

---

## Finding 4: Action Discontinuity (Jumpy Commands)

Between inference steps (every 80 steps = 16 actions × 5 inferences), massive jumps occur:

```
Step 0 -> 80: shoulder_pan jumped 71.3°
Step 0 -> 80: shoulder_lift jumped 155.9°
Step 0 -> 80: elbow_flex jumped 138.1°
Step 720 -> 800: shoulder_pan jumped 76.0°
Step 720 -> 800: shoulder_lift jumped 97.3°
Step 720 -> 800: elbow_flex jumped 107.5°
```

**This indicates:** Model predictions change drastically between inference calls, suggesting:
1. State feedback is causing mode switches
2. Model is uncertain about what to do next
3. Closed-loop accumulation of errors

---

## Finding 5: "Random" Behavior After Failed Grasp

**Observed pattern:**
- Step 720: Grasp attempt at shoulder_pan=-13.1°
- Step 800: Suddenly at shoulder_pan=56.2° (jumped 69°!)
- Step 880: shoulder_pan=60.1° (continued right)
- Step 960: shoulder_pan=33.5° (different direction)

**Explanation:** When grasp fails:
1. Model enters out-of-distribution state (holding nothing when expected to hold object)
2. Visual feedback shows no object in gripper
3. Model has no learned recovery behavior for this scenario
4. Actions become erratic because this state was rarely/never seen in training

---

## Root Cause Summary

| Issue | Root Cause | Severity |
|-------|-----------|----------|
| Grasping in wrong position | Arm reaches positions not seen during training grasps | HIGH |
| Grasp timing | Model learned "grasp while moving" from data | MEDIUM |
| Post-failure randomness | No recovery behavior learned | HIGH |
| Action discontinuity | Closed-loop error accumulation | MEDIUM |

### Primary Root Causes:

1. **Dataset directional bias (76% LEFT)** - Causes arm to drift left, miss objects
2. **Limited grasp position diversity** - Model only saw grasps in narrow shoulder_pan range [-45°, 31°]
3. **No failed grasp examples** - Training only contains successful grasps

---

## Recommended Next Steps (Priority Order)

### 1. Data Augmentation (HIGH PRIORITY - Immediate)
```python
# Mirror horizontally to balance directional bias
# This also expands grasp position range to [-31.5°, 45.7°]
```
- **Impact:** Fixes directional bias + doubles grasp position coverage
- **Effort:** Low (offline augmentation)

### 2. Collect Diverse Grasp Data (HIGH PRIORITY)
- Place blocks at different positions (left, center, right)
- Vary grasp heights and approach angles
- Include some intentional failure scenarios

### 3. Action Smoothing at Inference (MEDIUM - Quick Fix)
```python
# Limit action delta per step to prevent jumps
MAX_DELTA = 5.0  # degrees per step
action = np.clip(action, prev_action - MAX_DELTA, prev_action + MAX_DELTA)
```

### 4. Increase Training Steps (LOW PRIORITY)
- Current: 30k steps
- More training won't fix distribution issues but may help generalization
- Only do this AFTER fixing data issues

### 5. Vision Model Finetuning (NOT RECOMMENDED YET)
- The vision backbone is working - both cameras are being used
- The issue is not vision but arm positioning relative to seen objects
- Save this for later if other fixes don't work

---

## What NOT To Do

1. **Don't adjust camera** - Cameras are working correctly
2. **Don't blindly add more training steps** - Won't fix distribution issues
3. **Don't finetune vision model yet** - Not the root cause

---

## Verification Tests

After implementing fixes, verify with:

```bash
# 1. Check new dataset has balanced directional bias
python custom/scripts/ver1_6/analyze_dataset_bias.py --dataset <new_dataset>

# 2. Check grasp positions cover wider range
python -c "
from pathlib import Path
import pandas as pd
import numpy as np

# Verify grasp position diversity
# Should see shoulder_pan range > 60° (vs current 77°)
"

# 3. Run inference and check:
#    - Grasp attempts at positions within training distribution
#    - Smaller action jumps between inference steps
#    - Better recovery after missed grasps
```

---

## Conclusion

The grasp timing and vision-arm mismatch issues are primarily caused by:

1. **Dataset bias** forcing arm to drift to positions never seen during training grasps
2. **Limited grasp position diversity** in training data
3. **No failure recovery examples** in training

**Most important next step:** Implement horizontal data augmentation to balance directional bias AND expand grasp position coverage. This is a high-impact, low-effort fix that addresses multiple issues simultaneously.
