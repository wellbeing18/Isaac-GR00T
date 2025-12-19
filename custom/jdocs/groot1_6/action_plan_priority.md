# Combined Action Plan: Inference Issues Fix Priority

**Date:** 2024-12-19
**Based on:** `investigation_swing_left_bias.md`, `investigation_grasp_issues.md`

---

## Executive Summary

Two investigations identified the same root cause for multiple inference failures:

| Issue | Root Cause |
|-------|-----------|
| Swing left at start | 76% LEFT directional bias in dataset |
| Grasp in wrong position | Arm drifts to positions never seen during training grasps |
| Post-failure random behavior | No recovery examples in training |
| Action discontinuity | Closed-loop error accumulation from out-of-distribution states |

**Single highest-impact fix:** Horizontal data augmentation addresses ALL these issues simultaneously.

---

## Priority 1: Horizontal Data Augmentation (CRITICAL)

**Impact:** HIGH | **Effort:** LOW | **Do First**

### What it fixes:
1. ✅ Directional bias (76% LEFT → 50% balanced)
2. ✅ Grasp position coverage (doubles shoulder_pan range)
3. ✅ Out-of-distribution states (expands training distribution)

### Implementation:
```python
def mirror_episode(episode):
    """Mirror episode horizontally to balance left/right bias."""
    mirrored = copy.deepcopy(episode)

    # Flip joint angles that relate to horizontal direction
    mirrored['action'][:, 0] *= -1  # shoulder_pan
    mirrored['action'][:, 4] *= -1  # wrist_roll
    mirrored['state'][:, 0] *= -1
    mirrored['state'][:, 4] *= -1

    # Flip camera images horizontally
    for cam_key in ['head', 'wrist']:
        mirrored['video'][cam_key] = np.flip(episode['video'][cam_key], axis=-2)  # flip width

    return mirrored
```

### Expected results after augmentation:
- Directional bias: 50% LEFT / 50% RIGHT
- shoulder_pan grasp range: [-45.7°, 45.7°] (was [-45.7°, 31.5°])
- Effective dataset size: 140 episodes (was 70)

### Verification:
```bash
python custom/scripts/ver1_6/analyze_dataset_bias.py --dataset <augmented_dataset>
# Should show: ~50% LEFT, ~50% RIGHT
```

---

## Priority 2: Action Clipping at Inference (QUICK FIX)

**Impact:** MEDIUM | **Effort:** VERY LOW | **Immediate safety net**

### What it fixes:
- Prevents arm from going out-of-distribution during inference
- Acts as safety bounds while training on augmented data

### Implementation in inference script:
```python
# Add to infer_groot_so101_1_6.py after getting action
JOINT_BOUNDS = {
    'shoulder_pan': (-50.0, 70.0),   # Training range + margin
    'shoulder_lift': (-100.0, 80.0),
    'elbow_flex': (-90.0, 100.0),
    'wrist_flex': (-50.0, 100.0),
    'wrist_roll': (-80.0, 20.0),
    'gripper': (0.0, 45.0),
}

for i, (joint, (low, high)) in enumerate(JOINT_BOUNDS.items()):
    action[i] = np.clip(action[i], low, high)
```

### Why do this now:
- Takes 5 minutes to implement
- Prevents dangerous out-of-bounds movements
- Buys time while preparing augmented dataset

---

## Priority 3: Collect More Diverse Data (MEDIUM TERM)

**Impact:** HIGH | **Effort:** HIGH | **After augmentation shows improvement**

### What to collect:
1. **Block positions:** Left, center, AND right sides
2. **Starting positions:** Vary initial arm configurations
3. **Approach angles:** Multiple grasp approach directions
4. **Failure recovery:** Intentionally include some failed grasps and recovery attempts

### Data collection checklist:
- [ ] 20+ episodes with blocks on RIGHT side
- [ ] 20+ episodes with blocks in CENTER
- [ ] 10+ episodes with intentional grasp failures + recovery
- [ ] Vary starting shoulder_pan between -30° and +30°

### Verification:
```bash
python custom/scripts/ver1_6/analyze_dataset_bias.py --dataset <new_dataset>
# Target: 30-40% LEFT, 20-30% CENTER, 30-40% RIGHT
```

---

## Priority 4: Action Smoothing (OPTIONAL)

**Impact:** LOW-MEDIUM | **Effort:** LOW | **Only if jumps persist after P1-P3**

### What it fixes:
- Reduces sudden action jumps between inference steps
- Makes motion more fluid

### Implementation:
```python
MAX_DELTA_PER_STEP = 5.0  # degrees

def smooth_action(current_action, prev_action, max_delta=MAX_DELTA_PER_STEP):
    delta = current_action - prev_action
    clipped_delta = np.clip(delta, -max_delta, max_delta)
    return prev_action + clipped_delta
```

### When to implement:
- Only if action jumps persist after training on augmented/diverse data
- May mask underlying issues if done too early

---

## Priority 5: More Training Steps (LOW PRIORITY)

**Impact:** LOW | **Effort:** LOW | **Only AFTER data fixes**

### Current status:
- 30k steps with current (biased) data
- Open-loop evaluation looks good
- Problem is data distribution, not training duration

### When to do this:
- AFTER Priority 1 (augmentation) is complete
- AFTER verifying augmented data has balanced distribution
- Consider 50k-100k steps on augmented dataset

### Why NOT to do this now:
- More training on biased data will NOT fix the bias
- Will just reinforce "go left" behavior more strongly

---

## DO NOT DO (Yet)

| Action | Why Not |
|--------|---------|
| Finetune vision model | Cameras work fine, issue is arm positioning |
| Adjust camera position | Both cameras initialized correctly |
| Change action representation | Not the root cause |
| Add more training steps (without fixing data) | Will reinforce bad patterns |

---

## Implementation Timeline

```
Week 1:
├── Day 1-2: Implement Priority 2 (action clipping) - immediate safety
├── Day 2-3: Implement Priority 1 (data augmentation script)
├── Day 3-4: Generate augmented dataset
├── Day 4-5: Train on augmented dataset (30k steps)
└── Day 5-7: Test and verify improvements

Week 2 (if needed):
├── Day 1-3: Collect diverse data (Priority 3)
├── Day 3-5: Combine with augmented data
└── Day 5-7: Retrain and test
```

---

## Success Metrics

After implementing fixes, expect:

| Metric | Before | Target |
|--------|--------|--------|
| Directional bias | 76% LEFT | ~50% balanced |
| Grasp success rate | Low | >60% |
| Out-of-distribution states | Frequent | Rare |
| Action jumps >50° | Common | None |
| Post-failure recovery | Random | Purposeful |

### How to verify:
```bash
# 1. Check dataset balance
python custom/scripts/ver1_6/analyze_dataset_bias.py --dataset <dataset>

# 2. Run inference and check log
python custom/scripts/ver1_6/infer_groot_so101_1_6.py --checkpoint <new_checkpoint>

# 3. Analyze inference log
python /tmp/analyze_inference_issues.py  # Check for:
#    - No grasp attempts at shoulder_pan > 45° or < -45°
#    - Reduced action jumps
#    - Balanced left/right movements
```

---

## Summary: Ordered Priority List

| # | Action | Impact | Effort | When |
|---|--------|--------|--------|------|
| **1** | **Horizontal data augmentation** | **HIGH** | **LOW** | **NOW** |
| 2 | Action clipping (inference) | MEDIUM | VERY LOW | NOW (safety) |
| 3 | Collect diverse data | HIGH | HIGH | After P1 shows improvement |
| 4 | Action smoothing | LOW-MED | LOW | Only if needed |
| 5 | More training steps | LOW | LOW | After data fixes |

**Bottom line:** Implement horizontal data augmentation FIRST. It's the highest-impact, lowest-effort fix that addresses multiple issues at once.
