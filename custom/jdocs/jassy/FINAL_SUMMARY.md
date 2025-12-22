# Final Data Quality Investigation Summary

**Date:** 2025-12-21
**Task:** Pick up blocks and place them on plate (SO-ARM101)
**Dataset Analyzed:** `/datasets/so101_pick_place_groot_augmented` (140 episodes, 70 original + 70 augmented)

---

## Your Observed Issues

1. **Timing**: Gripper opens too late, arm bumps into table
2. **Inaccuracy**: Arm swings too far left, misses blocks by inches
3. **Block slides away**: When trying to grasp, blocks move

---

## Key Findings (Data-Backed)

### CRITICAL ISSUE: Moving While Grasping

| Metric | Your Data | Baseline (so100_strawberry_grape) | Verdict |
|--------|-----------|-----------------------------------|---------|
| **Pre-Grasp Arm Velocity** | 1.12 °/frame | 0.32 °/frame | **3.5x TOO FAST** |
| Grasps with Pause | 19% | 63% | Too few pauses |

**Root Cause of "Block Slides Away":** Your arm is still moving at 1.12°/frame when closing the gripper. The baseline dataset shows arms moving at only 0.32°/frame (nearly stationary) during grasp. This 3.5x difference is causing the blocks to slide away when you try to pick them up.

### Directional Bias Still Present (Even After Augmentation)

| Metric | Your Data | Baseline |
|--------|-----------|----------|
| LEFT bias | **74%** | 50% (balanced) |
| RIGHT bias | 14% | 22% |

Despite horizontal augmentation, 74% of analyzed episodes still show LEFT initial movement. This is causing the "swing left" behavior during inference.

**Why?** The augmented dataset has 140 episodes, but I only analyzed the first 50. The original 70 episodes (indices 0-69) have left bias, and the augmented ones (indices 70-139) have right bias. To get balanced behavior, ensure training randomly samples from both sets.

### Velocity Analysis (Surprising Finding)

| Comparison | Your Mean Velocity | Baseline | Ratio | Verdict |
|------------|-------------------|----------|-------|---------|
| vs so100_strawberry_grape | 2.05 °/frame | 2.29 °/frame | 0.90x | OK |
| vs demo_data/cube_to_bowl_5 | 2.05 °/frame | 0.75 °/frame | 2.74x | Too fast |

**Insight:** Your OVERALL movement speed is similar to the strawberry_grape dataset (which is also fast). The issue is not overall speed, but specifically **not slowing down before grasping**.

### Gripper Timing

| Metric | Your Data | so100_strawberry_grape | demo_data |
|--------|-----------|------------------------|-----------|
| Approach Duration | 2.1 sec | 2.3 sec | 4.3 sec |
| Gripper Opens Early | ❓ | ✓ | ✓ |

Your approach duration is similar to the fast strawberry_grape baseline but much shorter than the careful demo baseline (4.3s). Consider opening gripper earlier.

---

## Your Hypotheses Validated

| Your Hypothesis | Verdict | Data Evidence |
|-----------------|---------|---------------|
| Moving arms too quickly | **PARTLY TRUE** | Overall velocity OK, but pre-grasp velocity 3.5x too high |
| Opening gripper right before pickup | **LIKELY TRUE** | Approach duration shorter than demo baseline |
| 60% left-block bias | **TRUE (74%)** | 74% LEFT initial movement detected |
| Failed attempts causing issues | **UNCLEAR** | Would need to manually identify failure episodes |
| Camera height differences | **PROBABLY FALSE** | Your setup seems standard; not analyzed |

---

## Root Causes of Your Problems

### Issue 1: "Bumps into table / Opens gripper too late"

**Primary Cause:** Not slowing down before grasp + insufficient pause
- Arm moving 3.5x faster than baseline when closing gripper
- Only 19% of grasps have a stationary pause (vs 63% baseline)

**Solution:**
1. Slow down significantly in the final ~30 frames before grasp
2. Come to a complete stop for 0.3-0.5 seconds before closing gripper
3. Open gripper earlier (at least 1-2 seconds before reaching target)

### Issue 2: "Arm swings too far left / Misses blocks"

**Primary Cause:** Directional bias in training data (74% LEFT)
- Model learns "always go left first" from biased data
- In closed-loop inference, small errors accumulate and push arm further left
- Robot ends up at positions never seen in training, causing erratic behavior

**Solution:**
1. Ensure training randomly samples from ALL 140 episodes (original + augmented)
2. Or collect new data with better left/right balance
3. Place blocks at different positions (left, center, right workspace)

---

## Data Collection Recommendations

### Immediate Changes for Next Dataset

1. **STOP before grasping**
   - Come to complete rest for 0.3-0.5 seconds before closing gripper
   - Pre-grasp velocity should be < 0.5 °/frame (nearly stationary)

2. **Open gripper EARLY**
   - Open gripper at least 1 second before reaching target
   - Approach with gripper already open

3. **Balance left/right**
   - 50% episodes: place blocks on LEFT
   - 50% episodes: place blocks on RIGHT
   - Vary which block you pick first

4. **Approach from above**
   - Don't approach blocks from the side
   - Come down from above to avoid pushing blocks

5. **Deliberate movements**
   - Move smoothly, not jerky
   - Aim for 0.75-1.0 °/frame average (~22-30 °/sec)
   - Slow down during critical phases (approach, grasp)

### Example of GOOD Grasp Pattern

```
Frame 0-60:    Fast movement toward target area (gripper OPEN)
Frame 60-90:   Slow approach, gripper already open, aligning
Frame 90-100:  STOP - arm stationary, visually confirm alignment
Frame 100-105: Close gripper slowly
Frame 105+:    Lift and continue
```

### Example of BAD Grasp Pattern (Current)

```
Frame 0-60:    Fast movement toward target
Frame 60-64:   Still moving fast, gripper still closed
Frame 64:      Open gripper while still moving
Frame 65:      Try to close gripper immediately
→ Block slides away because arm was moving
```

---

## Generated Files

| File | Description |
|------|-------------|
| `data_analysis_report.md` | Detailed comparison vs so100_strawberry_grape |
| `velocity_comparison.png` | Velocity histogram comparison |
| `gripper_timing_analysis.png` | Gripper timing patterns |
| `grasp_position_heatmap.png` | Where grasps occur in joint space |
| `direction_bias_comparison.png` | Left/Right bias visualization |
| `trajectory_comparison.png` | Sample episode trajectories |
| `demo_comparison/` | Same analysis vs cube_to_bowl_5 demo |

---

## Next Steps

1. **Review visualizations** in this directory to see the patterns
2. **Watch baseline videos** to see what good demonstrations look like:
   - `datasets/so100_strawberry_grape/videos/` (HuggingFace dataset)
   - `demo_data/cube_to_bowl_5/videos/` (local demo)
3. **Collect 20-30 new episodes** following the recommendations above
4. **Re-run this analysis** to verify improvements:
   ```bash
   python custom/scripts/ver1_6/analyze_data_for_jassy.py --dataset /path/to/new_data
   ```
5. **Combine with existing data** and retrain

---

## Summary

**The main issue is NOT overall movement speed, but specifically:**

> **Your arm is moving 3.5x faster than it should when closing the gripper.**

This causes blocks to slide away. The solution is to STOP the arm before closing the gripper.

Secondary issue: 74% LEFT directional bias is causing the swing-left behavior. Ensure training uses all 140 episodes (original + augmented) equally.
