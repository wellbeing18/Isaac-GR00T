# Dataset Quality Comparison: Your Dataset vs Reference

## Executive Summary

Your data collection has **two issues** compared to the reference dataset (youliangtan/so101-table-cleanup):

| Metric | Your Dataset | Reference | Issue |
|--------|-------------|-----------|-------|
| ARM movement starts at | **3.8 seconds** (frame 113) | **0.5 seconds** (frame 16) | **7x slower to start** |
| Movement velocity | **0.30°/frame** | **0.94°/frame** | **3x slower movement** |

## Problem 1: Delayed Start (3.8s vs 0.5s)

### What's Happening
When you start an episode, the recording begins immediately when "Recording episode X" is announced. In the reference dataset, the operator starts moving the arm within **0.5 seconds**. In your dataset, it takes **3.8 seconds** before the arm starts moving.

### Timeline Comparison

**Reference Dataset (youliangtan):**
```
Frame 0   → Recording starts
Frame 16  → ARM movement begins (0.5s)
Frame 855 → Episode ends
```

**Your Dataset:**
```
Frame 0   → Recording starts
Frame 63  → WRIST starts rotating slowly
Frame 113 → ARM movement begins (3.8s)  ← 7x delay!
Frame 900 → Episode ends
```

### Why This Matters
- ~10-15% of each episode is wasted on "getting ready" time
- Model learns that "correct action = don't move" for initial frames
- During inference, model has learned hesitation bias

## Problem 2: Movement Too Slow (0.30 vs 0.94 °/frame)

### What's Happening
Your movements during data collection are **3x slower** than the reference. At 30Hz:
- Each frame is 33ms apart
- Reference moves ~0.94° per frame
- You move ~0.30° per frame

### Why This Matters for 30Hz
At 30Hz, if you move too slowly:
1. Many consecutive frames have nearly identical joint positions
2. Model learns that small/no movement is acceptable
3. During inference, model outputs very small action deltas

### Comparison Visualization
```
Reference (0.94°/frame):
Frame 0:   0° → Frame 1: 0.9° → Frame 2: 1.9° → Frame 3: 2.8° ...
           ↑ Distinct positions, clear trajectory

Your data (0.30°/frame):
Frame 0:   0° → Frame 1: 0.3° → Frame 2: 0.6° → Frame 3: 0.9° ...
           ↑ Very similar positions, subtle changes
```

## Recommendations

### 1. Start Moving Immediately
- During **reset phase**, position the leader arm at the starting position
- When you hear "Recording episode X", you should **already be ready**
- Start the task motion **within 0.5 seconds** of recording start
- Don't adjust wrist or fine-tune position - that's recorded!

### 2. Move Faster (3x faster than current)
Your current speed at 30Hz is too slow. You need to move approximately **3x faster**:

| Movement Type | Current Speed | Target Speed |
|--------------|---------------|--------------|
| Reaching | Very slow | Moderate/deliberate |
| Grasping | Very slow | Steady |
| Lifting | Very slow | Smooth continuous |

### 3. Use Video Comparison Tool
```bash
# View your videos
ffplay "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place/videos/observation.images.head/chunk-000/file-000.mp4"

# View reference videos
ffplay "/tmp/youliangtan_reference/videos/chunk-000/observation.images.top/episode_000000.mp4"
```

Watch both side-by-side to calibrate your movement speed.

## Action Items

1. **Re-collect data** with:
   - Immediate start (< 0.5s delay)
   - 3x faster movement
   - RIGHT ARROW immediately when done

2. **OR Trim existing data:**
   ```bash
   python custom/scripts/trim_stationary_periods.py \
       --input /home/jrobot/project/XLeRobot/datasets/left/pick_and_place \
       --output /home/jrobot/project/XLeRobot/datasets_trimmed/left/pick_and_place \
       --start_buffer 5 \
       --end_buffer 10
   ```

3. **Verify with quality tool:**
   ```bash
   python custom/scripts/verify_dataset_quality.py \
       --dataset /path/to/new/dataset \
       --reference youliangtan/so101-table-cleanup
   ```
