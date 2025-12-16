# ROOT CAUSE FOUND: Data Collection Timing Issue

## Summary

The inference issue (arms barely moving) is caused by **stationary data at the start of each episode** in the training dataset. This is a **data collection workflow issue**, not a conversion or model issue.

## Key Findings

### 1. Episode Structure Analysis

| Episode | Total Frames | Start Stationary | Active Movement | End Stationary |
|---------|-------------|------------------|-----------------|----------------|
| 0       | 1152        | 234 (7.8s)       | 749 (25.0s)     | 168 (5.6s)     |
| 1       | 1158        | 213 (7.1s)       | 776 (25.9s)     | 168 (5.6s)     |
| 2       | 1263        | 338 (11.3s)      | 752 (25.1s)     | 172 (5.7s)     |
| 3       | 964         | 194 (6.5s)       | 673 (22.4s)     | 96 (3.2s)      |
| 4       | 917         | 163 (5.4s)       | 670 (22.3s)     | 83 (2.8s)      |
| ...     | ...         | ...              | ...             | ...            |

**Averages across episodes:**
- Start stationary period: **7.6 seconds** (~23% of episode wasted)
- Active movement period: **22.2 seconds** (~67% useful data)
- End stationary period: **3.1 seconds** (~10% wasted)

### 2. Why This Happens

Looking at `lerobot-record` behavior (`lerobot/scripts/lerobot_record.py`):

```python
# Line 290-291 - Recording starts IMMEDIATELY
timestamp = 0
start_episode_t = time.perf_counter()
```

When you hear "Recording episode X", the recording has **already started**. The time you spend:
- Positioning the leader arm to starting position
- Mentally preparing
- Moving to the starting pose

...is ALL being recorded as part of the episode!

### 3. Why This Causes Inference Issues

1. **Training learns "don't move"**: The model sees ~23% of frames where the correct action is "stay still"
2. **Open-loop evaluation looks OK**: The first 100-150 frames are used for evaluation - this happens to be the stationary period, so MSE is low!
3. **Real inference fails**: When deployed, the model has learned a strong "stay still" bias

### 4. Why Official Eval Showed Low MSE

The NVIDIA official evaluation (`scripts/eval_policy.py`) uses first 150 steps by default:
- Your episodes have movement starting at frame ~200-300
- First 150 frames are stationary
- Model predicting "don't move" matches ground truth of "not moving"
- = Artificially low MSE!

## Solution

### Option A: Fix Data Collection Workflow (Recommended)

1. **Be in starting position BEFORE episode starts**
   - During reset phase, move leader arm to exact starting position
   - When "Recording episode X" is announced, you should already be ready

2. **Start moving IMMEDIATELY when recording begins**
   - Don't hesitate - the recording has started!

3. **Press RIGHT ARROW as soon as task is complete**
   - Don't wait, this adds useless "holding" data at the end

### Option B: Post-Process Existing Data

Create a script to trim the stationary periods from collected data:

```python
# Pseudocode for trimming
for episode in dataset:
    # Find first significant movement
    first_movement = find_first_movement(episode, threshold=0.5)

    # Find last significant movement
    last_movement = find_last_movement(episode, threshold=0.5)

    # Keep only active portion
    trimmed_episode = episode[first_movement:last_movement+buffer]
```

### Option C: Add "Start" Signal to Collection Script

Modify `collect_xlerobot_data.py` to add a "Press SPACE when ready to start" prompt before actual recording begins.

## Verification

To verify this is the issue, re-train on trimmed data and compare:
1. Trim first 200 frames and last 50 frames from each episode
2. Re-train model
3. Compare inference behavior

## Comparison with Pushpakcc Dataset

Their dataset likely has:
- Shorter or no stationary periods at start
- Operators who started moving immediately when recording began
- Better collection workflow

This explains why their model works while ours shows "flat" predictions.
