# Investigation Report: GR00T LoRA Inference Failure

**Date**: 2025-12-13
**Investigator**: Claude
**Checkpoint**: `/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251212_180908058/best`
**Dataset**: `/home/jrobot/project/XLeRobot/datasets_groot`

## Executive Summary

The model fails at inference because it was trained on **corrupted data**: all episodes loaded frames from Episode 0's video segment due to timestamp/video misalignment. The model learned to ignore vision and act as a blind state-to-action mapper.

## Root Cause

**Video/Action Misalignment** in `datasets_groot`:

1. Videos are **concatenated** into single `file-000.mp4` (~1660s total)
2. Parquet timestamps **reset to 0** for each episode instead of using cumulative offsets
3. Dataloader seeks by timestamp → all episodes get Episode 0's frames

**Evidence:**
```
Episode  0: timestamps 0.00s-38.37s → Video 0-38s    ✅ Correct
Episode 10: timestamps 0.00s-30.27s → Video 0-30s    ❌ Should be ~69-99s
Episode 35: timestamps 0.00s-21.87s → Video 0-22s    ❌ Should be ~127-149s
```

**info.json confirms wrong format:**
```json
"video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4"
```
Should be: `episode_{episode_index:06d}.mp4`

## Why MAE Looks Good

GPT's baseline analysis: `action := state` achieves **1.41° MAE**. Since actions are small increments from state at 30Hz, any model predicting near-state values achieves low MAE without learning the task.

## Source Dataset is Correct

The source dataset has **per-episode videos**:
```
/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place/videos/
  observation.images.head/chunk-000/
    episode_000000.mp4
    episode_000001.mp4
    ...
```

## How Did Corruption Happen?

The current `combine_groot_datasets.py` (lines 453-466) explicitly **rejects** datasets with `file-*.mp4` and requires `episode_*.mp4`. So the corrupted `datasets_groot` likely came from:

1. **An older/incomplete conversion step** that didn't run `split_videos_to_episodes()`
2. **A different combine path** or manual dataset creation
3. **Conversion script failure** that wasn't noticed (video splitting failed silently)

The current scripts have safeguards, but `datasets_groot` predates them or bypassed them.

---

## Fix Plan

### Phase 1: Fix Dataset

**Option A (Recommended)**: Use source dataset directly
```bash
# Source already has per-episode videos
DATASET="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place"
```

**Option B**: Re-run conversion with video splitting
```bash
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path "$SOURCE_DATASET" \
    --robot-type so101 \
    --dual-camera
```

### Phase 2: Verify Dataset Integrity

**Critical**: Before training, verify the dataset loads correctly by simulating the training dataloader.

Create a verification tool that:
1. Loads data exactly as `LeRobotSingleDataset` does during training
2. For each episode, verifies video frames match the episode (not Episode 0)
3. Computes baseline metrics to ensure data is discriminative
4. Provides visual verification

### Phase 3: Retrain

Retrain LoRA on verified dataset. Current checkpoint is invalid.

### Phase 4: Validate

- Use closed-loop simulation tests (not just open-loop MAE)
- Verify model responds to visual changes, not just state

---

## Proposed Verification Tool

Create `custom/scripts/verify_dataset_integrity.py` that:

### 1. Dataloader Simulation
```python
# Load exactly as training does
from gr00t.data.dataset import LeRobotSingleDataset
dataset = LeRobotSingleDataset(dataset_path, modality_configs, ...)

# Sample from multiple episodes
for ep_idx in [0, 10, 35, 69]:
    sample = dataset.get_step_data(ep_idx, step=0)
    # Check video frames are episode-specific
```

### 2. Visual Verification
- Extract first frame from each episode
- Display grid showing frame variations across episodes
- If all frames look identical → data is corrupted

### 3. Quantitative Checks
```python
# Check 1: Frame uniqueness across episodes
frame_hashes = {}
for ep_idx in range(num_episodes):
    frame = get_first_frame(ep_idx)
    h = hash(frame.tobytes())
    if h in frame_hashes:
        print(f"WARNING: Episode {ep_idx} has same frame as {frame_hashes[h]}")
    frame_hashes[h] = ep_idx

# Check 2: Timestamp continuity for consolidated videos
# If using file-000.mp4, timestamps must be cumulative
for ep_idx in range(1, num_episodes):
    prev_end = get_episode_end_timestamp(ep_idx - 1)
    curr_start = get_episode_start_timestamp(ep_idx)
    if curr_start < prev_end:
        print(f"ERROR: Episode {ep_idx} timestamp resets (should be > {prev_end})")

# Check 3: Baseline MAE sanity
# action=state baseline should NOT be close to model MAE
baseline_mae = compute_baseline_mae(dataset)  # ~1.4°
print(f"Baseline MAE (action=state): {baseline_mae:.2f}°")
print("Model must significantly beat this to be useful")
```

### 4. Integration with Existing Visualizer

Extend `custom/tools/dataset_visualizer` to:
- Show which video file each episode loads from
- Highlight if multiple episodes load identical frames
- Add "integrity check" tab with automated verification

---

## Verification Checklist

Before training on any dataset:

- [ ] `info.json` has `video_path` with `episode_{episode_index:06d}.mp4`
- [ ] Per-episode video files exist in `videos/*/chunk-*/`
- [ ] Visual check: frames from Episode 0, 10, 35 look different
- [ ] Quantitative check: frame hash uniqueness across episodes
- [ ] If using consolidated video: timestamps are cumulative, not per-episode
- [ ] Baseline MAE computed and documented

---

## Files to Create/Modify

1. **NEW**: `custom/scripts/verify_dataset_integrity.py` - Pre-training verification
2. **MODIFY**: `custom/tools/dataset_visualizer/` - Add integrity checks
3. **MODIFY**: `custom/scripts/train_groot_mvp.sh` - Add pre-flight dataset check

## Conclusion

The inference failure is a **data corruption issue**, not a model or training issue. The fix requires:
1. Using correctly-formatted dataset with per-episode videos
2. Adding verification tooling to prevent future data issues
3. Retraining on verified data
