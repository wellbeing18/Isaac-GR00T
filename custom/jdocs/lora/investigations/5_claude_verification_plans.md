# Dataset Verification Tool & Training Readiness

**Date**: 2025-12-13
**Author**: Claude

## Summary

Created `custom/scripts/verify_dataset_integrity.py` - a comprehensive pre-training verification tool that catches data corruption issues BEFORE wasting compute on training.

---

## Core Verification Principle: Training Perspective

**The verification must simulate exactly how training code loads data.** This means:

1. **Use the same dataloader**: `LeRobotSingleDataset` with same config
2. **Use the same sampler**: `BaseSampler` with shuffle=True
3. **Use the same collator**: `DefaultDataCollator`
4. **Use the same transforms**: `data_cfg.transform()`
5. **Test actual batching**: Create real batches, not just single samples

Any verification that doesn't match the training pipeline may miss issues that only appear during actual training.

### Why This Matters

The corrupted dataset passed basic checks:
- ✅ Parquet files exist and load correctly
- ✅ Video files exist
- ✅ Metadata is valid JSON

But FAILED training-perspective checks:
- ❌ Frame uniqueness (simulating dataloader's timestamp-based video seeking)
- ❌ Timestamp alignment for the video format being used

---

## Tool Created

### `custom/scripts/verify_dataset_integrity.py`

**Usage:**
```bash
# Verify the source dataset (VALID - use this for training)
python custom/scripts/verify_dataset_integrity.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place

# With visual report and JSON output
python custom/scripts/verify_dataset_integrity.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place \
    --visual --save-report

# Verify the corrupted dataset (INVALID - do NOT use)
python custom/scripts/verify_dataset_integrity.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot
```

**6 Verification Checks:**

| Check | Purpose | Catches |
|-------|---------|---------|
| 1. Video format | Verify per-episode videos exist | Consolidated video without proper indexing |
| 2. Timestamp alignment | Ensure timestamps match video format | Reset timestamps with consolidated video |
| 3. Frame uniqueness | Different episodes have different frames | All episodes loading same frame (corruption) |
| 4. Baseline MAE | Compute action=state baseline (~1.22°) | Reference for model performance |
| 5. Dataloader simulation | Test `LeRobotSingleDataset.get_step_data()` | Import/loading errors |
| 6. Training simulation | Full pipeline: shuffle, batch, collate | Batching, transform, collation issues |

---

## Test Results

### Source Dataset (Valid)
```
Dataset: /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place

[1/5] Video format:        PASS - Using per-episode video format
[2/5] Timestamp alignment: PASS - Timestamps start near 0 (correct for per-episode)
[3/5] Frame uniqueness:    PASS - All 5 sampled episodes have unique first frames
[4/5] Baseline MAE:        1.22° (reference)
[5/6] Dataloader:          PASS - 5 samples loaded
[6/6] Training simulation: PASS - 3 batches, shuffle works

VERIFICATION PASSED
```

### Corrupted Dataset (Invalid)
```
Dataset: /home/jrobot/project/XLeRobot/datasets_groot

[1/5] Video format:        FAIL - Consolidated video WITHOUT per-episode videos
[2/5] Timestamp alignment: FAIL - Timestamps reset to 0 for episodes [35, 69]
[3/5] Frame uniqueness:    FAIL - DUPLICATE FRAMES: ep14==ep0, ep28==ep0, ep42==ep0, ep56==ep0
[4/5] Baseline MAE:        1.22° (reference)
[5/6] Dataloader:          PASS
[6/6] Training simulation: PASS

VERIFICATION FAILED - 3 critical issues detected
```

---

## Key Findings

### Shuffle Does NOT Cause Misalignment

The training shuffle mechanism works correctly:
1. `BaseSampler` generates permuted indices based on `seed + epoch`
2. Each index maps to a fixed `(trajectory_id, base_index)` pair via `dataset.all_steps[index]`
3. Shuffling permutes WHICH samples are in a batch, not the internal mapping
4. Different epochs produce different orders (verified)

### Root Cause Confirmed

The data corruption is NOT from shuffling. It's from:
1. **Consolidated video** (`file-000.mp4`) containing all episodes concatenated
2. **Per-episode timestamps** starting at 0 instead of cumulative offsets
3. **Dataloader seeks by timestamp** → all episodes get Episode 0's frames

---

## Recommendations

### Immediate: Use Source Dataset

The source dataset is verified and ready:
```bash
DATASET="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place"
```

Update `custom/scripts/train_groot_mvp.sh`:
```bash
# Change from:
DATASET="${DATASET:-/home/jrobot/project/XLeRobot/datasets_groot}"

# To:
DATASET="${DATASET:-/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place}"
```

### Pre-Training Checklist

Before ANY training run:
```bash
# 1. Verify dataset integrity
python custom/scripts/verify_dataset_integrity.py --dataset "$DATASET"

# 2. Ensure all 6 checks pass
# 3. Note baseline MAE (model must beat ~1.22°)
```

### Future: Integrate Into Training Script

Add to `train_groot_mvp.sh`:
```bash
# Pre-flight dataset verification
echo "Verifying dataset integrity..."
python custom/scripts/verify_dataset_integrity.py --dataset "$DATASET"
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset verification failed. Fix issues before training."
    exit 1
fi
```

---

## Files Changed

| File | Change |
|------|--------|
| `custom/scripts/verify_dataset_integrity.py` | **NEW** - Comprehensive verification tool |
| `custom/jdocs/lora/investigations/4_claude_inference_investigation.md` | Investigation report |

---

## Next Steps

1. **Update training script** to use verified source dataset
2. **Add pre-flight check** to training script
3. **Retrain model** on verified data
4. **Re-evaluate** with both open-loop and closed-loop tests
5. **Test inference** on real robot

---

## Technical Details

### Training Data Flow (What We Verify)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BaseSampler (shuffle=True)                                          │
│     │                                                                    │
│     │  Generates shuffled indices: [42, 7, 1003, 28, ...]              │
│     │  ✓ VERIFIED: Different epochs produce different orders            │
│     ▼                                                                    │
│                                                                          │
│  2. Dataset.__getitem__(index)                                          │
│     │                                                                    │
│     │  Maps index → (trajectory_id, base_index) via self.all_steps     │
│     │  ✓ VERIFIED: Mapping is deterministic, shuffle doesn't break it   │
│     ▼                                                                    │
│                                                                          │
│  3. Dataset.get_step_data(trajectory_id, base_index)                    │
│     │                                                                    │
│     │  Loads parquet data for episode                                   │
│     │  Gets timestamp from parquet                                      │
│     │  Calls get_frames_by_timestamps(video_path, timestamps)           │
│     │                                                                    │
│     │  ⚠️ THIS IS WHERE CORRUPTION HAPPENS:                             │
│     │     - Episode 35, timestamp=0.0s                                  │
│     │     - Seeks to 0.0s in file-000.mp4                               │
│     │     - Gets Episode 0's frames instead of Episode 35's!            │
│     │                                                                    │
│     │  ✓ VERIFIED: Frame uniqueness check catches this                  │
│     ▼                                                                    │
│                                                                          │
│  4. Transforms (GR00TTransform)                                         │
│     │                                                                    │
│     │  Normalizes state/action                                          │
│     │  Processes video frames                                           │
│     │  ✓ VERIFIED: Transforms applied correctly                         │
│     ▼                                                                    │
│                                                                          │
│  5. DefaultDataCollator                                                  │
│     │                                                                    │
│     │  Batches samples together                                         │
│     │  Processes images for Eagle model                                 │
│     │  ✓ VERIFIED: Collation works without errors                       │
│     ▼                                                                    │
│                                                                          │
│  6. Model receives batch                                                 │
│     │                                                                    │
│     │  pixel_values, state, action, etc.                                │
│     │  Computes loss                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline Simulation Code

The verification tool simulates the exact training pipeline:

```python
# Same as gr00t/experiment/runner.py
data_cfg = load_data_config("so100_dualcam")
transforms = data_cfg.transform()

dataset = LeRobotSingleDataset(
    dataset_path=str(dataset_path),
    modality_configs=modality_config,
    video_backend="torchvision_av",
    transforms=transforms,
    embodiment_tag="new_embodiment",
)

# Same as gr00t/experiment/trainer.py
sampler = BaseSampler(dataset, shuffle=True, seed=42)
data_collator = DefaultDataCollator()

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    collate_fn=data_collator,
)

# Actually iterate and load batches
for batch in dataloader:
    # Verify batch shapes
    # Verify video variance (not all same frame)
    # Verify shuffle produces different order per epoch
```

### Frame Uniqueness Check

Simulates exactly what the dataloader does:
```python
# For consolidated video with reset timestamps:
# Episode 35, timestamp 0.0s → seeks to frame 0 in file-000.mp4
# This is WRONG - should be frame ~3800 (127s * 30fps)

# The check hashes first frames from multiple episodes
# If hashes match → data is corrupted
```

---

## Conclusion

The verification tool successfully:
1. ✅ Detects the video/timestamp misalignment bug
2. ✅ Confirms source dataset is valid for training
3. ✅ Simulates full training pipeline (shuffle, batch, collate)
4. ✅ Provides baseline MAE for model comparison

**Ready to proceed with training on verified source dataset.**
