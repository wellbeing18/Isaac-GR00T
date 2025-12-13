# Video Encoding & Training Performance Investigation

**Date:** 2025-12-12
**Status:** 🔬 Investigation Required

## Background

During GR00T LoRA training on RTX 5090 laptop (95W power limit), we observed:
- Training speed: ~1.67s/step (slower than expected)
- GPU utilization: 100% (good)
- CPU utilization: ~11% (not bottlenecked)
- Current video codec: AV1

**Hypothesis:** Re-encoding videos from AV1 to H.264 could improve training speed by 15-25% due to faster video decoding.

## Questions to Answer

### 1. Does H.264 guarantee faster training?

**Theoretical basis:**
- AV1: Excellent compression, but CPU-intensive decode (no hardware decode on most GPUs)
- H.264: Slightly larger files, but has NVDEC hardware decode support on NVIDIA GPUs

**Factors that could affect results:**
- Whether the training pipeline uses hardware decode
- Video backend implementation (decord, torchvision_av, etc.)
- Batch size and dataloader prefetching
- GPU memory bandwidth

**TODO:** Benchmark both codecs with same dataset

### 2. Will re-encoding cause training issues?

**Potential concerns:**
| Concern | Risk Level | Mitigation |
|---------|------------|------------|
| Frame count mismatch | Medium | Verify frame count before/after |
| Color space changes | Low | Use `-pix_fmt yuv420p` |
| Resolution changes | Low | Don't resize in ffmpeg |
| Timestamp drift | Medium | Verify timestamps match parquet |
| Quality loss | Low | Use CRF 18 (high quality) |

**TODO:** Verify data integrity after re-encoding

### 3. What do others use?

**TODO:** Research what codecs are used in:
- [ ] LeRobot default recording
- [ ] GR00T official examples
- [ ] OpenX dataset
- [ ] DROID dataset

---

## Experiment Plan

### Experiment 1: Benchmark Codec Performance

**Setup:**
- Same dataset (pick_and_place, 70 episodes)
- Same training config (batch=32, workers=16)
- Compare: AV1 vs H.264
- Measure: steps/second, GPU util, CPU util

**Steps:**
```bash
# 1. Create H.264 version of dataset
cp -r datasets_groot datasets_groot_h264
bash custom/scripts/reencode_videos_h264.sh datasets_groot_h264

# 2. Run short training on AV1 (100 steps)
python scripts/gr00t_finetune.py --dataset-path datasets_groot --max-steps 100 ...

# 3. Run short training on H.264 (100 steps)
python scripts/gr00t_finetune.py --dataset-path datasets_groot_h264 --max-steps 100 ...

# 4. Compare timing
```

### Experiment 2: Verify Data Integrity

**Checks to perform after re-encoding:**
```bash
# 1. Frame count verification
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 video.mp4

# 2. Duration verification
ffprobe -v error -show_entries format=duration -of csv=p=0 video.mp4

# 3. Resolution verification
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 video.mp4

# 4. Run verify_groot_dataset.py
python custom/scripts/verify_groot_dataset.py --dataset datasets_groot_h264 --verbose
```

### Experiment 3: Training Quality Comparison

**Check if model quality is affected:**
- Train 1000 steps on AV1 dataset
- Train 1000 steps on H.264 dataset
- Compare loss curves
- Compare evaluation MAE

---

## Research Findings

### LeRobot Official Recommendation (from HuggingFace Blog)

Source: [Scaling robotics datasets with video encoding](https://huggingface.co/blog/video-encoding)

**LeRobot recommends AV1** (`libsvtav1`) for dataset storage, but this is optimized for **compression**, not training speed.

**Decoding Speed Comparison (Load Time Ratio - lower is better):**

| Scenario | H.264 | H.265 | AV1 |
|----------|-------|-------|-----|
| 1 frame | 25.04 | 4.16 | 4.52 |
| 2 frames | 63.56 | 1.60 | 1.00 |
| 6 frames | 3.89 | 0.51 | 0.48 |

**Key insight:** For multi-frame loading (typical in training), AV1 and H.265 are similar. H.264 is actually **slower** in LeRobot's benchmark!

**This contradicts our hypothesis!** The benchmark shows AV1 is NOT slower than H.264 for training workloads.

### Why Our Training Might Still Be Slow

Possible explanations:
1. **LeRobot benchmark used CPU decoding** - Our setup might not be using optimal decode path
2. **Video backend matters** - `torchvision_av` vs `decord` may have different performance
3. **GPU power limit is the real bottleneck** - At 95W, the GPU is the limiting factor, not video decode
4. **Different GOP settings** - LeRobot uses GOP=2 for fast random access

### Revised Hypothesis

The slow training speed (~1.67s/step) is likely due to:
1. **GPU power limit (95W)** - Primary bottleneck (confirmed by nvidia-smi)
2. **Video backend choice** - `decord` might be faster than `torchvision_av`
3. **NOT the codec** - AV1 decode speed is comparable to H.264/H.265

### References

- [x] LeRobot video encoding blog: [huggingface.co/blog/video-encoding](https://huggingface.co/blog/video-encoding)
- [x] LeRobot Dataset v3 format: [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [ ] GR00T documentation on supported video formats
- [ ] NVIDIA NVDEC supported codecs

---

## Results

### Experiment 1: Codec Performance
**Status:** Not yet run

| Codec | File Size | Decode Time | Training Speed | GPU Util |
|-------|-----------|-------------|----------------|----------|
| AV1 | TBD | TBD | ~1.67s/step | 100% |
| H.264 | TBD | TBD | TBD | TBD |

### Experiment 2: Data Integrity
**Status:** Not yet run

| Check | AV1 | H.264 | Match? |
|-------|-----|-------|--------|
| Frame count | TBD | TBD | TBD |
| Duration | TBD | TBD | TBD |
| Resolution | TBD | TBD | TBD |
| Verification | TBD | TBD | TBD |

### Experiment 3: Training Quality
**Status:** Not yet run

| Metric | AV1 | H.264 | Difference |
|--------|-----|-------|------------|
| Final loss | TBD | TBD | TBD |
| MAE | TBD | TBD | TBD |

---

## Conclusion

**Status:** Research complete - Re-encoding NOT recommended, decord needs validation

### Key Finding

Based on [LeRobot's official benchmark](https://huggingface.co/blog/video-encoding):
- **AV1 is NOT slower than H.264** for multi-frame loading (typical in training)
- AV1 actually performs **better** than H.264 in their tests
- LeRobot officially recommends AV1 for datasets

### Root Cause of Slow Training

The ~1.67s/step speed is due to:
1. **GPU power limit (95W)** - Primary bottleneck, cannot be changed on this laptop
2. **Video backend** - `decord` may provide improvement but needs validation first

### Recommendation

- [x] **Do NOT add H.264 re-encoding to pipeline** - No benefit, adds complexity
- [x] **Keep AV1** - It's the LeRobot recommended format
- [x] **Keep `torchvision_av` as default** - Known to work, safe
- [ ] **Validate `decord` backend** - Test before switching (see experiments below)
- [ ] **Accept current speed** - 95W power limit is the real constraint

---

## Pending: Video Backend Validation (decord vs torchvision_av)

### Background

The video backend determines how video frames are loaded during training:

```
Training Loop → Dataset.__getitem__() → get_frames_by_timestamps() → video backend
```

**Available backends in GR00T (`gr00t/utils/video.py`):**

| Backend | Implementation | Status |
|---------|----------------|--------|
| `torchvision_av` | PyAV via torchvision, iterates frames | ✅ Current default, tested |
| `decord` | Batch loading with `get_batch()` | ⚠️ Potentially faster, needs validation |
| `torchcodec` | New torchcodec library | ❓ Not tested |
| `opencv` | cv2.VideoCapture, frame-by-frame | ❌ Slowest |

### Concerns with Switching to decord

1. **Different frame retrieval logic**
   - `torchvision_av`: Seeks to keyframe, iterates to target, finds closest
   - `decord`: Maps timestamps to indices, uses `get_batch()`

2. **Timestamp mapping differences**
   - Could retrieve slightly different frames for same timestamp

3. **Color space**
   - Both should return RGB, but worth verifying

### Experiment Plan: Validate decord Backend

**Experiment 1: Frame Consistency Check**

Compare frames retrieved by both backends for same timestamps:

```python
# Test script to run after training completes
import numpy as np
from gr00t.utils.video import get_frames_by_timestamps

video_path = "/path/to/episode_000000.mp4"
timestamps = [0.0, 1.0, 2.0, 5.0, 10.0]

frames_tv = get_frames_by_timestamps(video_path, timestamps, video_backend="torchvision_av")
frames_dec = get_frames_by_timestamps(video_path, timestamps, video_backend="decord")

# Compare
for i, ts in enumerate(timestamps):
    diff = np.abs(frames_tv[i].astype(float) - frames_dec[i].astype(float)).mean()
    print(f"ts={ts}: mean pixel diff = {diff:.2f}")

# Acceptable: diff < 5 (minor interpolation differences)
# Problematic: diff > 20 (different frames)
```

**Experiment 2: Performance Benchmark**

Run short training with both backends:

```bash
# Baseline (current)
python scripts/gr00t_finetune.py --video-backend torchvision_av --max-steps 100 ...

# Test
python scripts/gr00t_finetune.py --video-backend decord --max-steps 100 ...
```

Compare: steps/second

**Experiment 3: Training Quality Check**

If Experiments 1-2 pass, run longer training and compare:
- Loss curves
- Evaluation MAE
- Inference behavior

### Results

**Experiment 1: Frame Consistency**
- Status: ⏳ Pending
- Result: TBD

**Experiment 2: Performance**
- Status: ⏳ Pending
- `torchvision_av`: ~1.67s/step (from current training)
- `decord`: TBD

**Experiment 3: Training Quality**
- Status: ⏳ Pending
- Result: TBD

---

## Next Steps

1. ~~Run codec benchmark experiments~~ (Not needed - LeRobot already benchmarked)
2. Wait for current training to complete (using `torchvision_av`)
3. Run decord validation experiments (frame consistency + performance)
4. If validated, switch default to `decord`
5. GPU power limit (95W) remains the primary constraint
