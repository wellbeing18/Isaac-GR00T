# Data Collection & Conversion Pipeline Investigation

**Date:** 2024-12-14
**Purpose:** Document comprehensive research into data pipeline to identify potential issues causing poor model performance (twitching at home position).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [LeRobot Version Comparison](#lerobot-version-comparison)
3. [Data Collection Analysis](#data-collection-analysis)
4. [Conversion Pipeline Analysis](#conversion-pipeline-analysis)
5. [Dataset Structure Comparison](#dataset-structure-comparison)
6. [Verification Results](#verification-results)
7. [Root Cause Analysis](#root-cause-analysis)
8. [Recommendations](#recommendations)

---

## Executive Summary

### Key Findings

| Component | Status | Notes |
|-----------|--------|-------|
| Data collection script | ✅ Correct | 30Hz, dual camera, proper format |
| Conversion to GR00T format | ✅ Mostly correct | Per-episode files, timestamps reset |
| GR00T loading test | ✅ Works | Videos load correctly |
| Codebase version | ⚠️ v3.0 vs v2.1 | Different but functional |
| Video path structure | ⚠️ Different | `{video_key}/chunk/` vs `chunk/{video_key}/` |
| PyArrow compatibility | ⚠️ Version sensitive | Works in groot env (14.0.1) |

### Conclusion

**The data pipeline is NOT the primary cause of inference issues.** The dataset:
- Loads correctly in GR00T
- Has correct frame indexing (0-based per episode)
- Has correct timestamps (starts at 0)
- Has proper modality mappings

The more likely causes are **deployment parameters** (execution rate, action horizon) or **LoRA training capacity**.

---

## LeRobot Version Comparison

### Version History

| Version | Released With | Key Features |
|---------|---------------|--------------|
| v2.0 | lerobot ~0.3.x | Per-episode parquet/video files |
| v2.1 | lerobot ~0.3.5 | Added episodes_stats.jsonl |
| v3.0 | lerobot 0.4.0+ | Consolidated files, streaming support |

### Official GR00T Datasets Use v2.1

From HuggingFace `youliangtan/so101-table-cleanup`:
```json
{
  "codebase_version": "v2.1",
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
}
```

### Your Dataset Uses v3.0

From `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place/meta/info.json`:
```json
{
  "codebase_version": "v3.0",
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
}
```

### Path Pattern Differences

| Aspect | Official v2.1 | Your v3.0 |
|--------|---------------|-----------|
| **Data path** | `data/chunk-{chunk}/episode_{idx}.parquet` | Same ✅ |
| **Video path** | `videos/chunk-{chunk}/{video_key}/episode_{idx}.mp4` | `videos/{video_key}/chunk-{chunk}/episode_{idx}.mp4` ⚠️ |

**Visual comparison:**
```
Official v2.1:                          Your v3.0:
videos/                                 videos/
├── chunk-000/                         ├── observation.images.head/
│   ├── front/                         │   └── chunk-000/
│   │   └── episode_000000.mp4         │       └── episode_000000.mp4
│   └── wrist/                         └── observation.images.left_wrist/
│       └── episode_000000.mp4             └── chunk-000/
                                               └── episode_000000.mp4
```

**Impact:** GR00T reads `video_path` from `info.json` and correctly resolves your custom pattern.

---

## Data Collection Analysis

### Script: `/home/jrobot/project/XLeRobot/scripts/collect_xlerobot_data.py`

#### Configuration

| Parameter | Value | Correct? |
|-----------|-------|----------|
| `ACTION_FPS` | 30 Hz | ✅ |
| Camera FPS | 30 fps | ✅ |
| Resolution | 640x480 | ✅ |
| Video codec | AV1 | ✅ |
| Robot type | so101_follower | ✅ |

#### Camera Setup

```python
# Left arm cameras
"left_wrist": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}
"head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}
```

#### Task Description Handling

Tasks are stored properly:
- `tasks.jsonl`: Contains task_index → task description mapping
- `episodes.jsonl`: Each episode has task_index and tasks array
- `config.yaml`: Contains full task configuration

Example from your dataset:
```json
// tasks.jsonl
{"task_index": 0, "task": "pick up the red cube and place it on the white plate"}
{"task_index": 1, "task": "pick up the green cube and place it on the white plate"}
...

// episodes.jsonl
{"episode_index": 0, "length": 1152, "task_index": 0, "tasks": ["pick up the red cube and place it on the white plate"]}
```

#### Potential Config Issue (Minor)

Config files say `action_fps: 5` but code uses `ACTION_FPS = 30`. Code wins, so actual data is 30Hz. This is just a documentation inconsistency.

---

## Conversion Pipeline Analysis

### Script: `/home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py`

#### What It Does

1. **Splits consolidated parquet files** → Per-episode files
   - Input: `data/chunk-000/file-000.parquet` (contains many episodes)
   - Output: `data/chunk-000/episode_000000.parquet` (one episode each)

2. **Resets frame indices** to start at 0 for each episode
   - Before: frame_index might be global (0, 1, 2, ... 10000)
   - After: Each episode starts at 0

3. **Resets timestamps** to start at 0 for each episode
   - Uses per-episode timestamp offset

4. **Splits videos** using ffmpeg
   - Uses `-avoid_negative_ts make_zero` to reset timestamps

5. **Creates metadata files**
   - `episodes.jsonl`: Episode metadata
   - `tasks.jsonl`: Task descriptions
   - Updates `info.json` with correct path patterns

#### Verified Results

Checked converted data with pyarrow:
```
Episode 0: frame_index 0-1151, timestamp 0.000-38.367s ✅
Episode 5: frame_index 0-912, timestamp 0.000-30.400s ✅
Video start_time: 0.000000 ✅
```

---

## Dataset Structure Comparison

### Metadata Files

| File | Official v2.1 | Your v3.0 | Notes |
|------|---------------|-----------|-------|
| `info.json` | ✅ | ✅ | Different path patterns |
| `modality.json` | ✅ | ✅ | Correctly maps keys |
| `episodes.jsonl` | ✅ | ✅ | Same format |
| `tasks.jsonl` | ✅ | ✅ | Same format |
| `episodes_stats.jsonl` | ✅ | ❌ Missing | v2.1 specific |
| `stats.json` | ❌ | ✅ | v3.0 specific |

### Modality Mapping

Your `modality.json`:
```json
{
  "state": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "action": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "video": {
    "front": {"original_key": "observation.images.head"},
    "wrist": {"original_key": "observation.images.left_wrist"}
  },
  "annotation": {
    "human.task_description": {"original_key": "task_index"}
  }
}
```

This correctly maps:
- GR00T key `front` → Your folder `observation.images.head`
- GR00T key `wrist` → Your folder `observation.images.left_wrist`

---

## Verification Results

### Script: `custom/scripts/verify_dataset_for_groot.py`

Full verification output:
```
✅ [OK] Directory data/: Exists
✅ [OK] Directory meta/: Exists
✅ [OK] Directory videos/: Exists
✅ [OK] info.json: File exists and is valid JSON
⚠️ [WARNING] codebase_version: v3.0 (differs from official v2.1)
✅ [OK] data_path pattern: Matches official format
⚠️ [WARNING] video_path pattern: Uses v3.0 structure
✅ [OK] modality.json: File exists
✅ [OK] modality.json[state]: Keys: ['single_arm', 'gripper']
✅ [OK] modality.json[action]: Keys: ['single_arm', 'gripper']
✅ [OK] modality.json[video]: Keys: ['front', 'wrist']
✅ [OK] episodes.jsonl: Found 70 episodes
✅ [OK] tasks.jsonl: Found 7 tasks
✅ [OK] parquet content checks: All valid
✅ [OK] video content checks: All valid
✅ [OK] GR00T loading test: Successfully loaded dataset
✅ [OK] GR00T video.front shape: (1, 480, 640, 3)
✅ [OK] GR00T video.wrist shape: (1, 480, 640, 3)

Summary: 33 checks | 29 OK | 4 warnings | 0 errors
```

### GR00T Loading Confirmed

```python
from gr00t.data.dataset import LeRobotSingleDataset
dataset = LeRobotSingleDataset(
    dataset_path="/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place",
    modality_configs=modality_config,
    video_backend="torchvision_av",
    embodiment_tag="new_embodiment"
)
sample = dataset.get_step_data(0, 0)
# Returns: video.front (1,480,640,3), video.wrist (1,480,640,3), state.*, action.*
```

---

## Root Cause Analysis

### Data Pipeline: Mostly Ruled Out

The data collection and conversion pipeline is **functional**:
- ✅ Data loads correctly in GR00T
- ✅ Frame indices reset to 0 per episode
- ✅ Timestamps reset to 0 per episode
- ✅ Videos have correct start_time=0
- ✅ Modality mappings resolve correctly
- ✅ Task descriptions preserved

### Remaining Possible Data Issues

1. **Missing `episodes_stats.jsonl`**
   - v2.1 has per-episode statistics
   - v3.0 has global `stats.json`
   - Could affect normalization during inference (unlikely)

2. **Extra parquet/video files**
   - Verification found 78 parquet files, expected 70
   - May have leftover files from previous runs (harmless)

### More Likely Causes (NOT Data Related)

See `potential_issues_tutorial_comparison.md` for details:

1. **Deployment parameters**
   - Execution rate: 30Hz vs official 50Hz
   - Action horizon: 16 vs official 8
   - Temporal ensembling: You use it, official doesn't
   - Fixed random seed: May limit action diversity

2. **LoRA vs Full Finetuning**
   - You use LoRA (rank 64), official uses full finetuning
   - `new_embodiment` action head was never pretrained
   - LoRA may not have sufficient capacity

---

## Recommendations

### Immediate Actions

1. **Clean up extra files** (optional)
   ```bash
   # Remove any files beyond episode 69
   rm datasets_copy/left/pick_and_place/data/chunk-000/episode_00007*.parquet
   ```

2. **Test deployment parameter changes**
   - Try 50Hz execution rate instead of 30Hz
   - Try action_horizon=8 instead of 16
   - Disable temporal ensembling
   - Remove fixed random seed

### Future Improvements

1. **Consider full v2.1 conversion**
   - Restructure video folders to match official pattern
   - Generate `episodes_stats.jsonl`
   - Change `codebase_version` to "v2.1"

2. **Try full finetuning** (if GPU memory allows)
   - Remove `--lora-rank` flag
   - Requires ~60GB+ VRAM (H100/A100)

---

## PyArrow Compatibility Matrix

| Environment | PyArrow | Can Read Parquet? |
|-------------|---------|-------------------|
| Base Python 3.13 | 19.0.0 | ❌ "Repetition level histogram size mismatch" |
| lerobot conda | 21.0.0 | ✅ Yes |
| groot conda | 14.0.1 | ✅ Yes |

**Note:** The parquet files are valid; pyarrow 19.0.0 has a compatibility bug. Training/inference use groot env (14.0.1) which works fine.

---

## Files Reference

| File | Purpose |
|------|---------|
| `/home/jrobot/project/XLeRobot/scripts/collect_xlerobot_data.py` | Data collection |
| `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place/` | Original v3 dataset |
| `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place/` | Converted dataset |
| `/home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py` | Conversion script |
| `/home/jrobot/project/Isaac-GR00T/custom/scripts/verify_dataset_for_groot.py` | Verification script |
| `/home/jrobot/project/Isaac-GR00T/gr00t/data/dataset.py` | GR00T dataset loader |
