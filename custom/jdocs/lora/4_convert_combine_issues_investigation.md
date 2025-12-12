# Convert & Combine Pipeline Issues Investigation

**Date:** 2025-12-12
**Status:** CONSENSUS REACHED - Per-Episode Video Splitting
**Reviewers:** Claude (Opus 4.5), GPT-5.2, Gemini

## Executive Summary

The current convert/combine pipeline has **critical synchronization bugs** that cause training failures. The root cause is a mismatch between how LeRobot stores video/data mappings and how the combine script handles video concatenation.

**Symptom:** Training crashes with `IndexError: index 845 is out of bounds for axis 0 with size 700`

---

## Table of Contents

1. [Background: How GR00T Loads Video Frames](#1-background-how-groot-loads-video-frames)
2. [How LeRobot v3 Format Works](#2-how-lerobot-v3-format-works)
3. [What the Convert Script Does](#3-what-the-convert-script-does)
4. [What the Combine Script Does](#4-what-the-combine-script-does)
5. [Identified Issues](#5-identified-issues)
6. [Evidence and Data](#6-evidence-and-data)
7. [Potential Solutions](#7-potential-solutions)
8. [Questions for Discussion](#8-questions-for-discussion)

---

## 1. Background: How GR00T Loads Video Frames

GR00T uses **timestamp-based video frame retrieval** (not frame index):

```python
# gr00t/data/dataset.py:700-710
timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
video_timestamp = timestamp[step_indices]

return get_frames_by_timestamps(
    video_path.as_posix(),
    video_timestamp,  # <-- Uses timestamp to find video frame
    video_backend=self.video_backend,
)
```

```python
# gr00t/utils/video.py:95-99
# For each timestamp, find closest video frame
frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
frames = vr.get_batch(indices)
```

**Key insight:** GR00T reads `timestamp` from parquet and uses it to seek into the video file.

---

## 2. How LeRobot v3 Format Works

### 2.1 Video Storage

LeRobot stores videos in chunks with **multiple video files per chunk**:

```
videos/observation.images.head/chunk-000/
├── file-000.mp4  # Episodes 0-9 (9901 frames, 330 seconds)
├── file-001.mp4  # Episodes 10-19 (7904 frames)
├── file-002.mp4  # Episodes 20-29 (6798 frames)
└── ...
```

Each video file contains ~10 episodes concatenated together.

### 2.2 Episode-to-Video Mapping

LeRobot stores **critical metadata** in `meta/episodes/chunk-000/file-*.parquet`:

```
episode_index | videos/.../file_index | videos/.../from_timestamp | videos/.../to_timestamp
--------------|----------------------|---------------------------|------------------------
0             | 0                    | 0.000                     | 38.400
1             | 0                    | 38.400                    | 77.000
2             | 0                    | 77.000                    | 119.100
...           | ...                  | ...                       | ...
9             | 0                    | 299.300                   | 330.033
10            | 1                    | 0.000                     | 30.267
```

- `file_index`: Which video file contains this episode
- `from_timestamp`: Start timestamp in that video file
- `to_timestamp`: End timestamp in that video file

### 2.3 Data Parquet Timestamps

In each data parquet (per-episode), `timestamp` column is **local** (starts at 0):

```
Episode 0: timestamp range 0.000s - 38.367s
Episode 1: timestamp range 0.000s - 38.567s  # Also starts at 0!
```

### 2.4 Video Path Pattern

```json
"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
```

The `{file_index}` comes from the episode metadata mapping.

---

## 3. What the Convert Script Does

**Script:** `custom/scripts/convert_lerobot_v3_to_groot.py`

### 3.1 Good Changes (Correct)

1. ✅ Splits consolidated `file-*.parquet` into per-episode `episode_*.parquet`
2. ✅ Resets `frame_index` to start at 0 for each episode
3. ✅ Creates `episodes.jsonl` with task_index (after recent fix)
4. ✅ Creates `tasks.jsonl` from tasks.parquet
5. ✅ Creates `modality.json` for GR00T

### 3.2 Issues (Need Review)

1. ❓ **Timestamps unchanged** - Each episode parquet keeps local timestamps (0-based)
2. ❓ **Video files unchanged** - Original multi-file structure preserved
3. ❓ **Episode metadata not updated** - `from_timestamp`/`to_timestamp` mapping lost in conversion to jsonl

---

## 4. What the Combine Script Does

**Script:** `custom/scripts/combine_groot_datasets.py`

### 4.1 Good Changes (After Recent Fix)

1. ✅ Prefers `episode_*.parquet` over `file-*.parquet`
2. ✅ Errors if only unconverted data found
3. ✅ Combines `episodes.jsonl`, `tasks.jsonl` correctly
4. ✅ Re-indexes episode numbers for multi-dataset combination

### 4.2 Critical Issues

1. ❌ **Video concatenation** - Merges all video files into single `file-000.mp4`
2. ❌ **No timestamp adjustment** - Parquet timestamps stay local (0-based) but video is now concatenated
3. ❌ **Lost episode-video mapping** - `from_timestamp`/`to_timestamp` not preserved/updated
4. ❌ **video_path pattern incorrect** - Still has `{file_index}` but only `file-000.mp4` exists

---

## 5. Identified Issues

### Issue #1: Video Concatenation Breaks Timestamp Sync (CRITICAL)

**Before combine:**
```
Episode 0 → file-000.mp4, timestamp 0-38s
Episode 10 → file-001.mp4, timestamp 0-30s
Episode 35 → file-003.mp4, timestamp 0-22s
```

**After combine:**
```
Episode 0 → file-000.mp4, timestamp 0-38s        ✅ Works (first episode)
Episode 10 → file-000.mp4, timestamp 0-30s       ❌ Gets frames from episode 0!
Episode 35 → file-000.mp4, timestamp 0-22s       ❌ Gets frames from episode 0!
```

**Result:** All episodes after the first will retrieve **wrong video frames**.

### Issue #2: Missing Episode-Video Mapping Metadata

The original LeRobot format has:
- `videos/.../file_index` - which video file
- `videos/.../from_timestamp` - start time in video
- `videos/.../to_timestamp` - end time in video

This metadata is **completely lost** in the convert/combine pipeline. It's not in `episodes.jsonl`.

### Issue #3: Duplicate Parquet Files (FIXED)

~~The combine script copied both `episode_*.parquet` and `file-*.parquet` causing duplicates.~~

**Status:** Fixed on 2025-12-12 by preferring `episode_*.parquet` and erroring on unconverted data.

### Issue #4: task_index All Zeros (FIXED)

~~The convert script didn't map task descriptions to task indices correctly.~~

**Status:** Fixed on 2025-12-12 by looking up tasks from `tasks.parquet` mapping.

---

## 6. Evidence and Data

### 6.1 Timestamp Analysis

**Combined dataset parquet timestamps:**
```python
Episode 0:  timestamp 0.000s - 38.367s, frame_index 0-1151
Episode 10: timestamp 0.000s - 30.267s, frame_index 0-908   # Starts at 0!
Episode 35: timestamp 0.000s - 21.867s, frame_index 0-656   # Starts at 0!
Episode 69: timestamp 0.000s - 18.533s, frame_index 0-556   # Starts at 0!
```

**Expected timestamps for concatenated video:**
```python
Episode 0:  timestamp 0.000s - 38.367s     # Correct
Episode 10: timestamp 329.9s - 360.2s      # Should be cumulative!
Episode 35: timestamp 944.8s - 966.6s      # Should be cumulative!
Episode 69: timestamp 1643.7s - 1662.2s    # Should be cumulative!
```

### 6.2 Video File Analysis

**Original (8 files):**
```
file-000.mp4: 9901 frames (episodes 0-9)
file-001.mp4: 7904 frames (episodes 10-19)
file-002.mp4: 6798 frames (episodes 20-29)
file-003.mp4: 6745 frames (episodes 30-39)
file-004.mp4: 700 frames  (episodes 40-49)
file-005.mp4: 5703 frames (episodes 50-59)
file-006.mp4: 5567 frames (episodes 60-69 partial)
file-007.mp4: 6551 frames (episodes continued)
Total: 49869 frames
```

**Combined (1 file):**
```
file-000.mp4: 49869 frames (all 70 episodes concatenated)
```

### 6.3 Verification Script Output

```
[8/8] Checking Data-Video Sync...
  ✅ PASS: 49869 frames synchronized
  Frames from info.json: 49869
  Frames from episodes.jsonl: 49869
  Frames from parquet files: 49869
  Frames from video files: 49869
```

**This is MISLEADING** - frame counts match but timestamps are NOT synchronized!

### 6.4 Training Error

```
IndexError: Caught IndexError in DataLoader worker process 0.
  File "gr00t/data/dataset.py", line 703, in get_video
    video_timestamp = timestamp[step_indices]
IndexError: index 845 is out of bounds for axis 0 with size 700
```

This happens because GR00T tries to access frame 845 but the video reader only found 700 frames at timestamp 0-23s (wrong video segment).

---

## 7. Research Findings: GR00T's Expected Format

### 7.1 Official Demo Data Structure

From `/home/jrobot/project/Isaac-GR00T/demo_data/robot_sim.PickNPlace/`:

```
robot_sim.PickNPlace/
├── meta/
│   ├── info.json
│   │   └── "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
│   │   └── "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
│   ├── modality.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── stats.json
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── episode_000001.parquet
└── videos/chunk-000/
    └── observation.images.ego_view/
        ├── episode_000000.mp4    # ONE video per episode!
        └── episode_000001.mp4
```

### 7.2 Key Findings

1. **GR00T expects ONE video file per episode** (not consolidated multi-episode videos)
2. **video_path pattern**: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`
3. **NO `{file_index}` support** - only `{episode_chunk}`, `{episode_index}`, `{video_key}`
4. **Timestamp column** in parquet starts at ~0 for each episode (local timestamps)
5. **Demo doesn't have `frame_index`** - only `timestamp` is required for video sync

### 7.3 How GR00T Loads Video Frames

```python
# gr00t/data/dataset.py:700-710
timestamp = self.curr_traj_data["timestamp"].to_numpy()  # From parquet
video_timestamp = timestamp[step_indices]
return get_frames_by_timestamps(video_path, video_timestamp, ...)

# gr00t/utils/video.py:95-99
frame_ts = vr.get_frame_timestamp(range(num_frames))  # Video frame times
indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)  # Nearest neighbor
frames = vr.get_batch(indices)
```

**The sync mechanism:**
- Read `timestamp` from parquet row
- Find nearest video frame by timestamp matching
- Each episode video starts at time 0, parquet timestamps also start near 0

---

## 8. Recommended Solution: One Video Per Episode

Based on research, **Option C (One Video Per Episode)** is the correct approach that matches GR00T's expected format.

### 8.1 Required Changes

**Convert Script (`convert_lerobot_v3_to_groot.py`):**
1. Split consolidated videos into per-episode videos
2. Update `video_path` pattern in info.json to:
   ```json
   "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
   ```

**Combine Script (`combine_groot_datasets.py`):**
1. Copy per-episode video files (not concatenate)
2. Rename video files to match new episode indices
3. Ensure `video_path` pattern uses episode-level naming

### 8.2 Video Splitting Strategy

Original LeRobot has consolidated videos with timestamp metadata:
```
file-000.mp4: episodes 0-9
  Episode 0: from_timestamp=0.0, to_timestamp=38.4
  Episode 1: from_timestamp=38.4, to_timestamp=77.0
  ...
```

Need to split using ffmpeg:
```bash
# Extract episode 0 from file-000.mp4
ffmpeg -i file-000.mp4 -ss 0.0 -to 38.4 -c copy episode_000000.mp4

# Extract episode 1 from file-000.mp4
ffmpeg -i file-000.mp4 -ss 38.4 -to 77.0 -c copy episode_000001.mp4
```

### 8.3 Implementation Plan

1. **Phase 1: Update Convert Script**
   - Read `from_timestamp`/`to_timestamp` from episodes parquet
   - Split each video file into per-episode videos
   - Update `video_path` pattern in info.json

2. **Phase 2: Update Combine Script**
   - Copy per-episode videos with renamed indices
   - No video concatenation needed
   - Update `video_path` pattern to match GR00T format

3. **Phase 3: Update Verification Script**
   - Check video file exists for each episode
   - Verify video duration matches episode length
   - Test actual frame extraction at sample timestamps

---

## 9. Rejected Solutions

### Option A: Keep Multi-File Structure
**Rejected because:** GR00T doesn't support `{file_index}` in video_path pattern.

### Option B: Concatenate with Timestamp Adjustment
**Rejected because:**
- GR00T expects per-episode videos
- Complex and error-prone
- Doesn't match official demo format

### Option D: Use LeRobot Native Loading
**Rejected because:**
- GR00T has its own dataset loader
- Would require significant code changes

## 10. Questions for Discussion

1. **~~Does GR00T support `{file_index}` in video_path?~~**
   - **ANSWERED: NO** - Only `{episode_chunk}`, `{episode_index}`, `{video_key}` supported

2. **~~What does NVIDIA's official example use?~~**
   - **ANSWERED:** Per-episode videos: `episode_{episode_index:06d}.mp4`

3. **Is video splitting via ffmpeg reliable?**
   - Using `-c copy` for speed vs re-encoding for accuracy?
   - Keyframe alignment issues with `-c copy`?
   - Should we re-encode to ensure clean cuts?

4. **How to handle timestamp drift?**
   - LeRobot `from_timestamp`/`to_timestamp` may have slight offsets
   - Should we re-calculate timestamps from fps and frame counts?

5. **Implementation priority:**
   - Should we fix convert script first (video splitting)?
   - Or fix both convert + combine together?
   - Quick workaround: manually split videos for current dataset?

---

## Appendix: File Locations

- Convert script: `custom/scripts/convert_lerobot_v3_to_groot.py`
- Combine script: `custom/scripts/combine_groot_datasets.py`
- Verification script: `custom/scripts/verify_groot_dataset.py`
- Source dataset: `/home/jrobot/project/XLeRobot/datasets copy/left/pick_and_place`
- Combined dataset: `/home/jrobot/project/XLeRobot/datasets_groot`
- GR00T dataset code: `gr00t/data/dataset.py`
- GR00T video utils: `gr00t/utils/video.py`

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-12 | Fixed task_index bug in convert script |
| 2025-12-12 | Fixed duplicate parquet bug in combine script |
| 2025-12-12 | Identified critical timestamp sync issue |
| 2025-12-12 | Created this investigation document |
| 2025-12-12 | Researched GR00T official format - confirmed per-episode video requirement |
| 2025-12-12 | Determined solution: split videos to per-episode format |
| 2025-12-12 | Multi-AI review (GPT-5.2, Gemini) - consensus on Strategy A |
| 2025-12-12 | Added Section 11: Multi-AI Consensus Analysis |

## Next Steps

1. **Implement video splitting in convert script**
   - Add function to split consolidated videos using ffmpeg
   - Use `from_timestamp`/`to_timestamp` from episodes parquet
   - Update `video_path` pattern to GR00T format

2. **Update combine script**
   - Remove video concatenation logic
   - Copy per-episode videos with renamed indices
   - Update `video_path` pattern

3. **Update verification script**
   - Add per-episode video existence check
   - Add video duration vs episode length check
   - Add sample frame extraction test

4. **Test end-to-end**
   - Convert source dataset
   - Combine (if needed)
   - Verify
   - Run training

---

## 11. Multi-AI Consensus Analysis

### 11.1 Reviewers

Three AI systems independently analyzed this issue:
- **Claude (Opus 4.5)**: Original investigator
- **GPT-5.2**: See `gemini_comments/18_gpt52_convert_combine_issuse_investigation.md`
- **Gemini**: See `gemini_comments/17_gemini_convert_combine_issuse_investigation.md`

### 11.2 Points of Agreement (All Three)

1. **Root Cause Confirmed**: Video concatenation + local timestamps = wrong frame retrieval
2. **Current combine script is broken**: Cannot safely concatenate without timestamp remapping
3. **Verification is insufficient**: Frame count matching doesn't verify actual synchronization
4. **GR00T loader constraints**: Only supports `{episode_chunk}`, `{episode_index}`, `{video_key}` placeholders

### 11.3 Two Valid Strategies Proposed

| Aspect | Strategy A (Claude/Gemini) | Strategy B2 (GPT-5.2) |
|--------|---------------------------|----------------------|
| **Video handling** | Split to per-episode videos | Keep original multi-episode files |
| **Timestamp handling** | Keep local (0-based) | Adjust: `ts + from_timestamp` |
| **Files created** | 140 video files (70 eps × 2 cams) | 16 video files (8 files × 2 cams) |
| **GR00T compatibility** | Matches demo exactly | Also valid (template-driven) |
| **FFmpeg required** | Yes (splitting) | No video modification |
| **Complexity** | Medium | Medium-High (timestamp math) |

### 11.4 Additional Issue from GPT-5.2

GPT-5.2 identified a potential bug not previously caught:
> "`frame_index` is NOT reset in `split_parquet_files()`. The function calls `reset_index(drop=True)` but does not set `episode_df["frame_index"] = np.arange(len(episode_df))`."

**Analysis**: Upon verification, the current dataset shows `frame_index` already starts at 0 per episode. This suggests LeRobot v3 may already store local `frame_index`. However, adding explicit reset is a defensive fix worth implementing.

### 11.5 Final Recommendation: Strategy A (Per-Episode Videos)

**Reasons for choosing Strategy A:**

1. **Exact Match with GR00T Demo Format**
   - Official `demo_data/robot_sim.PickNPlace` uses per-episode videos
   - Template: `videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4`
   - This is the "known working" format

2. **Simpler Mental Model**
   - Episode 5 → `episode_000005.parquet` → `episode_000005.mp4`
   - 1:1 mapping, no timestamp arithmetic
   - Timestamps start at 0 in parquet, video also starts at 0

3. **GPT-5.2's Strategy B2 Has Hidden Complexity**
   - Requires correct `from_timestamp` metadata (must verify accuracy)
   - Requires careful chunk remapping (`new_chunk = chunk * 100 + file_index`)
   - Requires timestamp adjustment in every parquet
   - If metadata is wrong → silent corruption

4. **File Count is Not a Major Concern**
   - 140 video files is manageable
   - Modern filesystems handle this easily
   - I/O overhead minimal (dataset fits in memory)

5. **FFmpeg Splitting is Reliable**
   - Using `-c copy` with LeRobot timestamp metadata
   - Fallback to re-encode if keyframe issues occur

### 11.6 Rejected: Strategy B2 (File-Aware Chunk Remapping)

While technically valid, Strategy B2 was rejected because:
- More complex timestamp arithmetic increases risk of bugs
- Requires trusting metadata accuracy
- Doesn't match GR00T's official demo format
- Debugging synchronization issues becomes harder

---

## 12. Implementation Checklist

### Phase 1: Convert Script Updates
- [ ] Add `split_videos()` function using ffmpeg
- [ ] Read `from_timestamp`/`to_timestamp` from `meta/episodes/chunk-*/file-*.parquet`
- [ ] Split each consolidated video into per-episode videos
- [ ] Update `info.json` with GR00T-compatible `video_path` template
- [ ] Add explicit `frame_index` reset (defensive fix from GPT-5.2)

### Phase 2: Combine Script Updates
- [ ] Remove `concatenate_video_files()` function entirely
- [ ] Implement `copy_episode_videos()` function
- [ ] Rename video files to match new episode indices
- [ ] Ensure `video_path` pattern uses episode-level naming

### Phase 3: Verification Script Updates
- [ ] Add per-episode video existence check
- [ ] Add actual frame extraction test (decode at sample timestamps)
- [ ] Verify video duration ≈ max(timestamp) for each episode
- [ ] Add visual sync check (dump sample frames for inspection)
