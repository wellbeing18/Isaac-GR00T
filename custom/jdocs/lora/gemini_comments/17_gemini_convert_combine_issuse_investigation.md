# Investigation Report: LeRobot v3 to GR00T Conversion & Combination Issues

**Date:** 2025-12-12
**Status:** Critical Issues Identified
**Author:** Gemini (AI Assistant)

## 1. Executive Summary

Following up on the initial investigation, I have confirmed that the **Video Concatenation Strategy** used in the current `combine_groot_datasets.py` script is fundamentally incompatible with the **Local Timestamp Strategy** used in `convert_lerobot_v3_to_groot.py`. This mismatch guarantees that GR00T will fail to load correct frames for any episode other than the first one in a chunk.

Additionally, the `verify_groot_dataset.py` script contains logic flaws that mask this issue by assuming the concatenated video structure is correct, leading to false positives during verification.

The **correct approach** is to strictly enforce a **One-Video-Per-Episode** structure throughout the pipeline, matching GR00T's expected format and simplifying synchronization.

---

## 2. Critical Issues Identified

### 2.1 The "Local Timestamp vs. Global Video" Mismatch (Fatal)

*   **The Mismatch:**
    *   **Convert Script:** Splits parquet files into `episode_XXX.parquet` and **resets timestamps to start at 0.0s** for each episode. This is "Local Time".
    *   **Combine Script:** Concatenates all videos in a chunk into a single `file-000.mp4`. This creates "Global Time".
*   **The Consequence:**
    *   GR00T loads `episode_005.parquet`. It sees `timestamp=0.5s`.
    *   It opens the video `file-000.mp4` (which contains episodes 0-10).
    *   It seeks to `0.5s` in the video.
    *   **Result:** It retrieves a frame from **Episode 0**, not Episode 5.
    *   Episode 5 actually starts at e.g. `timestamp=150.0s` in the concatenated video, but the parquet file has lost this information.

### 2.2 Verification Script False Positives

*   The `verify_groot_dataset.py` script attempts to verify synchronization by calculating an `expected_offset` (summing previous episode lengths).
*   It checks if `parquet_ts_min` matches this offset.
*   **The Flaw:** It assumes that if the timestamps *don't* match the offset (i.e., they start at 0), it's just a "warning" or might pass if the offset calculation matches the *video* duration check. It does not explicitly fail when it sees `timestamp=0` for episode 5 alongside a multi-episode video file.

### 2.3 Hardcoded Modality Assumptions

*   `convert_lerobot_v3_to_groot.py` hardcodes the action/state dimensions for `so100` and `so101` (lines 30-57).
*   **Risk:** If a user brings a custom robot or a LeRobot dataset with a different configuration (e.g., 7-DOF arm), the conversion will generate an incorrect `modality.json` without warning. The script should ideally infer these dimensions from `info.json` or `stats.json`.

---

## 3. Detailed Analysis of Scripts

### 3.1 `convert_lerobot_v3_to_groot.py`
*   **✅ Good:** Correctly splits `file-*.parquet` into `episode_*.parquet`.
*   **✅ Good:** Correctly resets `frame_index` to 0.
*   **❌ Bad:** Does **not** split video files. It leaves them as consolidated `chunk-000/file-*.mp4` (containing multiple episodes).
*   **❌ Bad:** Hardcoded robot configs (SO-100/SO-101) limits flexibility.

### 3.2 `combine_groot_datasets.py`
*   **❌ Bad:** Explicitly concatenates videos (`concatenate_video_files`). This destroys the 1:1 mapping needed for the local timestamps produced by the convert script.
*   **❌ Bad:** Forces `video_path` pattern to `file-000.mp4`, assuming a single video file per chunk.

### 3.3 `verify_groot_dataset.py`
*   **❌ Bad:** Verification logic is ambiguous regarding "Local vs Global" time. It allows the dataset to be in an inconsistent state (Local Parquet + Global Video) without hard failing.

---

## 4. The Correct Way (Proposed Solution)

To fix this, we must align the Data and Video representations. The most robust way for GR00T (and the one used in official demos) is **One Video Per Episode**.

### 4.1 Target Architecture
*   **Parquet:** `data/chunk-XXX/episode_YYY.parquet` (Timestamps 0.0 -> End)
*   **Video:** `videos/camera_name/chunk-XXX/episode_YYY.mp4` (Duration: 0.0 -> End)
*   **Info.json:** `"video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"`

### 4.2 Implementation Plan

#### Phase 1: Update `convert_lerobot_v3_to_groot.py`
1.  **Add Video Splitting:** Use `ffmpeg` to split the consolidated LeRobot video files into per-episode `episode_XXX.mp4` files.
    *   Use the `from_timestamp` / `to_timestamp` metadata (found in LeRobot's original consolidated parquet) to determine cut points.
    *   *Note:* Since we split parquet based on `episode_index`, we must do the same for video.
2.  **Update Info.json:** Change the `video_path` template to point to `episode_{episode_index:06d}.mp4`.
3.  **Dynamic Dimensions:** (Optional but recommended) Read dimensions from `stats.json` or `info.json` instead of hardcoding `ROBOT_CONFIGS`.

#### Phase 2: Update `combine_groot_datasets.py`
1.  **Remove Concatenation:** Delete the `concatenate_video_files` logic.
2.  **Simple Copy:** Just copy the `episode_XXX.mp4` files from source to destination, renaming them to match the new episode indices (just like the parquet files).
3.  **Chunking:** Ensure that if we put `episode_100.parquet` in `chunk-001`, we also put `episode_100.mp4` in `videos/.../chunk-001`.

#### Phase 3: Update `verify_groot_dataset.py`
1.  **Enforce 1:1 Mapping:** Fail if `episode_XXX.parquet` exists but `episode_XXX.mp4` does not.
2.  **Strict Timestamp Check:** Verify that `parquet_duration` ≈ `video_duration` for *each* episode file individually.

---

## 5. Specific Recommendations for Current Dataset
For the dataset at `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`:

1.  **Action/State Dimensions:** The dataset uses 6-DOF (5 arm + 1 gripper). The current script's hardcoded values (0-5, 5-6) are actually correct for this specific case.
2.  **Keys:** The keys `observation.images.head` and `observation.images.left_wrist` are correctly mapped.

**Immediate Action:**
I will proceed to modify the scripts to implement the **One-Video-Per-Episode** strategy. This is the only way to ensure reliable training with GR00T.


