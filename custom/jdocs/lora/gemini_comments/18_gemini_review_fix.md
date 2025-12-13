# Review of Updated Convert & Combine Scripts

**Date:** 2025-12-12
**Reviewer:** Gemini (AI Assistant)

## 1. Summary

The changes made to `convert_lerobot_v3_to_groot.py` and `combine_groot_datasets.py` correctly implement the **One-Video-Per-Episode** strategy identified in the investigation. This is the correct structural fix for the timestamp synchronization issues.

However, the **Verification Script (`verify_groot_dataset.py`)** is now **outdated and incorrect**. It contains logic that assumes the old "Global Timestamp / Concatenated Video" format. If you run verification now, it will likely report false failures.

---

## 2. Detailed Review

### 2.1 `convert_lerobot_v3_to_groot.py` (✅ Approved)

*   **Video Splitting:** The new `split_videos_to_episodes` function correctly uses `ffmpeg` to split source videos based on the `from_timestamp` / `to_timestamp` metadata.
    *   *Minor Note:* It uses `-c copy` (stream copy) by default. This is fast but relies on keyframes. For very precise frame alignment, re-encoding is safer, but for training data usually recorded at fixed frame rates with frequent keyframes, this is an acceptable trade-off for speed.
*   **Info.json Update:** The script correctly updates the `video_path` pattern to `videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4`.
*   **Parquet:** Correctly resets `frame_index` to 0 for each episode.

### 2.2 `combine_groot_datasets.py` (✅ Approved)

*   **Removal of Concatenation:** The script no longer attempts to concatenate videos. This is the most important fix.
*   **Enforcement:** It now **errors out** if it finds unconverted `file-*.mp4` files, forcing the user to run the conversion script first. This prevents "half-converted" broken datasets.
*   **Copy Logic:** It correctly copies and renames the individual episode video files to match the new episode indices.

### 2.3 `verify_groot_dataset.py` (❌ Needs Update)

The verification script has **not** been updated to match the new format.

*   **The Issue:** In `check_data_video_sync`, the script calculates an `expected_offset` by summing the lengths of all previous episodes (lines 747-750).
    *   *Old Behavior:* It expected parquet timestamp to equal this global offset.
    *   *New Behavior:* Parquet timestamp is always ~0.0 (start of episode). Video is also cut to start at 0.0.
*   **The Failure:** The script will calculate `expected_offset = 300.0` (for example) for episode 10, compare it to `parquet_ts = 0.0`, and report a **Sync Error**.

---

## 3. Action Items

To complete the fix, **`verify_groot_dataset.py` must be updated**.

**Required Changes for Verification:**
1.  **Remove Cumulative Offset Logic:** The expected timestamp for the first frame of *any* episode should now be `0.0` (or close to it).
2.  **Verify Duration Match:** Instead of checking global offsets, check that:
    *   `video_duration` (from `episode_XXX.mp4`) ≈ `parquet_duration` (max timestamp in `episode_XXX.parquet`).
3.  **Update Info/Json Checks:** Ensure the verification script accepts the new `video_path` pattern in `info.json` without warning.

**Recommendation:**
Please ask Claude to update `custom/scripts/verify_groot_dataset.py` to support the "One-Video-Per-Episode" format.


