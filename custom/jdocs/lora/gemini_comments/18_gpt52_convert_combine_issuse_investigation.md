# 18 — Further Investigation: LeRobot v3 → GR00T conversion/combination (pick_and_place)

**Date:** 2025-12-12  
**Author:** GPT-5.2 (Cursor agent)  
**Scope:** Latest collected LeRobot v3 dataset at `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place` and the current Isaac-GR00T custom conversion/combination tooling.

---

## Executive summary (what’s actually broken)

Your latest dataset is in **LeRobot v3 multi-file-per-chunk layout**:

- **Data:** `data/chunk-000/file-000..007.parquet`
- **Episode metadata:** `meta/episodes/chunk-000/file-000..007.parquet`
- **Video:** `videos/observation.images.{head,left_wrist}/chunk-000/file-000..007.mp4`
- **Info template:** `meta/info.json` uses `{chunk_index}` and `{file_index}` placeholders

GR00T’s dataset loader **does not know about `file_index`** and **does not apply any per-episode timestamp offset**. It only:

1) Computes `episode_chunk = episode_index // chunks_size`  
2) Formats paths using **only** `episode_chunk`, `episode_index`, `video_key`  
3) Reads `timestamp` directly from the episode parquet and uses it to seek frames in the referenced video

So **any pipeline that produces**:

- a video file that contains multiple episodes (or multiple source files concatenated), **and**
- per-episode parquet timestamps that remain “local” (starting near 0 for each episode)

…will silently create **wrong video supervision** (wrong frames for most episodes) and can also trigger runtime failures like the `IndexError` you saw.

---

## Facts from the repo (source of truth)

### 1) GR00T loader constraints (hard requirements)

From `gr00t/data/dataset.py`:

- `video_path` is formatted with:
  - `episode_chunk`
  - `episode_index`
  - `video_key`
- `data_path` is formatted with:
  - `episode_chunk`
  - `episode_index`
- Video frames are fetched by **timestamps read from the parquet** (no extra offsets):
  - `timestamp = self.curr_traj_data["timestamp"].to_numpy()`
  - `video_timestamp = timestamp[step_indices]`
  - `get_frames_by_timestamps(video_path, video_timestamp, ...)`

Implications:

- **LeRobot v3 templates** like:
  - `data_path = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"`
  - `video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"`
  are **not directly usable**, because GR00T never supplies `chunk_index` / `file_index` in `.format(...)`.

- GR00T will not use `meta/episodes/chunk-000/file-*.parquet` mappings (e.g., `from_timestamp`, `to_timestamp`); those do not exist in GR00T’s loading path unless you explicitly bake them into `timestamp` or split videos.

### 2) GR00T “demo_data” shows one valid format (but not the only possible format)

`demo_data/robot_sim.PickNPlace/meta/info.json` uses:

- `data_path`: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
- `video_path`: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`

This implies a **per-episode video** design (timestamps can stay local).

However: GR00T’s loader is **template-driven**, so it can also support “per-chunk file-000.mp4” designs **as long as timestamps match that video**.

### 3) Current custom tooling status (important mismatches)

#### `custom/scripts/convert_lerobot_v3_to_groot.py`

What it does well:

- Creates `meta/modality.json` (required by GR00T)
- Generates `meta/tasks.jsonl` from `meta/tasks.parquet`
- Generates `meta/episodes.jsonl` from `meta/episodes/chunk-000/file-*.parquet`
- Splits `data/chunk-000/file-*.parquet` into per-episode `data/chunk-000/episode_*.parquet`

New issue discovered (beyond your existing doc):

- **`frame_index` is NOT reset** in `split_parquet_files()`.
  - The function calls `episode_df.reset_index(drop=True)` but does not set `episode_df["frame_index"] = np.arange(len(episode_df))`.
  - Your verification tooling expects `frame_index` starts at 0 per episode.
  - GR00T itself can also use `frame_index` for `action.task_progress`.

Video/timestamp handling:

- The conversion script **does not**:
  - rewrite `meta/info.json` templates into GR00T-compatible placeholders, nor
  - split videos, nor
  - adjust `timestamp` to match any changed/combined video layout.

#### `custom/scripts/combine_groot_datasets.py`

Key behavior relevant to this investigation:

- It produces a GR00T-style `info.json` like:
  - `data_path = data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
  - `video_path = videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4`
- It **copies and renames parquet** files into chunk directories.
- For videos, it groups by `chunk-XXX` and then:
  - copies single mp4 to `file-000.mp4`, or
  - **concatenates multiple mp4 files into `file-000.mp4`** (per chunk) using ffmpeg concat.

Critical issue (still present):

- **Concatenation requires timestamp remapping**, but no timestamp remapping occurs.
  - With LeRobot v3, episodes can be split across `file-000..007.mp4`, and the correct mapping is described by LeRobot’s episode metadata (`file_index`, `from_timestamp`, `to_timestamp`).
  - After concatenation, the episode’s correct position in the new `file-000.mp4` becomes:
    - `t_concat = file_offset[file_index] + t_within_file`
  - If parquet timestamps remain “local” (0-based per episode) or even “within original file”, the GR00T loader will seek the wrong region and retrieve wrong frames.

---

## Your latest dataset: what it implies for “correctness”

From `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place/meta/info.json`:

- `chunks_size = 1000`
- `total_episodes = 70`
- `video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"`

This is **LeRobot v3 canonical**, but it is **not** GR00T-loadable without additional conversion because:

- GR00T formats with `{episode_chunk}` and `{episode_index}` (not `{chunk_index}` / `{file_index}`).
- GR00T does not read LeRobot’s `meta/episodes/chunk-000/file-*.parquet` to learn `file_index` or `from_timestamp`.

Therefore, a “correct combined dataset for GR00T” must choose one of the following strategies:

- **Strategy A (per-episode video files)**: make `video_path` include `{episode_index}` and ensure the referenced mp4 contains only that episode (timestamps stay local).
- **Strategy B (per-chunk single video file)**: keep a shared video file for multiple episodes (e.g., `file-000.mp4`), but then **parquet timestamps must be global in that video’s timeline** (not local), and the template must map each episode to the correct shared file.

Your current combine script assumes Strategy B but does not enforce the timestamp invariants, and the conversion script does not prepare the data for it.

---

## Additional issues & risks (beyond the known “timestamp vs concat” bug)

### Issue 1 — `frame_index` reset is missing (conversion bug)

If the source parquet’s `frame_index` is global or non-0-based per episode, then:

- Your `verify_groot_dataset.py` / verification expectations can fail
- GR00T’s optional `action.task_progress` label path can be wrong

**Fix direction:** during per-episode parquet write, always reset:

- `frame_index = 0..len-1`
- (optionally) set `episode_index` column to the new episode index if you renumber episodes

### Issue 2 — `info.json` template compatibility is not guaranteed unless combine is used

If you train directly on a “converted but not combined” dataset:

- `convert_lerobot_v3_to_groot.py` does not rewrite `meta/info.json`
- Your dataset may still contain LeRobot v3 `video_path` placeholders (`file_index`) that GR00T cannot format

**Fix direction:** conversion should rewrite `meta/info.json` into a GR00T-loadable template (or training should always use a combined output that rewrites it).

### Issue 3 — Verification scripts can give false confidence

Both `custom/scripts/verify_groot_dataset.py` and `custom/scripts/verify_groot_training_setup.py` mostly validate:

- presence of files
- counts match
- sample videos are readable

But these checks can still pass even when **the model is trained on the wrong frames**, because:

- “total frame counts” can match while “frame-to-timestamp mapping” is wrong

**Missing validation** (recommended):

- For sampled episodes, confirm that:
  - `max(timestamp)` is within the referenced video’s duration
  - timestamps are monotonic and spacing is plausible
  - reading a small set of frames at timestamps succeeds without returning repeated frames or obviously wrong segments

### Issue 4 — AV1 codec + backend mismatches

Your dataset videos are `av1`. Some backends (notably decord) often fail on AV1.

**Practical implication:** Always use a backend that supports AV1 (PyAV/torchvision_av/torchcodec with AV1-enabled ffmpeg build).

---

## The “correct way” to build GR00T-trainable combined data (recommended approach)

### Recommendation: Strategy B2 — “file-aware chunk remapping” (no video splitting, no concatenation)

This is the most robust/scale-friendly approach for LeRobot v3 datasets like yours.

**Goal:** remove `file_index` from runtime by making each `(chunk_index, file_index)` become its own GR00T `episode_chunk`, so GR00T only ever needs `episode_chunk` to locate the correct video file.

#### High-level idea

For each original LeRobot video file:

- Original: `videos/{video_key}/chunk-000/file-003.mp4`
- New: `videos/{video_key}/chunk-{NEW_CHUNK}/file-000.mp4`

For each episode inside that original file, create:

- `data/chunk-{NEW_CHUNK}/episode_{NEW_EPISODE_INDEX:06d}.parquet`
- `meta/episodes.jsonl` row for `NEW_EPISODE_INDEX`

And most importantly:

- Rewrite parquet `timestamp` so that it is **absolute within `file-000.mp4`**:
  - `timestamp_new = timestamp_local + from_timestamp`

This ensures GR00T’s timestamp seek lands on the correct portion of the video, even though multiple episodes still share the same video file.

#### Required inputs (available in LeRobot v3)

From `meta/episodes/chunk-000/file-*.parquet` you typically have (per episode):

- `episode_index` (original episode id)
- `videos/.../file_index` (which `file-XYZ.mp4` contains it)
- `videos/.../from_timestamp` and `to_timestamp` (episode segment bounds in that file)
- `length` (frames)
- task metadata (`task_index` or `tasks`)

#### Output format (GR00T-loadable)

Create/ensure in output dataset:

- `meta/modality.json` (GR00T format)
- `meta/tasks.jsonl`
- `meta/episodes.jsonl` (with new episode indices and lengths)
- `meta/info.json`:
  - `chunks_size`: keep at 1000 (or any >= max episodes per new chunk)
  - `data_path`: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
  - `video_path`: `videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4`
- `videos/{video_key}/chunk-{episode_chunk}/file-000.mp4` (exactly one mp4 per chunk)
- `data/chunk-{episode_chunk}/episode_{episode_index}.parquet`

#### How to assign new indices

Let:

- `chunk_size = 1000` (GR00T uses `episode_index // chunk_size`)
- Define a deterministic mapping from each source `(chunk_index, file_index)` to a new `episode_chunk`:
  - Example: `new_chunk = source_chunk * 100 + file_index` (100 is safe upper bound if file_index < 100)
  - Or build a contiguous mapping in sorted order of existing files.

Then for episodes inside that file, assign:

- `new_episode_index = new_chunk * chunk_size + local_episode_id_within_that_file`

This guarantees:

- `new_episode_index // chunk_size == new_chunk`
- GR00T will format the correct `chunk-{new_chunk}` video path

#### Why this is better than concatenating videos

- No ffmpeg concat step (avoids keyframe/PTS corner cases)
- No need to compute file duration offsets
- Preserves original AV1 bitstreams exactly (fast copy)
- Keeps a compact number of videos (one per original file, not one per episode)

---

## Alternative valid approach (only if you prefer demo-style per-episode videos)

### Strategy A — “split per episode video” (demo_data style)

Set in `meta/info.json`:

- `video_path = videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4`

Then produce one mp4 per episode. Parquet timestamps can remain local (near 0).

Downsides:

- Requires slicing video segments (ffmpeg), often forces re-encode or keyframe-aligned cutting
- Larger number of files and more I/O overhead
- Needs updates to `verify_groot_training_setup.py` (it currently assumes `file-000.mp4`)

---

## What should change in the current scripts (so the pipeline becomes correct)

### 1) Update conversion to be “video-mapping aware”

In `custom/scripts/convert_lerobot_v3_to_groot.py`, add a conversion mode that:

- reads `meta/episodes/chunk-*/file-*.parquet` and builds a per-episode mapping:
  - `(episode_index -> source_file_index, from_timestamp, length, task_index, …)`
- writes per-episode parquet with:
  - `frame_index` reset to 0..T-1
  - `timestamp` rewritten to `timestamp + from_timestamp` (for Strategy B2)
- writes/copies videos into GR00T chunk layout:
  - `videos/{video_key}/chunk-{new_chunk}/file-000.mp4`

### 2) Make combine refuse unsafe multi-file chunks (or require pre-converted B2 layout)

In `custom/scripts/combine_groot_datasets.py`:

- If it encounters multiple mp4 files inside a chunk, **error out** (or require an explicit `--allow-concat` flag), because concatenation without timestamp remap is a data-corruption trap.
- Prefer to combine already “B2-converted” datasets by:
  - copying `chunk-*` directories directly and applying a chunk offset
  - renumbering episodes if desired (not strictly required if chunk namespaces don’t collide)

### 3) Strengthen verification to actually test sync

Augment verification with checks like:

- For N sampled episodes:
  - load parquet timestamps
  - confirm min/max timestamps are within the referenced video duration
  - decode 3 frames (start/mid/end) by timestamp and ensure decoding succeeds
  - optionally dump these frames to disk for human inspection (“sync visualization tool”)

This catches the exact class of “counts match but supervision is wrong” failures.

---

## Recommended concrete workflow for your dataset (pick_and_place)

### Step 0 — Work on a copy (still recommended)

Keep raw data immutable:

- Source: `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`
- Working: `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place`

### Step 1 — Convert using Strategy B2 (file-aware chunk remapping)

Produce an output dataset directory already GR00T-loadable (single-dataset case), e.g.:

- `/home/jrobot/project/XLeRobot/datasets_groot_pick_and_place`

### Step 2 — (Optional) Combine multiple datasets

Only combine datasets that already satisfy:

- one video file per chunk (i.e., `.../chunk-XXX/file-000.mp4` only)
- per-episode parquet files
- compatible `modality.json`

### Step 3 — Verify with sync-focused checks

Require passing:

- structure checks
- **sync checks** (timestamp-to-video)

Then proceed to LoRA finetuning.

---

## Open questions / follow-ups to fully close the loop

These are the items that determine which strategy is safest without “guessing”:

1) **What are the exact timestamp semantics in the raw `file-*.parquet`?**  
   - If timestamps are already absolute within their `file-XYZ.mp4`, Strategy B2 becomes simpler (just remap chunks/episodes; no timestamp shift).
   - If timestamps are episode-local (likely), Strategy B2 must add `from_timestamp`.

2) **Do the LeRobot episode parquet files include `from_timestamp`/`to_timestamp` and `file_index` as expected?**  
   - Your existing investigation doc suggests yes; this report assumes that metadata is available and reliable.

3) **How many episodes per source file, and is it consistent?**  
   - If it’s inconsistent, you must not use “episode_index // N” heuristics; rely on the metadata mapping.

---

## Bottom line recommendation

For `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`, the **most correct and least fragile** path is:

- **Do not concatenate videos** during combine.
- Convert using **file-aware chunk remapping** (Strategy B2):
  - one mp4 per chunk (`file-000.mp4`)
  - per-episode parquet
  - timestamps adjusted to match the shared mp4 timeline (using `from_timestamp`)

This aligns exactly with GR00T’s loader constraints and avoids silent supervision corruption.



