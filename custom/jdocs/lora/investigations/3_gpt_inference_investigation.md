### Investigation Report: Poor Inference Performance (GR00T LoRA, GPT)

**Date**: 2025-12-13  
**Checkpoint**: `/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251212_180908058/best`  
**Dataset**: `/home/jrobot/project/XLeRobot/datasets_groot`

### Executive summary
- **Primary root cause (confirmed)**: **video/action misalignment** in `datasets_groot` due to consolidated videos (`file-000.mp4`) + **per-episode parquet timestamps resetting to 0**, while `LeRobotSingleDataset` loads frames **by timestamp** from the video file. This makes many episodes sample frames from the start of the concatenated video → the policy learns to **ignore vision** (appears “blind”) and relies on state priors.
- **Secondary (confirmed)**: open-loop “MAE looks good” can be a **false signal** here. A trivial baseline (`predict action := current state`) achieves **~1.41° MAE** on the dataset, so MAE alone does not imply task success.
- **Tooling issue (fixed)**: the training “diagnosis” step was effectively broken for this dataset layout (sample loading failed), reducing confidence in the offline validation loop.

### Evidence (repo + dataset)
- **Dataset `video_path` points to consolidated video**:

```json
// /home/jrobot/project/XLeRobot/datasets_groot/meta/info.json
"video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4"
```

- **No per-episode videos exist**:
  - `videos/observation.images.head/chunk-000/file-000.mp4`
  - `videos/observation.images.left_wrist/chunk-000/file-000.mp4`

- **Per-episode parquet timestamps reset** (but global `index` is continuous):
  - Example:
    - `episode_000010.parquet`: `timestamp[0]=0.0`, `index[0]=9901`
    - `episode_000020.parquet`: `timestamp[0]=0.0`, `index[0]=17805`

- **Dataloader behavior confirms the failure mode**: `LeRobotSingleDataset.get_video()` uses the parquet `timestamp` to fetch frames from the video file:

```python
# gr00t/data/dataset.py (LeRobotSingleDataset.get_video)
timestamp = self.curr_traj_data["timestamp"].to_numpy()
video_timestamp = timestamp[step_indices]
return get_frames_by_timestamps(video_path, video_timestamp, ...)
```

Given `file-000.mp4` + `timestamp` starting at ~0s for every episode, **many episodes will request ~0s frames** from the same video file.

- **Metric blind spot (quantified)**: baseline `action := state` MAE over all frames (49,869) is:
  - Per-joint MAE ≈ `[0.6968, 2.1590, 2.3633, 1.0237, 0.7278, 1.5106]`
  - Overall MAE ≈ **1.4135°**
  - This can make “good MAE” compatible with “robot barely moves / can’t complete the task”.

- **Training diagnosis failure (log evidence)**: `diagnosis.log` in the training output shows `diagnose_groot_inference.py` loaded **0 samples** (path/layout mismatch), so it did not validate visual/task conditioning.

### Fixes applied in Isaac-GR00T (to make checks reliable)
- **Added** `custom/scripts/infer_groot_so101.py` (compatibility shim re-exporting `deprecated_infer_groot_so101.py`) so diagnostic scripts importing `infer_groot_so101` work again.
- **Fixed** `custom/scripts/diagnose_groot_inference.py` to load samples via `LeRobotSingleDataset` (instead of manually guessing video paths) and to coerce images to uint8.
- **Enhanced** `custom/scripts/evaluate_groot_checkpoint.py` to report a baseline metric: **predict action=state**.

### Recommended remediation (priority order)
1. **Fix dataset video alignment (must-do)**
   - Preferred: generate **per-episode** videos and set `meta/info.json` to:

```json
"video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
```

   - Practically: re-run conversion on the **original LeRobot v3 dataset** (needs `meta/episodes/...` to cut correct segments), then re-build `datasets_groot`.
   - Alternative (not recommended): keep consolidated `file-000.mp4` but rewrite parquet `timestamp` to **global** time within the file (e.g. `timestamp = index / fps`) and verify the backend can seek accurately by timestamp.

2. **Retrain** the LoRA on the corrected dataset (current checkpoint likely trained with corrupted vision supervision).

3. **Validate with closed-loop + conditioning tests**
   - Use `custom/scripts/diagnose_closed_loop_sim.py` (closed-loop drift/ratio is a better proxy than open-loop MAE).
   - Use `custom/scripts/diagnose_groot_inference.py` (ensure sensitivity to image/task, not just state).

### Notes
- **Normalization**: checkpoint metadata indicates state/action are **absolute** and inference code uses absolute targets; this is likely not the primary failure compared to the video alignment bug.


