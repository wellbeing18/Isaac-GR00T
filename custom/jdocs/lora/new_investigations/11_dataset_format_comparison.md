# Dataset Format Comparison Report

## Datasets Analyzed

1. **Your Dataset:** `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place/`
2. **youliangtan/so101-table-cleanup:** `/tmp/so101-table-cleanup/`
3. **c299m/so101-pen-in-box-v2:** `/tmp/so101-pen-in-box-v2/`
4. **5hadytru/so101_grasp_1:** `/tmp/so101_grasp_1/`

---

## info.json Comparison

| Field | Your Dataset | youliangtan | c299m | 5hadytru |
|-------|-------------|-------------|-------|----------|
| codebase_version | **v3.0** | v2.1 | v2.1 | **v3.0** |
| robot_type | so101_follower | so101_follower | so101_follower | so101_follower |
| total_episodes | 70 | 80 | 130 | 210 |
| fps | 30 | 30 | 30 | 30 |

### Video Path Pattern (CRITICAL DIFFERENCE)

| Dataset | video_path |
|---------|-----------|
| Your Dataset | `videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4` |
| youliangtan | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| c299m | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| 5hadytru | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |

**Observation:**
- v2.1 uses: `videos/chunk-{chunk}/{video_key}/...`
- v3.0 uses: `videos/{video_key}/chunk-{chunk}/...`

Your dataset follows v3.0 convention which is correct for your version.

### Data Path Pattern

| Dataset | data_path |
|---------|----------|
| Your Dataset | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| youliangtan | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| c299m | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| 5hadytru | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |

---

## Video Key Names (IMPORTANT)

### Your Dataset (info.json features)
```
observation.images.head      -> mapped to video.front via modality.json
observation.images.left_wrist -> mapped to video.wrist via modality.json
```

### youliangtan Dataset
```
observation.images.front
observation.images.wrist
```

### c299m Dataset
```
observation.images.front
observation.images.wrist
```

### 5hadytru Dataset
```
observation.images.front
observation.images.overhead   (different second camera!)
```

**FINDING:** Your video keys (`head`, `left_wrist`) differ from reference (`front`, `wrist`).
Your modality.json maps them correctly to GR00T's expected `video.front` and `video.wrist`.

---

## modality.json Comparison

### Your Dataset
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
    }
}
```

### c299m Dataset
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
        "front": {"original_key": "observation.images.front"},
        "wrist": {"original_key": "observation.images.wrist"}
    }
}
```

### 5hadytru Dataset
```json
{
    "video": {
        "overhead": {"original_key": "observation.images.overhead"},
        "front": {"original_key": "observation.images.front"}
    }
}
```

### youliangtan Dataset
**NO modality.json file!**

---

## Physical File Structure

### Your Dataset
```
videos/
├── observation.images.head/
│   └── chunk-000/
│       └── episode_000000.mp4 ...
└── observation.images.left_wrist/
    └── chunk-000/
        └── episode_000000.mp4 ...
```

### youliangtan Dataset (v2.1)
```
videos/
└── chunk-000/
    ├── observation.images.front/
    │   └── episode_000000.mp4 ...
    └── observation.images.wrist/
        └── episode_000000.mp4 ...
```

### 5hadytru Dataset (v3.0)
```
videos/
├── observation.images.front/
│   └── chunk-000/
│       └── file-000.mp4 ...
└── observation.images.overhead/
    └── chunk-000/
        └── file-000.mp4 ...
```

**FINDING:** Your physical structure matches v3.0 convention (5hadytru).

---

## State/Action Data Ranges (All in Degrees)

| Joint | Your Dataset | youliangtan | c299m | 5hadytru |
|-------|-------------|-------------|-------|----------|
| shoulder_pan | [-27.6, 48.3] | [-53.5, 25.0] | [-66.9, 37.7] | [-74.0, 78.4] |
| shoulder_lift | [-100, 64.9] | [-99.4, 54.4] | [-99.9, 70.8] | [-100, 67.3] |
| elbow_flex | [-61.8, 100] | [-47.4, 98.2] | [-99.3, 95.0] | [-100, 99.9] |
| wrist_flex | [-75.8, 81.3] | [-43.4, 89.2] | [32.6, 99.5] | [-10.5, 100] |
| wrist_roll | [-61.5, 4.5] | [-55.7, 3.0] | [-48.0, 54.0] | [-56.0, 63.9] |
| gripper | [0, 35.9] | [0.5, 19.5] | [0.7, 31.8] | [0, 61.2] |

**FINDING:** All datasets use degrees. Ranges vary based on task but are all reasonable.

---

## Summary of Differences

| Aspect | Your Dataset | Working References | Assessment |
|--------|-------------|-------------------|------------|
| Version | v3.0 | v2.1 or v3.0 | OK |
| Video path pattern | v3.0 style | varies | OK (matches version) |
| Video key names | head/left_wrist | front/wrist | OK (modality.json maps) |
| modality.json | Present | Sometimes missing | OK |
| State/action ranges | Degrees | Degrees | OK |
| Data shape | 6-DOF | 6-DOF | OK |

**No obvious format issues found.** The key areas to verify are:
1. Video loading with your specific video_path pattern
2. modality.json original_key mapping being applied correctly
