# LeRobot v3 to GR00T Dataset Conversion Guide

**Date:** 2025-11-24
**Purpose:** Bridge the gap between LeRobot v3 dataset format and GR00T v2 expectations

---

## Problem Overview

GR00T's training pipeline expects **LeRobot v2 format**, but newer datasets are collected using **LeRobot v3**. While the core data (videos, states, actions) is compatible, the metadata format differs.

### Format Differences

| Component | LeRobot v3 | GR00T Expected (v2) |
|-----------|-----------|---------------------|
| **modality.json** | Per-field schema with types | Index-based mappings |
| **stats.json** | Scalar count values | Per-dimension count arrays |
| **episodes metadata** | Parquet in `meta/episodes/` | JSONL in `meta/episodes.jsonl` |
| **tasks metadata** | Parquet in `meta/tasks.parquet` | JSONL in `meta/tasks.jsonl` |

---

## Required Conversions

### 1. modality.json Format

**Problem:** LeRobot v3 uses descriptive schema format, GR00T expects index-based mapping.

**LeRobot v3 Format (Wrong):**
```json
{
  "observation.images.head": {
    "type": "image",
    "shape": [480, 640, 3],
    "info": {...}
  },
  "observation.state": {
    "type": "state",
    "shape": [6],
    "names": ["joint1", "joint2", ...]
  },
  "action": {
    "type": "action",
    "shape": [6],
    "names": ["joint1", "joint2", ...]
  }
}
```

**GR00T Format (Correct):**
```json
{
    "state": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "video": {
        "front": {
            "original_key": "observation.images.head"
        },
        "wrist": {
            "original_key": "observation.images.left_wrist"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
```

**Key Differences:**
- **State/Action:** Index ranges instead of full schema
- **Video:** Mapping from GR00T names to dataset keys via `original_key`
- **Annotation:** Task description mapping

**Conversion Logic:**
1. Map state/action to component ranges (arm joints + gripper)
2. Map camera keys to GR00T standard names (front, wrist)
3. Map task_index to annotation field

---

### 2. stats.json Count Field

**Problem:** LeRobot v3 has scalar `count`, GR00T expects per-dimension array.

**LeRobot v3 (Wrong):**
```json
{
  "action": {
    "count": [1500],           ← Single value
    "mean": [a, b, c, d, e, f],
    "std": [a, b, c, d, e, f]
  }
}
```

**GR00T Expected (Correct):**
```json
{
  "action": {
    "count": [1500, 1500, 1500, 1500, 1500, 1500],  ← Per-dimension
    "mean": [a, b, c, d, e, f],
    "std": [a, b, c, d, e, f]
  }
}
```

**Conversion Logic:**
1. Read `count` field
2. Get dimension from `mean` or `std` field
3. Replicate count value across all dimensions
4. Apply to both `action` and `observation.state`

**Why This Matters:**
GR00T extracts statistics for sub-components (e.g., `single_arm` [0:5], `gripper` [5:6]). With scalar count, accessing `count[5]` fails.

---

### 3. episodes.jsonl Generation

**Problem:** LeRobot v3 stores episodes in Parquet, GR00T expects JSONL.

**LeRobot v3 Location:**
```
meta/episodes/chunk-000/file-000.parquet
```

**GR00T Expected Location:**
```
meta/episodes.jsonl
```

**JSONL Format:**
```json
{"episode_index": 0, "length": 150}
{"episode_index": 1, "length": 150}
...
```

**Conversion Logic:**
1. Read `total_episodes` and `total_frames` from `meta/info.json`
2. Calculate frames per episode (assume equal distribution)
3. Generate one JSON line per episode with `episode_index` and `length`

**Note:** If episodes have varying lengths, need to read from Parquet (if accessible) or episodes directory structure.

---

### 4. tasks.jsonl Generation

**Problem:** LeRobot v3 stores tasks in Parquet, GR00T expects JSONL.

**LeRobot v3 Location:**
```
meta/tasks.parquet
```

**GR00T Expected Location:**
```
meta/tasks.jsonl
```

**JSONL Format:**
```json
{"task_index": 0, "task": "grasp object"}
```

**Conversion Logic:**
1. Read `total_tasks` from `meta/info.json`
2. If single task, create simple entry with task_index=0
3. If multiple tasks, try to read from `tasks.parquet` or generate generic descriptions

---

## Conversion Workflow

### Manual Conversion Steps

**Step 1: Create modality.json**
```bash
# For SO-100/SO-101 dual-camera setup
cat > meta/modality.json << 'EOF'
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
EOF
```

**Step 2: Fix stats.json**
```python
import json

with open('meta/stats.json') as f:
    stats = json.load(f)

for key in ["action", "observation.state"]:
    if key in stats:
        old_count = stats[key]["count"]
        dimension = len(stats[key]["mean"])
        if len(old_count) == 1:
            stats[key]["count"] = [old_count[0]] * dimension

with open('meta/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

**Step 3: Generate episodes.jsonl**
```python
import json

with open('meta/info.json') as f:
    info = json.load(f)

total_episodes = info['total_episodes']
total_frames = info['total_frames']
frames_per_episode = total_frames // total_episodes

with open('meta/episodes.jsonl', 'w') as f:
    for i in range(total_episodes):
        f.write(json.dumps({"episode_index": i, "length": frames_per_episode}) + '\n')
```

**Step 4: Generate tasks.jsonl**
```python
import json

task_data = {"task_index": 0, "task": "grasp object"}
with open('meta/tasks.jsonl', 'w') as f:
    f.write(json.dumps(task_data) + '\n')
```

---

## Automated Conversion Script

See: `/home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py`

Usage:
```bash
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path /path/to/dataset \
    --robot-type so101  # or so100, unitree_g1, etc.
```

---

## Validation After Conversion

### Check Required Files Exist

```bash
ls -la meta/
# Should show:
# - info.json (original)
# - stats.json (modified)
# - modality.json (new GR00T format)
# - episodes.jsonl (new)
# - tasks.jsonl (new)
```

### Verify modality.json Format

```python
import json

with open('meta/modality.json') as f:
    modality = json.load(f)

# Should have these top-level keys
assert 'state' in modality
assert 'action' in modality
assert 'video' in modality
assert 'annotation' in modality

# Check state/action have index ranges
assert 'single_arm' in modality['state']
assert 'start' in modality['state']['single_arm']
assert 'end' in modality['state']['single_arm']
```

### Verify stats.json Count

```python
import json

with open('meta/stats.json') as f:
    stats = json.load(f)

# Count should be per-dimension
action_count = stats['action']['count']
action_mean = stats['action']['mean']

assert len(action_count) == len(action_mean), "Count should match dimension"
```

### Verify JSONL Files

```bash
# Check episodes.jsonl
wc -l meta/episodes.jsonl
# Should match total_episodes in info.json

head -1 meta/episodes.jsonl
# Should show: {"episode_index": 0, "length": N}

# Check tasks.jsonl
cat meta/tasks.jsonl
# Should show at least one task
```

---

## Robot-Specific Configurations

### SO-100 / SO-101 (6 DOF)

**State/Action Mapping:**
- Indices [0:5] → 5 arm joints (shoulder pan/lift, elbow, wrist flex/roll)
- Index [5:6] → 1 gripper joint

**Camera Mapping:**
- Single camera: `observation.images.webcam` → `video.webcam`
- Dual camera:
  - `observation.images.head` → `video.front`
  - `observation.images.left_wrist` → `video.wrist`

### UnitreeG1 (Different DOF)

Would need different index ranges based on arm configuration.

---

## Common Issues and Solutions

### Issue 1: IndexError during dataset loading

**Error:**
```
IndexError: index 5 is out of bounds for axis 0 with size 1
```

**Cause:** stats.json has scalar count

**Solution:** Run Step 2 (Fix stats.json)

### Issue 2: ValidationError for modality.json

**Error:**
```
pydantic_core._pydantic_core.ValidationError: Field 'state' required
```

**Cause:** Using LeRobot v3 format instead of GR00T format

**Solution:** Run Step 1 (Create modality.json)

### Issue 3: FileNotFoundError for episodes.jsonl

**Error:**
```
FileNotFoundError: meta/episodes.jsonl
```

**Cause:** JSONL not generated

**Solution:** Run Step 3 (Generate episodes.jsonl)

### Issue 4: FileNotFoundError for tasks.jsonl

**Error:**
```
FileNotFoundError: meta/tasks.jsonl
```

**Cause:** JSONL not generated

**Solution:** Run Step 4 (Generate tasks.jsonl)

---

## Summary Checklist

Before training with GR00T, verify:

- [ ] `meta/modality.json` exists and uses GR00T format (index ranges + original_key)
- [ ] `meta/stats.json` has per-dimension count arrays (not scalar)
- [ ] `meta/episodes.jsonl` exists with one line per episode
- [ ] `meta/tasks.jsonl` exists with at least one task
- [ ] Original files still exist:
  - [ ] `meta/info.json`
  - [ ] `meta/episodes/` directory
  - [ ] `meta/tasks.parquet`

**All conversions are non-destructive** - original files remain intact!

---

## References

- GR00T Dataset Code: `/home/jrobot/project/Isaac-GR00T/gr00t/data/dataset.py`
- LeRobot v3 Format: https://github.com/huggingface/lerobot
- Example SO-100 config: `/home/jrobot/project/Isaac-GR00T/examples/SO-100/so100_dualcam__modality.json`
