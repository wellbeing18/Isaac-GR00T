# modality.json Format Fix

**Date:** 2025-11-24
**Issue:** Pydantic validation error - Wrong modality.json format

---

## Problem

Training started but failed with validation error:

```
pydantic_core._pydantic_core.ValidationError: 5 validation errors for LeRobotModalityMetadata
state
  Field required [type=missing, ...]
action.type
  Input should be a valid dictionary or instance of LeRobotActionMetadata [type=model_type, ...]
video
  Field required [type=missing, ...]
```

---

## Root Cause

GR00T expects a **specific modality.json format** that differs from standard LeRobot format.

### Wrong Format (LeRobot Style)

```json
{
  "observation.images.head": {
    "type": "image",
    "shape": [480, 640, 3],
    ...
  },
  "action": {
    "type": "action",
    "shape": [6],
    ...
  }
}
```

### Correct Format (GR00T Style)

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

---

## Key Differences

### 1. State/Action Format

**GR00T uses index ranges** instead of full definitions:

```json
"state": {
    "single_arm": {
        "start": 0,    // Indices 0-4 (5 joints)
        "end": 5
    },
    "gripper": {
        "start": 5,    // Index 5 (gripper)
        "end": 6
    }
}
```

This maps to your 6D state vector:
- `[0:5]` → shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
- `[5:6]` → gripper

### 2. Video Format

**GR00T uses "original_key" mapping**:

```json
"video": {
    "front": {
        "original_key": "observation.images.head"
    },
    "wrist": {
        "original_key": "observation.images.left_wrist"
    }
}
```

This maps GR00T's internal names (`front`, `wrist`) to your dataset keys.

### 3. Annotation Format

**GR00T expects task description mapping**:

```json
"annotation": {
    "human.task_description": {
        "original_key": "task_index"
    }
}
```

---

## What Was Fixed

### 1. Updated Dataset modality.json

**File:** `/home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/modality.json`

Replaced with correct GR00T format.

### 2. Updated Training Script

**File:** `custom/scripts/train_groot_mini_mvp.sh`

Updated auto-creation logic to generate correct format.

---

## Explanation of Mappings

### Your Dataset → GR00T

**State (6D vector):**
```
Your dataset:        GR00T mapping:
[0] shoulder_pan  →  state.single_arm (start=0, end=5)
[1] shoulder_lift →  state.single_arm
[2] elbow_flex    →  state.single_arm
[3] wrist_flex    →  state.single_arm
[4] wrist_roll    →  state.single_arm
[5] gripper       →  state.gripper (start=5, end=6)
```

**Action (6D vector):**
```
Same mapping as state:
[0:5] → action.single_arm
[5:6] → action.gripper
```

**Video (2 cameras):**
```
Your dataset:                    GR00T mapping:
observation.images.head       →  video.front
observation.images.left_wrist →  video.wrist
```

**Task:**
```
Your dataset:  GR00T mapping:
task_index  →  annotation.human.task_description
```

---

## Why This Format?

GR00T's modality format is designed for:

1. **Efficiency:** Index ranges instead of full metadata
2. **Flexibility:** Maps any dataset keys to GR00T's expected names
3. **Multi-robot support:** Standardized names across different robots

The `original_key` field allows GR00T to work with datasets that use different naming conventions.

---

## Testing the Fix

### 1. Verify modality.json

```bash
cat /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/modality.json
```

Should show GR00T format (with "state", "action", "video", "annotation").

### 2. Re-run Mini-MVP

```bash
cd ~/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

**Expected:** Should pass validation and start loading dataset.

### 3. Check for New Errors

Look for different errors after the validation passes:
```bash
tail -30 /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test/mini_mvp_training.log
```

---

## Reference: SO-100 Example

The correct format is based on NVIDIA's official example:
`/home/jrobot/project/Isaac-GR00T/examples/SO-100/so100_dualcam__modality.json`

This is the template for SO-100 robots with dual cameras (front + wrist).

---

## Next Potential Issues

After fixing modality.json, watch for:

1. **Dataset loading errors** - Video codec compatibility
2. **Model loading** - Downloading GR00T-N1.5-3B weights
3. **VRAM issues** - 18-20GB expected for batch_size=4
4. **Data shape mismatches** - Verify camera resolution (480x640)

---

## Summary

**Problem:** Used LeRobot-style modality.json, but GR00T expects different format

**Solution:** Updated to GR00T format with:
- Index-based state/action definitions
- `original_key` mappings for videos
- Annotation mapping for task descriptions

**Status:** ✅ Fixed
- Dataset modality.json updated
- Training script updated
- Ready to re-run mini-MVP

**Next:** Re-run training and monitor for new errors.
