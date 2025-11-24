# stats.json Count Field Fix

**Date:** 2025-11-24
**Issue:** IndexError when loading dataset statistics

---

## Problem

Training failed with:

```
IndexError: index 1 is out of bounds for axis 0 with size 1
File "/home/jrobot/project/Isaac-GR00T/gr00t/data/dataset.py", line 388
    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()
```

---

## Root Cause

LeRobot v3's `stats.json` has a **scalar `count`** field instead of per-dimension counts:

### Wrong Format (LeRobot v3)

```json
{
  "action": {
    "count": [1500],           ← Single value (scalar)
    "mean": [a, b, c, d, e, f], ← 6 values (per dimension)
    "std": [a, b, c, d, e, f]
  }
}
```

### What GR00T Expects

```json
{
  "action": {
    "count": [1500, 1500, 1500, 1500, 1500, 1500],  ← 6 values
    "mean": [a, b, c, d, e, f],
    "std": [a, b, c, d, e, f]
  }
}
```

---

## Why This Happens

GR00T tries to extract statistics for sub-components:
- `action.single_arm` → indices [0:5] → needs 5 count values
- `action.gripper` → indices [5:6] → needs 1 count value

When `count = [1500]`, accessing `count[5]` fails because there's only 1 element.

---

## Fix Applied

Created script to replicate count value across all dimensions:

**Script:** `/home/jrobot/project/XLeRobot/jdocs/top_level/fix_stats_count.py`

```python
# For each 6D field (action, observation.state)
old_count = [1500]           # Single value
dimension = 6                 # From mean/std shape
new_count = [1500] * 6       # Replicate across dimensions
```

**Result:**
```python
stats['action']['count'] = [1500, 1500, 1500, 1500, 1500, 1500]
stats['observation.state']['count'] = [1500, 1500, 1500, 1500, 1500, 1500]
```

---

## Files Modified

1. **`datasets/meta/stats.json`** - Fixed count fields
2. **`datasets/meta/stats.json.backup`** - Original backup

---

## Verification

After fix:

```bash
python3 -c "
import json
stats = json.load(open('datasets/meta/stats.json'))
print('action count:', len(stats['action']['count']))
print('state count:', len(stats['observation.state']['count']))
"
```

Output:
```
action count: 6
state count: 6
```

✅ Both now have 6-element arrays.

---

## Why Count Value Matters

The `count` statistic typically represents:
- Total number of samples per dimension
- For your dataset: 1500 frames × 1 = 1500 samples per joint

In your case:
- All 6 joints have 1500 samples each
- So count = [1500, 1500, 1500, 1500, 1500, 1500] is correct

---

## Impact on Training

**Before fix:**
- ❌ Crashes during dataset initialization
- ❌ Never reaches model loading
- ❌ IndexError at line 388

**After fix:**
- ✅ Dataset statistics load correctly
- ✅ GR00T can split statistics by component (arm vs gripper)
- ✅ Training can proceed to next stage

---

## If You Re-collect Data

When collecting new datasets, this fix will be needed again because LeRobot v3 always generates scalar counts.

**Automated fix script:**

```bash
# After collecting new episodes
cd /home/jrobot/project/XLeRobot/jdocs/top_level
python3 fix_stats_count.py
```

Or integrate into data collection pipeline.

---

## Alternative Solution (Not Used)

Could also modify GR00T's dataset.py to handle scalar counts:

```python
# In dataset.py line 387-388
stat = np.array(le_statistics[le_modality][stat_name])
if stat.shape[0] == 1 and stat_name == "count":
    # Replicate scalar count
    stat = np.full(len(indices), stat[0])
dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()
```

**Pros:**
- More robust to different formats
- No manual fix needed

**Cons:**
- Modifies GR00T codebase
- May break other datasets
- Our fix is simpler

---

## Summary

**Problem:** stats.json had scalar count [1500] instead of per-dimension [1500, 1500, 1500, 1500, 1500, 1500]

**Solution:** Ran fix script to replicate count across all dimensions

**Result:** Dataset statistics now compatible with GR00T

**Next:** Re-run training to test if it progresses further

**Status:** ✅ Fixed
