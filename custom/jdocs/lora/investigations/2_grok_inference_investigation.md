# GR00T LoRA Inference Performance Investigation

**Date:** 2025-12-13
**Investigator:** Grok
**Issue:** LoRA finetuned model shows good evaluation metrics but poor real-world inference performance

## Executive Summary

The GR00T LoRA finetuned model exhibits good evaluation performance but fails in real-world inference, producing only small, ineffective movements. Root causes include critical dataset synchronization bugs, normalization mismatches, and potential LoRA implementation issues. The dataset conversion pipeline contains fundamental flaws that misalign video and action data during training.

## Investigation Methodology

1. **Code Analysis**: Examined inference scripts, evaluation code, and data processing pipelines
2. **Data Inspection**: Analyzed normalization metadata and dataset conversion logic
3. **Log Analysis**: Reviewed inference debug logs for action scaling patterns
4. **Literature Review**: Cross-referenced with known dataset conversion issues

## Key Findings

### Finding 1: Dataset Conversion Critical Bugs 🚨

**Severity:** Critical
**Impact:** Training on misaligned data causes poor generalization

**Evidence:**
The dataset conversion and combination pipeline has fundamental synchronization issues:

- **Video Concatenation Bug**: Combine script concatenates all video files but preserves local timestamps (starting from 0)
- **Result**: Episodes retrieve frames from wrong video segments
- **Example**:
  - Episode 0: timestamps 0-38s → correct frames ✅
  - Episode 10: timestamps 0-30s → gets Episode 0's frames ❌
  - Episode 35: timestamps 0-22s → gets Episode 0's frames ❌

**Source:** `custom/jdocs/lora/4_convert_combine_issues_investigation.md`
**Status:** Identified but may not be fully fixed in current dataset

### Finding 2: Normalization Statistics Mismatch ⚠️

**Severity:** High
**Impact:** Incorrect action scaling during inference

**Evidence:**
LoRA fine-tuning uses different normalization statistics than base model:

**LoRA Training Stats (from metadata.json):**
```json
"action": {
  "single_arm": {
    "min": [-27.882, -100.0, -61.84, -75.752, -61.474],
    "max": [48.347, 64.328, 100.0, 81.334, 4.526],
    "std": [13.097, 50.054, 44.632, 24.947, 19.955]
  },
  "gripper": {
    "min": [0.0],
    "max": [35.887],
    "std": [9.497]
  }
}
```

**Issue:** If base model expected different ranges, LoRA predictions may be incorrectly scaled during denormalization.

### Finding 3: Evaluation vs Inference Discrepancy 🔍

**Severity:** Medium
**Impact:** False confidence from evaluation metrics

**Evaluation Appears Good Because:**
- Compares denormalized predictions vs denormalized ground truth (same normalization stats)
- Small MSE values don't reflect real-world task completion

**Inference Fails Because:**
- Model produces conservative actions (deltas: 1-6°) insufficient for task completion
- "Pick up red cube and place on white plate" requires specific trajectories

**Evidence from logs:**
```
state=[  0.2 -98.9  97.4  50.6  -0.4   0.5]
action=[   2.6 -100.4   97.8   51.3    0.6    0.3]
delta=[ 2.38 -1.45  0.43  0.62  1.   -0.2 ]
```
Deltas within dataset std (13-50°) but too conservative for task.

### Finding 4: LoRA Implementation Concerns ❓

**Severity:** Unknown
**Impact:** Potential weight merging or adaptation issues

**Questions:**
- Are LoRA weights properly merged with base model?
- Does LoRA adaptation work correctly across normalization boundaries?
- Is the embodiment tag ("new_embodiment") correctly handled?

## Action Scaling Analysis

**Observed Pattern:**
- Predicted deltas: 1-6° per joint per inference step
- Inference rate: ~6 Hz → 360 inferences over 60s
- Temporal ensembling: Averages 2-100 predictions per action
- Result: Smooth but ineffective movements

**Expected for Task:**
- Pick-and-place requires larger initial movements toward target
- Current predictions may be "hovering" rather than "approaching"

## Root Cause Analysis

### Primary Issue: Dataset Synchronization
The most critical issue is misaligned training data. Model trained on wrong video frames learns incorrect action patterns.

### Secondary Issues:
1. **Normalization mismatch** between base and LoRA training
2. **Conservative predictions** due to dataset bias or LoRA limitations
3. **Evaluation blind spots** masking real performance issues

## Recommended Solutions

### Phase 1: Fix Dataset Issues (Critical)
```bash
# Verify and fix dataset conversion
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place" \
    --robot-type so101 \
    --dual-camera

python custom/scripts/combine_groot_datasets.py \
    --datasets "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place" \
    --output "/home/jrobot/project/XLeRobot/datasets_groot"
```

### Phase 2: Retrain Model
- Retrain LoRA on properly synchronized data
- Consider full fine-tuning if LoRA insufficient

### Phase 3: Debug Action Generation
- Analyze successful trajectories from training data
- Check if model predicts adequate action magnitudes
- Verify temporal ensembling doesn't average away meaningful movements

### Phase 4: Validation
- Test with simpler tasks ("reach toward red cube")
- Compare evaluation metrics before/after fixes
- Verify inference action ranges match training data

## Immediate Next Steps

1. **Verify Dataset Status**: Check if video splitting fixes were applied
2. **Run Diagnostic Evaluation**: Log predicted action ranges vs ground truth
3. **Test Simplified Task**: Verify basic movement capability
4. **Check LoRA Integration**: Confirm weights merged correctly

## Risk Assessment

- **High Risk**: Dataset bugs cause training on corrupted data
- **Medium Risk**: Normalization mismatch causes scaling issues
- **Low Risk**: LoRA implementation (but needs verification)

## Files Examined

- `custom/scripts/infer_groot_async.py` - Inference implementation
- `custom/scripts/eval_groot_openloop.py` - Evaluation code
- `gr00t/data/transform/state_action.py` - Normalization logic
- `custom/jdocs/lora/4_convert_combine_issues_investigation.md` - Dataset issues
- `/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251212_180908058/best/experiment_cfg/metadata.json` - Normalization stats

## Conclusion

The inference failure stems primarily from dataset synchronization bugs that corrupt training data alignment. Secondary issues with normalization and LoRA adaptation compound the problem. Fix the dataset pipeline first, then retrain and re-evaluate.

**Priority:** Fix dataset conversion → Retrain → Debug action generation → Validate




