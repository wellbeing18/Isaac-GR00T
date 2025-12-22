# GR00T LoRA Inference Performance Investigation - New Training Run

**Date:** 2025-12-13
**Investigator:** Grok
**Issue:** New LoRA training (20251213_155740026) shows excellent evaluation (3.63° MAE) but poor real-world inference performance
**Training:** Used unconverted LeRobot dataset directly

## Executive Summary

The new LoRA training run shows dramatically improved evaluation metrics (3.63° MAE vs previous runs), but real-world inference still fails with ineffective small movements. The model demonstrates excellent memorization of training trajectories but poor generalization to new task instances. Root cause appears to be insufficient visual generalization and task understanding for real-world deployment.

## Investigation Methodology

1. **Training Analysis**: Compared new run with previous training attempts
2. **Evaluation Review**: Analyzed excellent evaluation metrics (3.63° MAE, 93.5% Acc@10)
3. **Inference Log Analysis**: Examined action prediction patterns and movement trajectories
4. **Dataset Verification**: Confirmed training used unconverted LeRobot dataset
5. **Root Cause Analysis**: Identified generalization vs memorization issues

## Key Findings

### Finding 1: Training Used Unconverted Dataset ⚠️

**Critical Discovery:** Training was performed on unconverted LeRobot dataset directly.

**Evidence:**
```bash
# Training command used:
dataset_path: ['/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place']

# But evaluation command in report shows:
--dataset /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place
```

**Impact:**
- Training bypassed the convert/combine pipeline entirely
- No dataset synchronization fixes were applied
- Model trained on potentially corrupted/misaligned data
- But evaluation still shows excellent performance

### Finding 2: Excellent Evaluation Performance ✅

**Training Results:**
- **MAE:** 3.63° (excellent, much better than previous runs)
- **Acc@5:** 76.2%
- **Acc@10:** 93.5%
- **Per-joint MAE:** 2.2° - 5.4° (shoulder_pan/wrist_roll best, elbow/wrist_flex worst)

**Evaluation Details:**
- Tested on same dataset as training
- 300 samples evaluated
- Baseline state-copy MAE: 1.35° (shows room for improvement)

### Finding 3: Inference Shows Exploratory Behavior, Not Task Execution ❌

**Movement Patterns Observed:**
```
Chunk 0: Small adjustments around home position
  delta: [ 0.89  1.01 -0.24 -8.11 -2.07  0.06]  # Mostly wrist adjustments

Chunk 1: Some elbow movement
  delta: [-3.08  1.15 -8.1  -1.29  2.96  0.65]  # Elbow drop, gripper open

Chunk 2-4: Continued small corrections
  deltas: 1-6° movements, no clear approach trajectory
```

**Problem:** Model makes exploratory adjustments but doesn't commit to task-directed movements toward the red cube.

### Finding 4: Memorization vs Generalization Gap 📊

**Evaluation (Memorization):** Excellent - model accurately predicts actions for exact training trajectories
**Inference (Generalization):** Poor - model fails on new task instances despite same hardware setup

**Likely Causes:**
1. **Visual Domain Shift:** Real camera inputs differ from training data (lighting, angles, object positions)
2. **Task Understanding:** Language model may not effectively guide policy toward specific task
3. **Training Data Limitations:** Model learned specific trajectories but not general task-solving
4. **LoRA Adaptation Limits:** Fine-tuning insufficient for embodiment/domain shift

## Action Pattern Analysis

**Predicted Movements:**
- **Magnitude:** 1-8° per joint per action (conservative)
- **Pattern:** Small corrections rather than purposeful approach
- **Trajectory:** No clear progression toward cube grasping/placement
- **Horizon:** 16-step predictions show local adjustments, not task completion

**Expected for Task:**
- **Approach Phase:** Large elbow/shoulder movements to position near cube
- **Grasp Phase:** Wrist/gripper coordination for precise grasping
- **Placement Phase:** Transport and positioning on plate

**Observed:** Model stays in "exploration mode" around home position.

## Root Cause Analysis

### Primary Issue: Insufficient Generalization
Model demonstrates excellent trajectory memorization but poor task generalization.

### Secondary Issues:
1. **Training Data Quality:** Used unconverted dataset (potential alignment issues)
2. **Visual Mismatch:** Real-world camera inputs differ from training data
3. **Task Prompt Processing:** Language understanding may not guide policy effectively
4. **LoRA Limitations:** Fine-tuning may not capture sufficient task-level understanding

### Technical Details:
- **Normalization:** Same stats used (from unconverted dataset)
- **Hardware:** Same dual-camera setup (head + wrist)
- **Task:** Same format ("pick up the red cube and place it on the white plate")
- **Inference:** Async producer-consumer architecture working correctly

## Recommended Solutions

### Phase 1: Improve Training Data Quality
```bash
# Use properly converted and synchronized dataset
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place" \
    --robot-type so101 \
    --dual-camera

python custom/scripts/combine_groot_datasets.py \
    --datasets "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place" \
    --output "/home/jrobot/project/XLeRobot/datasets_groot"
```

### Phase 2: Enhance Training Diversity
- **Multi-task Training:** Train on varied object positions/scenarios
- **Data Augmentation:** Add visual augmentations for robustness
- **Curriculum Learning:** Start with simple tasks, progress to complex

### Phase 3: Improve Task Understanding
- **Prompt Engineering:** Test different task formulations
- **Language Fine-tuning:** Fine-tune language model component
- **Task Embeddings:** Verify task encoding effectiveness

### Phase 4: Validation and Testing
- **Visual Debugging:** Record and analyze camera inputs during inference
- **Sim-to-Real Transfer:** Test in simulation first
- **Progressive Tasks:** Start with simpler tasks (reach, touch) before pick-and-place

## Immediate Debugging Steps

1. **Verify Visual Inputs:** Record camera images during inference to check for corruption/domain shift
2. **Test Simpler Tasks:** Try "move toward red cube" or "touch red cube"
3. **Analyze Task Embeddings:** Check if language model produces consistent task representations
4. **Compare Training vs Inference Visuals:** Ensure camera calibration and preprocessing match

## Risk Assessment

- **High Risk:** Training on unconverted dataset may have introduced subtle corruption
- **Medium Risk:** Model lacks generalization capability for real deployment
- **Low Risk:** LoRA implementation (weights load/merge correctly)

## Files Analyzed

- Training: `/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_155740026/`
- Evaluation: `evaluation_results.json` (3.63° MAE, 93.5% Acc@10)
- Inference: `custom/logs/infer_groot_async_20251213_195749.log`
- Dataset: `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place` (unconverted)

## Conclusion

The new training run achieved excellent evaluation performance (3.63° MAE) but still fails real-world inference due to poor generalization. The model memorizes training trajectories well but cannot adapt to new task instances. Key issues include training data quality, visual domain shift, and insufficient task-level understanding.

**Next Steps:**
1. Fix dataset conversion pipeline
2. Retrain on properly synchronized data
3. Add training diversity for better generalization
4. Debug task understanding and visual processing

**Priority:** Dataset fixes → Enhanced training → Task understanding → Real-world validation







