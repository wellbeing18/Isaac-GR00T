# GR00T Finetuning Performance Investigation Report & Action Plan

**Status**: Completed
**Date**: 2025-12-14
**Objective**: Root cause analysis of poor inference performance in GR00T finetuned model.

## 1. Executive Summary
The investigation confirms that the **data pipeline (collection, conversion, loading) is correct**, but the **training duration and inference configuration are suboptimal**. The poor performance stems from two key issues:
1.  **Underfitting**: The model has high Open Loop MSE (~15.6), indicating it hasn't learned the task effectively after 10k steps (~3 epochs).
2.  **Inference Configuration**: Temporal Ensembling was DISABLED in the latest inference run, leading to jerky, non-smooth control.

## 2. Pipeline Overview

```mermaid
graph TD
    A[Data Collection] -->|collect_xlerobot_data.py| B(LeRobot v3 Dataset)
    B -->|convert_lerobot_v3_to_groot.py| C(GR00T v2 Dataset)
    C -->|train_groot_mvp.sh| D(Finetuned Model + LoRA)
    D -->|infer_groot_async.py| E[Robot Inference]
    
    subgraph Data Collection
    A1[Robot Hardware] --> A
    A2[Task Presets] --> A
    A3[ACTION_FPS=30] --> A
    end
    
    subgraph Data Conversion
    B1[Modality Mapping] --> C
    B2[Video Splitting (Per-Episode)] --> C
    B3[Parquet Splitting (Per-Episode)] --> C
    B4[Stats Calculation] --> C
    end
    
    subgraph Training
    C1[Data Config: so100_dualcam] --> D
    C2[Steps: 10k (~3 Epochs)] --> D
    C3[Embodiment: new_embodiment] --> D
    end
    
    subgraph Inference
    D1[Async Producer (~5Hz)] --> E
    D2[Async Consumer (30Hz)] --> E
    D3[Temporal Ensembling (DISABLED!)] --> E
    end
```

## 3. Module-by-Module Analysis

### 3.1 Data Collection & Conversion (PASSED)
- **Verification**: Ran `verify_dataset_integrity.py`.
- **Results**:
    - **Video Format**: Correct (per-episode).
    - **Timestamps**: Correct (start near 0).
    - **Frame Uniqueness**: Passed (no "blind model" issue).
    - **Stats**: Valid (non-scalar).
- **Conclusion**: The dataset is healthy and ready for training.

### 3.2 Training Process (WARNING)
- **Config**: `so100_dualcam` correctly maps modalities and uses Min-Max normalization.
- **Embodiment**: Uses `new_embodiment` (ID 31). This is standard for custom robots.
- **Issue**: Training for **10,000 steps** with a dataset of **50,000 frames** (batch size 16) results in only **~3.2 epochs**.
    - For a new embodiment trained from scratch (LoRA), this is likely insufficient.
    - **Open Loop Eval**: MSE of **15.6** confirms the model is underfitting (target MSE should be < 5.0).

### 3.3 Inference Process (FAILED)
- **Log Analysis**: `infer_groot_async_20251214_101456.log`
- **Issue 1**: `[ASYNC] Temporal ensembling DISABLED - using only latest prediction`.
    - This flag (`--no-ensemble`) forces the robot to jump between trajectories every ~6 steps (5Hz producer vs 30Hz consumer), causing jerky motion.
- **Issue 2**: Inference rate is ~5.2Hz. While acceptable with ensembling, it is too slow without it.

## 4. Comparison with Reference
- **Pushpakcc/gr00t-so100_dualcam-finetuned**:
    - Uses `SO-101 robot arm` description.
    - Likely trained for more epochs or on a larger dataset.
- **Our Setup**:
    - Dataset: ~70 episodes (Small).
    - Training: 10k steps (Short).

## 5. Action Plan

### Immediate Fixes
1.  **Enable Temporal Ensembling**:
    - Ensure `infer_groot_async.py` is called **WITHOUT** `--no-ensemble`.
    - Verify `max_predictions` is set to 4 (default).

2.  **Increase Training Duration**:
    - Increase `MAX_STEPS` to **30,000** or **50,000** (approx 10-15 epochs).
    - Monitor Open Loop MSE. Do not deploy until MSE < 5.0.

3.  **Data Augmentation / Collection**:
    - 70 episodes is the bare minimum. Aim for **100+ episodes** for robust generalization.

### Long Term
- **Embodiment Tuning**: If `new_embodiment` remains stubborn, experiment with mapping to `gr1` (ID 24) to leverage humanoid priors, though this is a fallback.
- **Hardware**: Ensure camera latency is minimized to keep the "Producer" rate closer to 10Hz.

## 6. Commands to Run

**1. Resume Training (Longer):**
```bash
# Edit custom/scripts/train_groot_mvp.sh
# Set MAX_STEPS=30000
./custom/scripts/train_groot_mvp.sh
```

**2. Correct Inference:**
```bash
python custom/scripts/infer_groot_async.py \
    --model-path /path/to/best_checkpoint \
    --action-horizon 16 \
    --use-first-n-actions 8 \
    # DO NOT ADD --no-ensemble
```
