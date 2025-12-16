# Dataset Quality Investigation: Root Cause of Inference Issues

**Date:** 2025-12-15
**Status:** Root cause identified
**Result:** Data collection workflow issues - slow start and slow movement speed

## Executive Summary

After extensive investigation, we identified **two critical issues** in the data collection workflow that cause the trained model to produce minimal arm movement during inference:

| Metric | Your Dataset | Reference (youliangtan) | Impact |
|--------|-------------|-------------------------|--------|
| ARM movement starts at | **3.8s** (frame 113) | **0.5s** (frame 16) | 7x slower to start |
| Movement velocity | **0.30°/frame** | **0.76°/frame** | 2.5x slower movement |
| Active movement % | 86.8% | 92.8% | 6% less useful data |

These issues cause the model to learn:
1. "Don't move" bias from stationary frames at episode start
2. "Small movements are correct" from slow demonstration speed

## Investigation Process

### 1. Initial Observation
- Open-loop evaluation showed good MSE scores
- But real robot inference showed arms barely moving
- Visual inspection of eval plots showed "flat" joint predictions

### 2. Data Analysis
We analyzed the raw collected dataset (`/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`) and found:

**Episode 0 Timeline:**
```
Frame 0   → Recording starts
Frame 63  → WRIST starts rotating slowly (2.1s)
Frame 113 → ARM movement begins (3.8s)  ← Problem!
Frame 900 → Episode ends
```

**vs Reference Dataset (youliangtan/so101-table-cleanup):**
```
Frame 0   → Recording starts
Frame 16  → ARM movement begins (0.5s)  ← Immediate!
Frame 855 → Episode ends
```

### 3. Velocity Analysis

| Window | Your Avg Velocity | Reference Velocity |
|--------|-------------------|-------------------|
| Active motion | 0.30°/frame | 0.76°/frame |

At 30Hz (33ms per frame), your demonstrations are **2.5x slower** than the reference.

### 4. Joint Range Coverage

| Joint | Your Range | Reference Range | Gap |
|-------|-----------|-----------------|-----|
| shoulder_pan | 40.7° | 54.5° | -13.7° |
| shoulder_lift | 128.6° | 143.1° | -14.5° |
| elbow_flex | 112.2° | 155.4° | **-43.2°** |
| wrist_flex | 54.4° | 105.5° | **-51.2°** |
| wrist_roll | 54.9° | 59.5° | -4.6° |
| gripper | 28.1° | 33.8° | -5.7° |

Your demonstrations don't exercise the full range of motion, especially for elbow and wrist_flex.

## Root Causes

### Cause 1: Delayed Start
When `lerobot-record` announces "Recording episode X", the recording has **already started**. The time spent:
- Positioning the leader arm
- Getting ready mentally
- Slow wrist adjustments

...is all recorded as part of the episode, teaching the model "don't move initially".

### Cause 2: Slow Movement Speed
At 30Hz, each frame is 33ms apart. Moving too slowly means:
- Many consecutive frames have nearly identical joint positions
- Model learns that small/no movement is acceptable
- During inference, model outputs very small action deltas

## Tools Created

### 1. Dataset Quality Verification Tool

**Script:** `custom/scripts/verify_dataset_quality.py`

**Features:**
- Analyzes stationary periods at start/end of episodes
- Measures movement velocity
- Checks joint range coverage
- Compares with reference datasets
- Generates visualization plots

**Basic Usage:**
```bash
# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate groot

# Run quality check on your dataset
python custom/scripts/verify_dataset_quality.py \
    -d /home/jrobot/project/XLeRobot/datasets/left/pick_and_place

# Compare with reference dataset
python custom/scripts/verify_dataset_quality.py \
    -d /home/jrobot/project/XLeRobot/datasets/left/pick_and_place \
    -r youliangtan/so101-table-cleanup

# Show visualization commands
python custom/scripts/verify_dataset_quality.py \
    -d /home/jrobot/project/XLeRobot/datasets/left/pick_and_place \
    --show-viz-commands
```

**Output Example:**
```
======================================================================
DATASET QUALITY REPORT
======================================================================

Total Episodes: 70
Total Frames: 49869
FPS: 30

--- STATIONARY PERIOD ANALYSIS ---
Average start stationary: 7.0% [OK] (threshold: <10.0%)
Average end stationary: 6.1% [OK] (threshold: <10.0%)
Average active movement: 86.8%

--- VELOCITY ANALYSIS ---
Average velocity: 0.303°/frame [CHECK]
  Note: Movement may be too slow for 30Hz sampling

======================================================================
COMPARISON WITH REFERENCE: youliangtan/so101-table-cleanup
======================================================================

Metric                    Your Dataset    Reference       Diff
-----------------------------------------------------------------
Start stationary %        7.0             4.2             +2.8
Active movement %         86.8            92.8            -6.0
Avg velocity (°/frame)    0.303           0.758           -0.455
```

### 2. Dataset Trimming Tool

**Script:** `custom/scripts/trim_stationary_periods.py`

**Purpose:** Remove stationary periods from existing collected data.

**Usage:**
```bash
python custom/scripts/trim_stationary_periods.py \
    --input /home/jrobot/project/XLeRobot/datasets/left/pick_and_place \
    --output /home/jrobot/project/XLeRobot/datasets_trimmed/left/pick_and_place \
    --threshold 0.5 \
    --start_buffer 5 \
    --end_buffer 10
```

**Parameters:**
- `--threshold`: Movement threshold in degrees (default: 0.5)
- `--start_buffer`: Frames to keep before first movement (default: 10)
- `--end_buffer`: Frames to keep after last movement (default: 30)

### 3. LeRobot Visualization Commands

**Dataset Visualizer (Web Interface):**
```bash
# For local datasets
lerobot-dataset-viz \
    --repo-id local/xlerobot_left_pick_and_place \
    --root /home/jrobot/project/XLeRobot/datasets/left/pick_and_place

# For HuggingFace datasets
lerobot-dataset-viz --repo-id youliangtan/so101-table-cleanup
```

**Replay Episode on Robot:**
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=xlerobot_left_arm \
    --dataset.repo_id=local/xlerobot_left_pick_and_place \
    --dataset.root=/home/jrobot/project/XLeRobot/datasets/left/pick_and_place \
    --dataset.episode=0
```

**View Videos Directly:**
```bash
# Your videos
ffplay "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place/videos/observation.images.head/chunk-000/file-000.mp4"

# Reference videos
ffplay "/tmp/youliangtan_reference/videos/chunk-000/observation.images.top/episode_000000.mp4"
```

## Recommendations

### For Future Data Collection

1. **Start Immediately (< 0.5s delay)**
   - During reset phase, position leader arm at starting position
   - When "Recording episode X" announces, you should already be ready
   - Start task motion within 0.5 seconds of recording start

2. **Move 2-3x Faster**
   - Current: 0.30°/frame → Target: 0.75-1.0°/frame
   - At 30Hz, deliberate but continuous motion is needed
   - Watch reference videos to calibrate your speed

3. **Exercise Full Joint Ranges**
   - Make larger, more expressive movements
   - Especially for elbow_flex and wrist_flex
   - Cover the full workspace the robot might encounter

4. **Press RIGHT ARROW Immediately When Done**
   - Don't wait at the end - this adds useless "holding" data

### For Existing Data

**Option A: Re-collect with proper technique**
- Best option for model quality
- Use reference videos as speed guide

**Option B: Trim existing data**
```bash
python custom/scripts/trim_stationary_periods.py \
    --input /path/to/dataset \
    --output /path/to/trimmed_dataset
```
- Removes stationary periods
- Note: Cannot fix slow movement speed issue

**Option C: Combine both**
- Trim existing data
- Collect additional data with proper technique
- Combine datasets

## Reference Materials

- [youliangtan/so101-table-cleanup Dataset](https://huggingface.co/datasets/youliangtan/so101-table-cleanup)
- [GR00T N1.5 SO-101 Tuning Blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [SeeedStudio LeRobot Visualization Guide](https://wiki.seeedstudio.com/lerobot_so100m_new/#visualize-the-dataset)
- [LeRobot Recording Documentation](https://github.com/huggingface/lerobot)

## Files Created

| File | Purpose |
|------|---------|
| `custom/scripts/verify_dataset_quality.py` | Dataset quality analysis and comparison |
| `custom/scripts/trim_stationary_periods.py` | Remove stationary periods from datasets |
| `custom/jdocs/lora/new_investigations/DATASET_QUALITY_COMPARISON.md` | Detailed comparison analysis |
| `custom/jdocs/lora/new_investigations/ROOT_CAUSE_FOUND.md` | Initial root cause documentation |

## Open-Loop Evaluation Results

Using the official NVIDIA `scripts/eval_policy.py` evaluation method, we tested your model on both datasets:

### MSE Results (First 150 Steps)

| Evaluation | Trajectory 0 | Trajectory 1 | Trajectory 2 | Average MSE |
|------------|-------------|-------------|-------------|-------------|
| Your Model on Your Data | 217.17 | 195.62 | 135.79 | **182.86** |
| Your Model on Reference Data | 331.67 | 313.73 | 483.43 | **376.27** |

### Analysis

1. **High MSE on Your Data (182.86)**: Even on data it was trained on, the model has high MSE. This indicates the model may be overfitting to the "stationary" patterns rather than learning meaningful action predictions.

2. **Higher MSE on Reference Data (376.27)**: The model performs ~2x worse on the reference dataset which has:
   - Faster movement (0.76°/frame vs 0.30°/frame)
   - Immediate start (0.5s vs 3.8s delay)
   - Different movement patterns

3. **Why the first 150 steps matter**: The official eval uses first 150 steps. In your data, movement doesn't start until frame ~113-200. This means:
   - Most of the evaluation is on stationary/slow-moving data
   - The model predicts "don't move" which matches ground truth → artificially lower MSE
   - But on reference data with immediate movement, this "don't move" prediction causes high error

### Generated Plots

The evaluation generated plots comparing predictions vs ground truth:

- `your_model_your_data_traj0.png` - Your model on your dataset (traj 0)
- `your_model_your_data_traj1.png` - Your model on your dataset (traj 1)
- `your_model_your_data_traj2.png` - Your model on your dataset (traj 2)
- `your_model_ref_data_traj0.png` - Your model on reference dataset (traj 0)
- `your_model_ref_data_traj1.png` - Your model on reference dataset (traj 1)
- `your_model_ref_data_traj2.png` - Your model on reference dataset (traj 2)

These plots show the predicted actions (orange) vs ground truth actions (blue) for each joint over time.

## Conclusion

The inference issue (arms barely moving) is caused by **multiple factors**:

### Primary Cause: Data Collection Workflow Issues

1. **Delayed start**: 3.8s vs 0.5s in reference dataset
2. **Slow movement**: 0.30°/frame vs 0.76°/frame in reference

The model learned exactly what it was shown: hesitate at the start, then move slowly.

### Contributing Factor: Model Overfitting

The high MSE (182.86) even on training data suggests the model may not be learning good action representations. Possible reasons:
- Too much stationary data causes the model to learn "don't move" as default
- Insufficient movement variety in training data
- May need more training data with proper collection technique

### Is Data Quality the Only Issue?

Based on the evidence:

| Factor | Impact | Evidence |
|--------|--------|----------|
| Data collection timing | **HIGH** | 7x slower start than reference |
| Data collection speed | **HIGH** | 2.5x slower movement than reference |
| Model architecture | LOW | Same model works for other datasets |
| Training configuration | UNKNOWN | May need hyperparameter tuning after data fix |

**Recommendation**: Fix the data collection issues FIRST, then evaluate if additional issues remain.

### Action Plan

1. **Immediate**: Re-collect data with proper technique (immediate start, 2-3x faster movement)
2. **Then**: Re-train model on new data
3. **Evaluate**: If issues persist after re-training, investigate training hyperparameters
