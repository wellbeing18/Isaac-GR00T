# GR00T N1.5 LoRA Finetuning Guide for SO-101 (v2)

## Complete Guide with Monitoring, Evaluation, and Diagnosis

**Last Updated:** 2025-12-12 (v2.3 - Updated for One-Video-Per-Episode format)
**Robot:** SO-ARM101 Left Arm (6 DOF)
**GPU:** RTX 5090 (24GB VRAM)
**Model:** GR00T N1.5 (3B parameters)
**Approach:** LoRA (Low-Rank Adaptation)

**Status Summary:**
| Phase | Status |
|-------|--------|
| Mini-MVP (100 steps) | ✅ COMPLETED |
| LoRA Loading Fix | ✅ COMPLETED |
| Monitoring Scripts | ✅ COMPLETED |
| MVP Training (5K steps) | 🎯 READY TO RUN |
| Full Training (10K steps) | ⏳ PENDING |

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v2](#whats-new-in-v2)
3. [Quick Start - 5 Minute Setup](#quick-start---5-minute-setup)
4. [Complete Workflow](#complete-workflow)
   - [Phase 1: Data Preparation](#phase-1-data-preparation)
   - [Phase 2: Pre-Training Verification](#phase-2-pre-training-verification)
   - [Phase 3: Training](#phase-3-training)
   - [Phase 4: Post-Training Evaluation](#phase-4-post-training-evaluation)
   - [Phase 5: Inference & Deployment](#phase-5-inference--deployment)
5. [Script Reference](#script-reference)
6. [Training Configurations](#training-configurations)
7. [Critical Fixes Applied](#critical-fixes-applied)
8. [Troubleshooting](#troubleshooting)
9. [Progress Tracking](#progress-tracking)

---

## Overview

### Why GR00T with LoRA?

**GR00T N1.5** is NVIDIA's 3B parameter robot foundation model. Using **LoRA (Low-Rank Adaptation)** makes finetuning practical on consumer hardware.

| Aspect | Full Finetuning | LoRA Finetuning |
|--------|-----------------|-----------------|
| VRAM Required | 58-70GB | **18-20GB** |
| Episodes Needed | 500+ | **50-100** |
| Training Time | Days | **1-2 hours** |
| Trainable Params | 3B (100%) | **3.2M (0.12%)** |

### Training Strategy

```
Mini-MVP (100 steps, 5-10 min)
└── Validates pipeline ✅ COMPLETED

MVP (5000 steps, ~1 hour)
└── First usable model → 🎯 CURRENT TARGET

Full Training (10,000 steps, ~2 hours)
└── Production model
```

---

## What's New in v2

### Critical Bug Fix: LoRA Adapter Loading ✅

**Problem Found:** GR00T training saves LoRA adapters as separate files (`adapter_config.json` + `adapter_model.safetensors`), but the standard `Gr00tPolicy` doesn't load them. This is identical to the Pi0.5 PEFT key mismatch issue.

**Solution Applied:** Updated `infer_groot_so101.py` with:
- `is_lora_checkpoint()` - Detects LoRA checkpoints
- `load_groot_with_lora()` - Loads base model, applies PEFT adapter, merges weights
- Auto-detection in inference class

### New Monitoring & Evaluation Scripts ✅

| Script | Purpose | Status |
|--------|---------|--------|
| `verify_groot_training_setup.py` | Pre-training dataset/model verification | ✅ Created |
| `evaluate_groot_checkpoint.py` | Post-training MAE evaluation | ✅ Created |
| `diagnose_groot_inference.py` | Inference behavior diagnosis | ✅ Created |
| `combine_groot_datasets.py` | Multi-task dataset combination | ✅ Created |

### Integrated Training Pipeline ✅

The training scripts now include:
- `train_groot_mini_mvp.sh` - Quick validation (100 steps, ~5 min)
- `train_groot_mvp.sh` - Full training (5K steps, ~1 hour) with best checkpoint selection

The MVP script (`train_groot_mvp.sh`) includes:
1. Pre-training verification
2. 5000 steps training
3. Automatic checkpoint evaluation
4. Best checkpoint selection (lowest MAE)
5. Inference diagnosis
6. Comprehensive report generation

---

## Quick Start - 5 Minute Setup

### For Existing Dataset (Already Converted to GR00T Format)

```bash
# 1. Activate environment
conda activate groot
cd /home/jrobot/project/Isaac-GR00T

# 2. Verify dataset (optional but recommended)
python custom/scripts/verify_groot_training_setup.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot

# 3. Run 5K training with evaluation (uses train_groot_mvp.sh)
bash custom/scripts/train_groot_mvp.sh

# 4. Run inference with best checkpoint
python custom/scripts/infer_groot_so101.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_TIMESTAMP/best \
    --task "pick red_cube from center"
```

### For New Dataset (LeRobot v3 Format)

```bash
# 1. Convert to GR00T format
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path /path/to/lerobot_v3_dataset \
    --robot-type so101 \
    --dual-camera \
    --task-description "pick red_cube from center"

# 2. Then follow steps above
```

---

## Complete Workflow

### Phase 1: Data Preparation

#### Important: Work on a Copy

**Always work on a copy of your original dataset to preserve the raw data:**

```bash
# Original collected data (DO NOT MODIFY)
/home/jrobot/project/XLeRobot/datasets

# Working copy for conversion (MODIFY THIS)
/home/jrobot/project/XLeRobot/datasets copy
```

#### Option A: Single Dataset Conversion

**Convert a single LeRobot v3 dataset to GR00T format:**

```bash
cd /home/jrobot/project/Isaac-GR00T
conda activate groot

python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path "/path/to/lerobot_dataset" \
    --robot-type so101 \
    --dual-camera \
    --task-description "pick red_cube from center"
```

**What this does:**
1. Converts `modality.json` to GR00T format
2. Fixes `stats.json` count fields (per-dimension)
3. Generates `episodes.jsonl` from episode metadata
4. Generates `tasks.jsonl` with task descriptions (uses original from tasks.parquet if available)
5. Splits parquet files into per-episode format with reset indices

#### Option B: Multi-Task Dataset (Recommended for 6-Task Training)

This is the workflow for combining multiple task datasets (grasp, pick, place, push, reach, release) into one GR00T-compatible dataset.

---

#### Key Concepts: LeRobot v3 → GR00T Conversion

Before diving into commands, here's what happens under the hood:

**1. Why Conversion is Needed**

LeRobot v3 and GR00T expect different formats:

| Aspect | LeRobot v3 | GR00T |
|--------|------------|-------|
| Episodes metadata | `meta/episodes/chunk-*/file-*.parquet` | `meta/episodes.jsonl` |
| Tasks metadata | `meta/tasks.parquet` | `meta/tasks.jsonl` |
| Modality config | Not present | `meta/modality.json` (required!) |
| Data files | `data/chunk-*/file-*.parquet` (consolidated) | `data/chunk-*/episode_XXXXXX.parquet` (per-episode) |
| Video files | `videos/.../chunk-*/file-*.mp4` (concatenated) | `videos/.../chunk-*/episode_XXXXXX.mp4` (per-episode) |
| Timestamps | Global (cumulative across episodes) | Local (each episode starts at 0) |

**2. The One-Video-Per-Episode Format (Critical!)**

GR00T expects **one video file per episode**, with timestamps starting at 0:

```
videos/observation.images.head/
└── chunk-000/
    ├── episode_000000.mp4   # Episode 0 video (timestamps 0 to ~23s)
    ├── episode_000001.mp4   # Episode 1 video (timestamps 0 to ~25s)
    └── ...
```

The conversion script uses `ffmpeg` to split the concatenated LeRobot videos into per-episode files based on `from_timestamp` / `to_timestamp` metadata.

**Why per-episode videos?**
- GR00T's `get_frames_by_timestamps()` expects video timestamps to match parquet timestamps
- With concatenated videos, episode 10 might start at timestamp 300s in the video but 0s in parquet
- Per-episode videos ensure both start at 0, preventing frame sync issues

**3. Combining Datasets**

When combining datasets, episode and chunk indices are shifted to avoid collisions:

```
Dataset A (pick):    episodes 0-9    →  Combined: episodes 0-9 (chunk-000)
Dataset B (place):   episodes 0-9    →  Combined: episodes 1000-1009 (chunk-001)
Dataset C (push):    episodes 0-9    →  Combined: episodes 2000-2009 (chunk-002)
```

Video files are copied with new episode indices: `episode_000000.mp4` → `episode_001000.mp4`

---

#### Step-by-Step Workflow

**Step 1: Copy original dataset to working directory**

```bash
# Create working copy (preserves original)
cp -r /home/jrobot/project/XLeRobot/datasets "/home/jrobot/project/XLeRobot/datasets copy"
```

**Step 2: Convert each task from LeRobot v3 to GR00T format**

```bash
cd /home/jrobot/project/Isaac-GR00T
conda activate groot

# Option A: Use the multi-task conversion script (recommended)
bash custom/scripts/convert_multitask_to_groot.sh

# Option B: Validate only (no modifications)
bash custom/scripts/convert_multitask_to_groot.sh --validate-only

# Option C: Manual conversion (if you need custom paths)
for task in grasp pick place push reach release; do
    python custom/scripts/convert_lerobot_v3_to_groot.py \
        --dataset-path "/home/jrobot/project/XLeRobot/datasets copy/left/$task" \
        --robot-type so101 \
        --dual-camera
done
```

**What the conversion does for each dataset (7 steps):**

```
BEFORE (LeRobot v3):                    AFTER (GR00T-compatible):
meta/                                   meta/
├── info.json                           ├── info.json (updated paths)
├── stats.json                          ├── stats.json (fixed counts)
├── tasks.parquet          ──────►      ├── tasks.jsonl
├── episodes/chunk-*/file-*.parquet     ├── episodes.jsonl (with task_index)
└── config.yaml                         └── modality.json (NEW!)

data/chunk-000/                         data/chunk-000/
└── file-000.parquet       ──────►      ├── episode_000000.parquet
    (all episodes)                      ├── episode_000001.parquet
                                        └── ... (per-episode, frame_index reset to 0)

videos/.../chunk-000/                   videos/.../chunk-000/
└── file-000.mp4           ──────►      ├── episode_000000.mp4
    (concatenated)                      ├── episode_000001.mp4
                                        └── ... (per-episode, timestamps start at 0)
```

**Conversion steps:**
1. [1/7] Create `modality.json` with GR00T format mapping
2. [2/7] Fix `stats.json` count fields (per-dimension arrays)
3. [3/7] Generate `episodes.jsonl` from episode metadata
4. [4/7] Generate `tasks.jsonl` with task descriptions
5. [5/7] Split parquet files into per-episode format (reset frame_index to 0)
6. [6/7] Split videos into per-episode files using ffmpeg
7. [7/7] Update `info.json` with GR00T-compatible path patterns

**Configuration** (edit `convert_multitask_to_groot.sh` to customize):
```bash
DATASETS_BASE="/home/jrobot/project/XLeRobot/datasets copy"  # Working copy!
ARM="left"
ROBOT_TYPE="so101"
TASKS=("pick" "place" "push" "reach" "grasp" "release")
```

**Step 3: Combine all tasks into one dataset**

```bash
python custom/scripts/combine_groot_datasets.py \
    --input-dir "/home/jrobot/project/XLeRobot/datasets copy/left" \
    --output "/home/jrobot/project/XLeRobot/datasets_groot"
```

Or specify datasets explicitly:
```bash
python custom/scripts/combine_groot_datasets.py \
    --datasets \
        "/home/jrobot/project/XLeRobot/datasets copy/left/pick" \
        "/home/jrobot/project/XLeRobot/datasets copy/left/place" \
        "/home/jrobot/project/XLeRobot/datasets copy/left/push" \
    --output "/home/jrobot/project/XLeRobot/datasets_groot"
```

**What the combine script does:**

```
INPUT (3 separate converted datasets):

pick/   (10 episodes: episode_000000 to episode_000009)
place/  (10 episodes: episode_000000 to episode_000009)
push/   (10 episodes: episode_000000 to episode_000009)

                    ↓ Combine with index shifting ↓

OUTPUT (1 combined dataset):

datasets_groot/
├── meta/
│   ├── tasks.jsonl      # 3 tasks: [0] pick, [1] place, [2] push
│   ├── episodes.jsonl   # 30 episodes with remapped task_index & episode_index
│   └── ...
├── data/
│   ├── chunk-000/       # pick episodes (0-9)
│   │   ├── episode_000000.parquet
│   │   └── ...
│   ├── chunk-001/       # place episodes (1000-1009)
│   │   ├── episode_001000.parquet
│   │   └── ...
│   └── chunk-002/       # push episodes (2000-2009)
│       └── ...
└── videos/
    ├── observation.images.head/
    │   ├── chunk-000/
    │   │   ├── episode_000000.mp4  # pick episode videos
    │   │   └── ...
    │   ├── chunk-001/
    │   │   ├── episode_001000.mp4  # place episode videos
    │   │   └── ...
    │   └── chunk-002/
    │       └── ...                 # push episode videos
    └── observation.images.left_wrist/
        └── ...                     # same structure
```

**Important:** The combine script requires datasets to be **already converted** (with `episode_*.mp4` files). It will error if it finds unconverted `file-*.mp4` files.

**Step 4: Verify the combined dataset (CRITICAL)**

```bash
# Comprehensive verification (checks meta, data, video synchronization)
python custom/scripts/verify_groot_dataset.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot \
    --verbose

# Expected output: ✅ ALL CHECKS PASSED
# If any check fails, fix issues before training!
```

Key verification checks:
- **Parquet Files**: Each file should contain exactly 1 episode with frame_index starting at 0
- **Data-Video Sync**: Total frames must match across info.json, episodes.jsonl, and parquet files
- **Task Distribution**: Episodes should have correct task_index values (not all 0!)

You can also verify the source dataset before combination:
```bash
python custom/scripts/verify_groot_dataset.py \
    --dataset "/home/jrobot/project/XLeRobot/datasets copy/left/pick_and_place" \
    --source --verbose
```

---

**Expected Combined Dataset Structure:**

```
datasets_groot/
├── meta/
│   ├── info.json           # Combined: total_episodes, total_frames, total_tasks
│   │                       # video_path: videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4
│   │                       # data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
│   ├── modality.json       # GR00T format mapping
│   ├── stats.json          # Combined normalization statistics
│   ├── episodes.jsonl      # All episodes with task_index
│   └── tasks.jsonl         # All task descriptions
├── data/
│   ├── chunk-000/          # First dataset's episodes (0-999)
│   │   ├── episode_000000.parquet   # Each file: single episode, frame_index starts at 0
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/          # Second dataset's episodes (1000-1999)
│   │   ├── episode_001000.parquet
│   │   └── ...
│   └── ...
└── videos/
    ├── observation.images.head/
    │   ├── chunk-000/
    │   │   ├── episode_000000.mp4   # Per-episode video, timestamps start at 0
    │   │   ├── episode_000001.mp4
    │   │   └── ...
    │   ├── chunk-001/
    │   │   ├── episode_001000.mp4
    │   │   └── ...
    │   └── ...
    └── observation.images.left_wrist/
        └── ...                       # Same structure
```

**Example 6-Task Dataset Summary:**

| Task | Episodes | Frames | Task Description |
|------|----------|--------|------------------|
| grasp | 10 | 523 | grasp the red cube |
| pick | 15 | 1870 | pick the red cube from the table |
| place | 15 | 1904 | place the red cube in the white bowl |
| push | 10 | 1791 | push the red cube to the green cube |
| reach | 10 | 756 | reach the red cube |
| release | 8 | 215 | release |
| **Total** | **68** | **7059** | **6 tasks** |

---

### Phase 2: Pre-Training Verification

**Always run verification before training:**

```bash
python custom/scripts/verify_groot_training_setup.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot
```

**What this checks:**
- Dataset structure (meta/, videos/, data/ directories)
- `modality.json` format and camera mapping
- `tasks.jsonl` task descriptions (should match inference)
- `stats.json` normalization statistics (per-dimension arrays)
- `episodes.jsonl` episode metadata
- Video file readability (optional, use `--skip-videos` to skip)
- Model and PEFT imports

**Expected output:**
```
======================================================================
 GR00T Pre-Training Verification
======================================================================

======================================================================
 Dataset Structure Verification
======================================================================
  [PASS] meta/ directory exists
  [PASS] modality.json exists
  [PASS] tasks.jsonl exists
  ...

======================================================================
 VERIFICATION SUMMARY
======================================================================

  ==================================================
  ALL CHECKS PASSED - Ready for training!
  ==================================================
```

**If verification fails:** Fix the issues before training. Common fixes:
- Re-run conversion script
- Check task description matches inference
- Verify video files are not corrupted

---

### Phase 3: Training

#### Quick Validation (Optional but Recommended)

Run a quick 100-step validation to ensure everything works:

```bash
cd /home/jrobot/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh  # 100 steps, ~5 minutes
```

#### MVP Training (5000 Steps) - Recommended Start

```bash
cd /home/jrobot/project/Isaac-GR00T
bash custom/scripts/train_groot_mvp.sh  # 5000 steps, ~1 hour
```

**Configuration:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 5000 | ~50-60 minutes |
| Batch Size | 4 | Conservative for memory |
| Learning Rate | 1e-4 | 4x higher than default (critical!) |
| LoRA Rank | 16 | Standard for VLMs |
| Save Steps | 500 | 10 checkpoints total |
| VRAM | ~18-20GB | With --no-tune_diffusion_model |
| Video Backend | torchvision_av | Default; try `decord` for faster training |

#### Training Speed Optimization

If training is slow (>1.5s/step), the main factors are:

**1. GPU Power Limit (Primary factor)**

Laptop GPUs are often power-limited. Check with `nvidia-smi`:
- RTX 5090 Laptop at 95W → ~1.67s/step
- RTX 5090 Laptop at 175W → ~1.0s/step (if cooling allows)

**2. Video Backend**

The script now defaults to `decord` which is optimized for ML training. You can change in `train_groot_mvp.sh`:
```bash
VIDEO_BACKEND=decord       # Default, faster for training
VIDEO_BACKEND=torchvision_av  # Alternative
```

**Note on video codec:** AV1 is the correct format. Per [LeRobot's benchmark](https://huggingface.co/blog/video-encoding), AV1 decodes faster than H.264 for multi-frame loading. Do NOT re-encode to H.264.

**Typical training speeds by hardware:**
| GPU | Power | Expected Speed |
|-----|-------|----------------|
| RTX 4090 Desktop | 450W | ~0.8-1.0s/step |
| RTX 5090 Laptop | 95-175W | ~1.2-1.7s/step |
| RTX 3090 | 350W | ~1.5-2.0s/step |

**What the MVP script does:**
1. **Step 1-2:** Environment check and pre-training verification
2. **Step 3-4:** Modality.json creation and pre-training checks
3. **Step 5:** Run 5K LoRA training with TensorBoard logging
4. **Step 6:** Evaluate all checkpoints (MAE, accuracy)
5. **Step 7:** Select best checkpoint (lowest MAE), create `best/` symlink
6. **Step 8:** Run inference diagnosis on best checkpoint
7. **Step 9:** Generate comprehensive report

**Output structure:**
```
outputs/groot_mvp_lora_TIMESTAMP/
├── best -> checkpoint-XXXX       # Symlink to best checkpoint
├── best_info.txt                 # MAE comparison for all checkpoints
├── checkpoint-500/
├── checkpoint-1000/
├── ...
├── evaluation_results.json
├── diagnosis_results.json
└── training_report.txt
```

**Monitor during training:**
```bash
# Terminal 1: Watch training progress
# (script shows progress bar)

# Terminal 2: Monitor GPU
watch -n 5 nvidia-smi

# Terminal 3: TensorBoard (optional)
tensorboard --logdir /home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_TIMESTAMP/runs
```

#### Full Training (10000 Steps)

For production models, modify the training script:

```bash
# Edit custom/scripts/train_groot_mvp.sh
# Change: MAX_STEPS=5000
# To:     MAX_STEPS=10000

# Then run
bash custom/scripts/train_groot_mvp.sh
```

---

### Phase 4: Post-Training Evaluation

#### Automatic Evaluation (Part of Training Script)

The training script automatically runs evaluation after training completes. Results are saved to:
- `$OUTPUT_DIR/evaluation_results.json`
- `$OUTPUT_DIR/evaluation.log`

#### Manual Evaluation

**Evaluate a specific checkpoint:**
```bash
python custom/scripts/evaluate_groot_checkpoint.py \
    --checkpoint /path/to/checkpoint \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot \
    --num-samples 300
```

**Evaluate all checkpoints in a training run:**
```bash
python custom/scripts/evaluate_groot_checkpoint.py \
    --training-dir /home/jrobot/project/XLeRobot/outputs/groot_5k_lora_TIMESTAMP \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot
```

**Output metrics:**
```
======================================================================
 COMPARISON SUMMARY
======================================================================

  Checkpoint           MAE     Acc@5°   Acc@10°   Acc@15°
  --------------------------------------------------------
  checkpoint-500      15.23°    25.3%     45.2%     62.1%
  checkpoint-1000     12.45°    32.1%     54.3%     71.2%
  checkpoint-2000      9.87°    41.2%     65.4%     80.3%
  ...
  checkpoint-5000      6.23°    58.4%     82.1%     91.5%

  Best checkpoint: checkpoint-5000
  Best MAE: 6.23°
```

**Success criteria:**
| Metric | Target |
|--------|--------|
| Overall MAE | < 10° |
| Per-Joint MAE | < 15° |
| Acc@5° | > 50% |
| Acc@10° | > 80% |

#### Inference Diagnosis

**Run diagnosis on a checkpoint:**
```bash
python custom/scripts/diagnose_groot_inference.py \
    --checkpoint /path/to/checkpoint \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot \
    --num-samples 5
```

**What this tests:**
1. **State sensitivity:** Action should change when state changes
2. **Image sensitivity:** Action should change when image changes
3. **Output consistency:** Same inputs should give same outputs
4. **Task conditioning:** Different tasks should produce different actions
5. **Action range:** Model should produce varied actions across samples

**Expected output:**
```
======================================================================
 DIAGNOSIS SUMMARY
======================================================================
  ✓ State sensitivity
  ✓ Image sensitivity
  ✓ Output consistency
  ✓ Task conditioning
  ✓ Action range OK

  5/5 tests passed

  All tests passed! Model behavior looks healthy.
```

---

### Phase 5: Inference & Deployment

#### Run Inference on Robot

```bash
python custom/scripts/infer_groot_so101.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_5k_lora_TIMESTAMP \
    --task "pick red_cube from center" \
    --actions-to-execute 100 \
    --action-horizon 12
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | Required | Path to checkpoint |
| `--task` | "pick red_cube from center" | Task description (must match training!) |
| `--data-config` | "so100_dualcam" | Data configuration |
| `--embodiment-tag` | "new_embodiment" | Embodiment tag |
| `--head-cam-idx` | 4 | Head camera device index |
| `--wrist-cam-idx` | 6 | Wrist camera device index |
| `--port` | "/dev/ttyACM2" | Robot serial port |
| `--actions-to-execute` | 100 | Number of action chunks |
| `--action-horizon` | 12 | Actions per inference |

**LoRA checkpoint auto-detection:**
The inference script automatically detects LoRA checkpoints and loads them correctly:
```
============================================================
[INFO] Detected LoRA checkpoint - using PEFT adapter loading
============================================================

[LoRA] Detected LoRA checkpoint
[LoRA] Base model: /home/jrobot/.cache/huggingface/hub/...
[LoRA] LoRA rank: 16
[LoRA] LoRA alpha: 16
[LoRA] Target modules: ['to_q', 'to_k', 'to_v']
[LoRA] Step 1/3: Loading base GR00T model...
[LoRA] Step 2/3: Loading PEFT adapter...
[LoRA] Step 3/3: Merging LoRA weights into base model...
[LoRA] GR00T model with LoRA loaded successfully!
```

---

## Script Reference

### Location
All scripts are in `/home/jrobot/project/Isaac-GR00T/custom/scripts/`

### Data Preparation Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `convert_lerobot_v3_to_groot.py` | Convert single LeRobot v3 dataset to GR00T format (includes video splitting) | `python convert_lerobot_v3_to_groot.py --dataset-path /path --robot-type so101 --dual-camera` |
| `convert_multitask_to_groot.sh` | Convert all 6 task datasets in batch | `bash convert_multitask_to_groot.sh` or `bash convert_multitask_to_groot.sh --validate-only` |
| `combine_groot_datasets.py` | Combine multiple converted datasets (requires per-episode videos) | `python combine_groot_datasets.py --input-dir /path --output /path/combined` |
| `verify_groot_dataset.py` | Verify dataset integrity and video/data sync | `python verify_groot_dataset.py --dataset /path --verbose` |

### Training & Verification Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_groot_mini_mvp.sh` | Quick validation (100 steps) | `bash train_groot_mini_mvp.sh` |
| `train_groot_mvp.sh` | Full training (5K steps) with best checkpoint | `bash train_groot_mvp.sh` |
| `verify_groot_training_setup.py` | Pre-training verification | `python verify_groot_training_setup.py --dataset /path` |

### Evaluation & Diagnosis Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `evaluate_groot_checkpoint.py` | Compute MAE and accuracy | `python evaluate_groot_checkpoint.py --training-dir /path` |
| `diagnose_groot_inference.py` | Test model behavior | `python diagnose_groot_inference.py --checkpoint /path --dataset /path` |

### Inference Script

| Script | Purpose | Usage |
|--------|---------|-------|
| `infer_groot_so101.py` | Run inference on robot | `python infer_groot_so101.py --model-path /path --task "..."` |

---

## Industry Best Practices (NVIDIA/HuggingFace)

Based on research from official NVIDIA getting_started guides and HuggingFace blog posts:

### Data Collection FPS - Clarification

**Important: Camera FPS vs Action FPS are different concepts!**

| Aspect | Recommendation | Your Setup | Status |
|--------|----------------|------------|--------|
| **Camera FPS** | 30 FPS (video capture) | 30 FPS | ✓ Correct |
| **Action FPS** | 5 FPS (model training) | 5 FPS | ✓ Correct |

**Why 5 FPS Action Frequency Works for Both GR00T and Pi0.5:**
| Model | Action Steps | At 5 FPS | Temporal Context |
|-------|-------------|----------|------------------|
| **Pi0.5** | 50 steps | 50 × 0.2s | **10 seconds** |
| **GR00T** | 16 steps | 16 × 0.2s | **3.2 seconds** |

**Bottom Line:** Your 5 FPS action data works for both GR00T and Pi0.5 finetuning.

### Diffusion Model Tuning

**`--no-tune_diffusion_model` is the CORRECT setting:**

| Component | With Flag | Without Flag |
|-----------|-----------|--------------|
| Visual Encoder | Frozen | Frozen |
| Language Model | Frozen | Frozen |
| **Projector** | **TRAINED** | **TRAINED** |
| **Diffusion Model (DiT)** | **FROZEN** | TRAINED |

**Why this is correct:**
- The DiT is a shared action head across all embodiments (pre-trained)
- Freezing it is NVIDIA's default and recommended setting
- VRAM: 18-20GB (vs 58-70GB for full finetuning)
- Required for RTX 4090/5090 (24GB VRAM)

**Note:** `--no-tune_diffusion_model` does NOT disable LoRA. LoRA is controlled by `--lora-rank`.

### Resume Training (5K → 10K)

GR00T supports resuming training via `--resume` flag:

```bash
# Phase 1: Initial 5K training (use train_groot_mvp.sh)
bash custom/scripts/train_groot_mvp.sh

# Phase 2: Validate checkpoint
python custom/scripts/evaluate_groot_checkpoint.py --training-dir OUTPUT_DIR

# Phase 3: Resume from 5K to 10K
python scripts/gr00t_finetune.py \
    --dataset-path /home/jrobot/project/XLeRobot/datasets_groot \
    --output-dir OUTPUT_DIR \
    --max-steps 10000 \
    --save-steps 500 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --no-tune_diffusion_model \
    --dataloader_num_workers 16 \
    --resume
```

---

## Training Configurations

### MVP (5000 Steps) - Recommended

```bash
MAX_STEPS=5000
SAVE_STEPS=500
BATCH_SIZE=4
LEARNING_RATE=1e-4
LORA_RANK=16
NUM_WORKERS=16  # Per HuggingFace recommendation
```

**Expected results:**
- Duration: ~50-60 minutes
- VRAM: 18-20GB
- Final loss: < 0.3
- MAE: < 10°

### Full Training (10000 Steps)

```bash
MAX_STEPS=10000
SAVE_STEPS=1000
BATCH_SIZE=4
LEARNING_RATE=1e-4
LORA_RANK=16
```

**Expected results:**
- Duration: ~2 hours
- VRAM: 18-20GB
- Final loss: < 0.2
- MAE: < 8°

### Quick Test (100 Steps)

For pipeline validation only:
```bash
MAX_STEPS=100
SAVE_STEPS=100
BATCH_SIZE=4
LEARNING_RATE=1e-4
LORA_RANK=16
```

---

## Critical Fixes Applied

### Issue 1: LoRA Adapter Loading ✅ FIXED

**Problem:** GR00T saves PEFT adapters separately (`adapter_config.json` + `adapter_model.safetensors`), but `Gr00tPolicy` doesn't load them.

**Checkpoint structure:**
```
checkpoint/
├── adapter_config.json          # PEFT config
├── adapter_model.safetensors    # LoRA weights (13MB)
└── experiment_cfg/
    └── metadata.json            # Normalization stats
```

**Solution:** Added to `infer_groot_so101.py`:
```python
def load_groot_with_lora(model_path, ...):
    from peft import PeftModel
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    # 1. Load base model
    base_model = GR00T_N1_5.from_pretrained(base_model_path)

    # 2. Load PEFT adapter
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # 3. Merge LoRA into base weights
    merged_model = peft_model.merge_and_unload()

    return policy_with_merged_model
```

### Issue 2: Task Description Mismatch ✅ FIXED

**Problem:** Inference task description didn't match training data.

**Solution:** Updated defaults to match training:
- Training: `tasks.jsonl` contains `"pick red_cube from center"`
- Inference: Default task is now `"pick red_cube from center"`

### Issue 3: Missing Evaluation Metrics ✅ FIXED

**Problem:** No way to evaluate checkpoint quality or compare checkpoints.

**Solution:** Created `evaluate_groot_checkpoint.py` with:
- Overall and per-joint MAE
- Accuracy at thresholds (5°, 10°, 15°)
- Checkpoint comparison

### Issue 4: No Inference Diagnosis ✅ FIXED

**Problem:** No way to debug model behavior issues.

**Solution:** Created `diagnose_groot_inference.py` with:
- State/image sensitivity tests
- Task conditioning tests
- Output consistency tests

---

## Troubleshooting

### LoRA Checkpoint Not Loading

**Symptoms:**
```
Model produces random actions
Actions don't change with input
```

**Cause:** Standard `Gr00tPolicy` doesn't load PEFT adapters.

**Solution:** Use `infer_groot_so101.py` which auto-detects and loads LoRA checkpoints correctly.

### High MAE on Specific Joints

**Symptoms:**
```
shoulder_lift: 40° MAE
elbow_flex: 35° MAE
Other joints: < 10° MAE
```

**Causes:**
1. Not enough training data for those poses
2. Training data doesn't cover full range of motion

**Solutions:**
1. Collect more episodes with varied poses
2. Ensure starting poses are within training distribution
3. Train for more steps

### Task Description Mismatch

**Symptoms:**
```
Model doesn't respond to task
Model produces wrong actions for task
```

**Cause:** Inference task doesn't match training task.

**Solution:**
1. Check `datasets_groot/meta/tasks.jsonl` for exact task description
2. Use same task in inference: `--task "exact task from training"`

### Model Not Sensitive to Inputs

**Symptoms:**
```
State sensitivity test: FAIL
Mean action diff: 0.001° (should be > 1°)
```

**Causes:**
1. LoRA weights not loaded
2. Learning rate too low during training
3. Not enough training steps

**Solutions:**
1. Verify checkpoint is LoRA type and loaded correctly
2. Use learning rate 1e-4 (not 2.5e-5)
3. Train for more steps

### CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `BATCH_SIZE=2`
2. Clear GPU: `python -c "import torch; torch.cuda.empty_cache()"`
3. Kill other GPU processes: `nvidia-smi` then `kill <PID>`

---

## Progress Tracking

### Completed ✅

- [x] Environment setup (groot conda env)
- [x] RTX 5090 Blackwell compatibility (PyTorch 2.10.0, flash-attn sm_120)
- [x] Mini-MVP validation (100 steps, pipeline verified)
- [x] Dataset conversion script (`convert_lerobot_v3_to_groot.py`)
- [x] **LoRA adapter loading fix** (`infer_groot_so101.py`)
- [x] **Pre-training verification script** (`verify_groot_training_setup.py`)
- [x] **Post-training evaluation script** (`evaluate_groot_checkpoint.py`)
- [x] **Inference diagnosis script** (`diagnose_groot_inference.py`)
- [x] **Multi-dataset combination script** (`combine_groot_datasets.py`)
- [x] **Integrated training pipeline** (5K steps with evaluation)
- [x] Task description fix (matches training data)
- [x] Documentation (this guide)

### Current Target 🎯

- [ ] Run 5K MVP training
- [ ] Evaluate checkpoints (target MAE < 10°)
- [ ] Run inference diagnosis (target 5/5 tests pass)
- [ ] Test on robot

### Pending ⏳

- [ ] Collect more episodes for multi-task training
- [ ] Run 10K full training
- [ ] Deploy production model

---

## Quick Command Reference

```bash
# Activate environment
conda activate groot
cd /home/jrobot/project/Isaac-GR00T

# Convert dataset
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --dataset-path /path/to/dataset --robot-type so101 --dual-camera

# Verify before training
python custom/scripts/verify_groot_training_setup.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot

# Run quick validation (optional)
bash custom/scripts/train_groot_mini_mvp.sh  # 100 steps, ~5 min

# Run full 5K training
bash custom/scripts/train_groot_mvp.sh  # 5000 steps, ~1 hour

# Evaluate checkpoint (done automatically by train_groot_mvp.sh)
python custom/scripts/evaluate_groot_checkpoint.py \
    --checkpoint /path/to/output/best \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot

# Diagnose inference (done automatically by train_groot_mvp.sh)
python custom/scripts/diagnose_groot_inference.py \
    --checkpoint /path/to/output/best \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot

# Run inference with best checkpoint
python custom/scripts/infer_groot_so101.py \
    --model-path /path/to/output/best \
    --task "pick red_cube from center"

# Monitor GPU
watch -n 5 nvidia-smi
```

---

**Current Status: LoRA Loading Fixed, Monitoring Scripts Created - Ready for 5K Training!**

**Next Action:** Run `bash custom/scripts/train_groot_mvp.sh` to start 5K training with automatic evaluation and best checkpoint selection.
