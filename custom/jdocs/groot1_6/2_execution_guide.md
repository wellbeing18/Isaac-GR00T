# GR00T 1.6 SO-101 Execution Guide

This guide walks through running the complete GR00T 1.6 pipeline for SO-101 robot finetuning.

## Prerequisites

- Conda environment `groot` activated
- LeRobot v3 dataset at `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`
- NVIDIA GPU with 24GB VRAM (RTX 4090/5090)

```bash
conda activate groot
cd /home/jrobot/project/Isaac-GR00T
```

---

## Step 1: Download GR00T 1.6 Model (Already Done)

The model has been downloaded to HuggingFace cache. Verify:

```bash
ls -la ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/
```

If needed, re-download:
```bash
python custom/scripts/ver1_6/download_groot_1_6.py
```

**Expected output:** Model files (~6.57 GB) in HuggingFace cache.

---

## Step 2: Convert Dataset

Convert your LeRobot v3 dataset to GR00T 1.6 format.

```bash
python custom/scripts/ver1_6/convert_lerobot_v3_to_groot_1_6.py
```

**Key variables to modify** (at top of script):
```python
INPUT_DATASET = "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place"
OUTPUT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"
```

**Expected output:**
- Dataset created at `datasets/so101_pick_place_groot/`
- Contains: `meta/`, `data/`, `videos/` directories
- Statistics computed: `meta/stats.json`, `meta/relative_stats.json`

**Time:** ~5-10 minutes depending on dataset size.

---

## Step 3: Verify Dataset

Validate the converted dataset structure and integrity.

```bash
python custom/scripts/ver1_6/verify_groot_dataset.py --dataset datasets/so101_pick_place_groot
```

Optional: Run with full loader test:
```bash
python custom/scripts/ver1_6/verify_groot_dataset.py --dataset datasets/so101_pick_place_groot --load-test
```

**Expected output:**
```
======================================================================
VERIFICATION SUMMARY
======================================================================

  RESULT: ALL CHECKS PASSED
  Dataset is ready for GR00T 1.6 training!
======================================================================
```

**Troubleshooting:**
- If `modality.json` errors: Check `custom/cfgs/so101_modality.json`
- If dimension errors: Verify ARM_JOINTS=5, GRIPPER_DIMS=1 in conversion script
- If video errors: Check video files exist in `videos/` directory

---

## Step 3.5: Generate Relative Stats (CRITICAL)

**This step is required for GR00T 1.6 to work with relative actions.**

If verification shows "relative_stats.json not found", generate it:

```bash
python -c "
import sys
sys.path.insert(0, '.')

# Import modality config to register NEW_EMBODIMENT
import importlib.util
spec = importlib.util.spec_from_file_location('so101_config', 'custom/scripts/ver1_6/so101_config_1_6.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from gr00t.data.stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag

print('Generating relative_stats.json...')
generate_rel_stats('datasets/so101_pick_place_groot', EmbodimentTag.NEW_EMBODIMENT)
print('Done!')
"
```

**Expected output:**
```
[so101_config_1_6] Registered SO-101 config for NEW_EMBODIMENT
Generating relative_stats.json...
Generating relative stats for datasets/so101_pick_place_groot EmbodimentTag.NEW_EMBODIMENT single_arm
Loading trajectories for key single_arm: 100%|██████████| 70/70 [00:02<00:00, 29.15it/s]
Done!
```

**Verify creation:**
```bash
ls -la datasets/so101_pick_place_groot/meta/relative_stats.json
```

**Why this is needed:**
- GR00T 1.6 uses `use_relative_action=True` by default
- Relative stats normalize delta actions (arm movements relative to current position)
- Without this file, action decoding will fail during training/inference

---

## Step 4: Verify Zero-Shot Inference

Test the base model loads correctly and produces valid outputs.

```bash
python custom/scripts/ver1_6/verify_zeroshot_1_6.py
```

**Expected output:**
```
======================================================================
VERIFICATION SUMMARY
======================================================================
  RESULT: All checks passed!
  Zero-shot inference is working correctly.

  The model may not produce meaningful actions without finetuning,
  but the pipeline is verified to be working.
======================================================================
```

**What this verifies:**
- Modality config registration (NEW_EMBODIMENT tag)
- Model loading (~7-8 GB VRAM)
- Dataset loading with modality config
- Inference produces correct output shapes
- No NaN/Inf in outputs
- Inference timing (~40-50ms on RTX 5090)

**Troubleshooting:**
- CUDA OOM: Close other GPU applications
- Modality config error: Check `so101_config_1_6.py` is correct
- Dataset loading error: Run Step 3 again

---

## Step 5: MVP Training (1000 Steps)

### 5a. Run MVP Training

Train for 1000 steps (~1 hour) with checkpoints at 500 and 1000 steps:

```bash
MAX_STEPS=1000 SAVE_STEPS=500 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh
```

**Monitor training:**
- Watch terminal for loss values (should decrease)
- No wandb/tensorboard - metrics verified via open-loop eval

**Expected behavior:**
- Loss should decrease from ~1.0 to ~0.1-0.3
- VRAM usage: ~20-22 GB
- Speed: ~2-3 steps/second on RTX 5090

### 5b. Evaluate Checkpoints

Compare checkpoint-500 vs checkpoint-1000 to verify training is improving:

```bash
# Evaluate early checkpoint
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-500 \
    --output-dir eval_outputs/checkpoint_500 \
    --num-trajectories 5

# Evaluate MVP checkpoint
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-1000 \
    --output-dir eval_outputs/checkpoint_1000 \
    --num-trajectories 5
```

### 5c. Compare Checkpoints

| Metric | Checkpoint-500 | Checkpoint-1000 | Decision |
|--------|----------------|-----------------|----------|
| MSE | Higher | Lower | If MSE decreased → training working |
| MAE | Higher | Lower | If MAE improved → proceed to full training |

**Decision criteria:**
- **Checkpoint-1000 MSE < Checkpoint-500 MSE**: ✅ Training effective, proceed to full training
- **Checkpoint-1000 MSE ≈ Checkpoint-500 MSE**: ⚠️ Might be plateauing, check loss curve
- **Checkpoint-1000 MSE > Checkpoint-500 MSE**: ❌ Training issue, debug before continuing

**Note:** Baseline (zero-shot) evaluation is not possible because the base model doesn't have normalization
parameters for NEW_EMBODIMENT. Compare between training checkpoints instead.

---

## Step 6: Full Training

After MVP shows improvement over baseline:

```bash
bash custom/scripts/ver1_6/train_groot_so101_1_6.sh
```

**Key hyperparameters** (modify via environment):
```bash
# Lower batch size if OOM:
GLOBAL_BATCH_SIZE=8 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh
```

**Training time estimates:**
| Steps | Approx. Time |
|-------|--------------|
| 1000  | ~1 hour      |
| 5000  | ~3 hours     |
| 10000 | ~6 hours     |
| 45000 | ~15 hours    |

**Checkpoints saved to:** `outputs/groot_1_6_so101/checkpoint-{step}/`

### Running Training in Background (Survives Session Close)

For long training runs, use `nohup` to keep training running after you close the terminal:

```bash
cd /home/jrobot/project/Isaac-GR00T
source ~/anaconda3/etc/profile.d/conda.sh && conda activate groot

# Run 45k training in background
nohup bash -c 'MAX_STEPS=45000 SAVE_STEPS=1500 SAVE_TOTAL_LIMIT=15 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh' > outputs/training_45k.log 2>&1 &

# Save the process ID
echo $!
```

**Monitoring:**
```bash
# Watch training progress
tail -f outputs/training_45k.log

# Or watch the formatted log inside output folder
tail -f outputs/groot_1_6_augmented_*/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if training is still running
ps aux | grep train_groot

# List saved checkpoints
ls -la outputs/groot_1_6_augmented_*/checkpoint-*
```

**Logs location:**
| Log | Location | Content |
|-----|----------|---------|
| nohup log | `outputs/training_45k.log` | All stdout/stderr |
| training log | `outputs/groot_1_6_augmented_*/training.log` | Formatted with timestamps |

**Troubleshooting:**
- CUDA OOM: Reduce `GLOBAL_BATCH_SIZE` to 8 or 4
- Slow training: Check GPU utilization with `nvidia-smi`

---

## Step 6.5: Resume Training (Extended Training)

### When to Resume vs. Start Fresh

**Start Fresh (do NOT resume) when:**
- Data distribution changed (e.g., added data augmentation)
- Regularization parameters changed (e.g., added dropout)
- Architecture changes
- The model seems broken/unstable

**Resume IS safe when:**
- Simply training for more steps with SAME settings
- Continuing training after interruption

### Resume Modes

#### MODE 1: Planned Training (RECOMMENDED)
Set `MAX_STEPS` to your target upfront. Stop at any checkpoint, evaluate, resume with SAME max_steps.

```bash
# Start training to 45k
MAX_STEPS=45000 SAVE_STEPS=3000 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh

# Stop after checkpoint-15000 (Ctrl+C), evaluate it
# Resume with SAME max_steps - LR schedule continues correctly
RESUME_FROM=outputs/.../checkpoint-15000 MAX_STEPS=45000 \
bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
```

**Why this works:** HuggingFace Trainer saves scheduler state. When you resume with same `max_steps`, the LR continues from where it left off.

#### MODE 2: Extend Training (Beyond original max_steps)
If you need more steps than originally planned, use constant LR.

```bash
# Original was 15k, now want 45k - MUST use constant LR
RESUME_FROM=outputs/.../checkpoint-15000 MAX_STEPS=45000 \
LR_SCHEDULER_TYPE=constant LEARNING_RATE=1e-5 \
bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
```

**Why constant LR:** Cosine scheduler recalculates when max_steps changes, causing LR jumps.

#### MODE 3: Unknown Duration
Use constant LR from the start if you don't know how long to train.

```bash
LR_SCHEDULER_TYPE=constant MAX_STEPS=50000 bash custom/scripts/ver1_6/train_groot_so101_augmented.sh

# Later, extend to 100k - no scheduler issues
RESUME_FROM=outputs/.../checkpoint-50000 MAX_STEPS=100000 \
bash custom/scripts/ver1_6/train_groot_so101_augmented.sh
```

### The LR Jump Problem (Why Mode 2 needs constant LR)

**Problem:** When resuming with different `max_steps`, the cosine LR scheduler recalculates:

```
# Training 15k steps with cosine decay:
Step 15000: LR = ~1e-12 (near zero, converged)

# Resume with max_steps=45000 (cosine):
Step 15001: LR = cosine(15001/45000) = ~7.5e-05  # JUMPED UP!
# This destabilizes the converged model!
```

**Evidence from experiments:**
- 15k checkpoint: MAE 2.28°, MSE 10.6 (good)
- 27k checkpoint (after bad resume): MAE 2.61°, MSE 15.3 (WORSE!)
- Root cause: LR jumped from 1e-12 to 7.5e-05

### Technical Details

**How proper resume works:**
1. Script always loads ORIGINAL base model (`nvidia/GR00T-N1.6-3B`)
2. Checkpoint path passed via `--resume_from_checkpoint`
3. HuggingFace Trainer loads weights, optimizer, scheduler from checkpoint
4. Training continues from saved step

**Modified files for resume support:**
- `gr00t/configs/finetune_config.py` - Added `resume_from_checkpoint`, `lr_scheduler_type`
- `gr00t/configs/training/training_config.py` - Added `resume_from_checkpoint`
- `gr00t/experiment/launch_finetune.py` - Passes parameters to config
- `gr00t/experiment/experiment.py` - Uses resume_from_checkpoint in trainer
- `custom/scripts/ver1_6/train_groot_so101_augmented.sh` - Full resume support

**Available LR scheduler types:**
- `cosine` - Default for fresh training (decays to ~0 at max_steps)
- `constant` - Recommended for extended training
- `linear` - Linear decay
- `constant_with_warmup` - Constant after warmup period

**Script auto-detection:**
The training script reads `trainer_state.json` from checkpoint to detect if you're extending beyond original `max_steps`. It will warn and prompt for confirmation if using cosine scheduler in that case.

---

## Step 7: Open-Loop Evaluation (After Full Training)

Evaluate the finetuned model on dataset trajectories (no robot needed).

```bash
# Evaluate latest checkpoint
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000

# Evaluate specific trajectories
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000 \
    --traj-ids 0 1 2 3 4

# Evaluate all trajectories
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000 \
    --num-trajectories -1
```

**Expected output:**
```
======================================================================
EVALUATION SUMMARY
======================================================================
  Trajectories evaluated: 5
  Overall MSE: 0.003456 ± 0.001234
  Overall MAE: 0.045678 ± 0.012345

  Per-Joint MSE:
    shoulder_pan: 0.002345
    shoulder_lift: 0.004567
    elbow_flex: 0.003456
    wrist_flex: 0.002890
    wrist_roll: 0.001234
    gripper: 0.006789

  RESULT: EXCELLENT - Model should work well on robot
======================================================================
```

**Interpretation:**
| MSE Range | Interpretation |
|-----------|----------------|
| < 0.01    | Excellent - should work well |
| 0.01-0.05 | Good - may need tuning |
| > 0.05    | Poor - investigate issues |

**Output files:**
- `eval_outputs/openloop_1_6/evaluation_results.json` - Metrics
- `eval_outputs/openloop_1_6/traj_*.png` - Trajectory plots

---

## Step 8: Robot Inference

Run inference on the real SO-101 robot.

### 8a. Dry Run (No Robot Actions)

Test the pipeline without moving the robot:

```bash
python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000 \
    --dry-run \
    --task "pick up the blocks and place them on the plate"
```

### 8b. Real Robot Inference

**Before running:**
1. Power on robot
2. Connect cameras (check `v4l2-ctl --list-devices`)
3. Connect robot arm (check `ls /dev/ttyACM*`)
4. Clear workspace of obstacles
5. Move robot to safe starting position

```bash
python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000 \
    --task "pick up the blocks and place them on the plate" \
    --duration 30
```

**Options:**
```bash
# Record images during inference
--record

# Custom task description
--task "place the cube in the bowl"

# Longer duration
--duration 60

# Different checkpoint
--checkpoint outputs/groot_1_6_so101/checkpoint-5000
```

**Stop inference:** Press `Ctrl+C` for clean shutdown.

**Troubleshooting:**
- Camera not found: Check device indices in `custom/cfgs/so101_hardware.yaml`
- Robot not responding: Check serial port in hardware config
- Actions too aggressive: May need to adjust action scaling in the script

---

## Quick Reference

### All Commands in Order

```bash
# Activate environment
conda activate groot
cd /home/jrobot/project/Isaac-GR00T

# 1. Convert dataset (if not already done)
python custom/scripts/ver1_6/convert_lerobot_v3_to_groot_1_6.py

# 2. Verify dataset
python custom/scripts/ver1_6/verify_groot_dataset.py --dataset datasets/so101_pick_place_groot --load-test

# 3. Generate relative stats (CRITICAL - do this after conversion)
python -c "
import sys; sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('c', 'custom/scripts/ver1_6/so101_config_1_6.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from gr00t.data.stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('datasets/so101_pick_place_groot', EmbodimentTag.NEW_EMBODIMENT)
"

# 4. Verify zero-shot (optional but recommended)
python custom/scripts/ver1_6/verify_zeroshot_1_6.py --dataset datasets/so101_pick_place_groot

# 5. MVP training (1000 steps, saves at 500 and 1000)
MAX_STEPS=1000 SAVE_STEPS=500 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh

# 6. Evaluate checkpoints - compare 500 vs 1000 to verify training works
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-500 \
    --output-dir eval_outputs/checkpoint_500 \
    --num-trajectories 5

python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-1000 \
    --output-dir eval_outputs/checkpoint_1000 \
    --num-trajectories 5

# 7. Full training (after MVP shows checkpoint-1000 better than checkpoint-500)
bash custom/scripts/ver1_6/train_groot_so101_1_6.sh

# 8. Final evaluation
python custom/scripts/ver1_6/eval_openloop_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000

# 9. Robot inference
python custom/scripts/ver1_6/infer_groot_so101_1_6.py \
    --checkpoint outputs/groot_1_6_so101/checkpoint-10000 \
    --task "pick the red cube"
```

### File Locations

| Type | Location |
|------|----------|
| Scripts | `custom/scripts/ver1_6/` |
| Modality config | `custom/cfgs/so101_modality.json` |
| Hardware config | `custom/cfgs/so101_hardware.yaml` |
| Converted dataset | `datasets/so101_pick_place_groot/` |
| Training outputs | `outputs/groot_1_6_so101/` |
| Evaluation outputs | `eval_outputs/openloop_1_6/` |
| Documentation | `custom/jdocs/groot1_6/` |

### Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| MAX_STEPS | 10000 | Training iterations |
| LEARNING_RATE | 1e-4 | NVIDIA recommended |
| GLOBAL_BATCH_SIZE | 16 | Reduce if OOM |
| ACTION_HORIZON | 16 | Actions per inference |
| NUM_DENOISING_STEPS | 4 | DiT iterations |

---

## Next Steps After Successful Pipeline

1. **Collect more data** - More diverse demonstrations improve generalization
2. **Tune hyperparameters** - Adjust learning rate, batch size based on results
3. **Try different tasks** - Modify task descriptions for new behaviors
4. **Deploy to Jetson** - Export model for edge deployment (see NVIDIA docs)

---

## Common Issues & Solutions

### Environment Issues

**Problem:** `ModuleNotFoundError: No module named 'av'` (or lmdb, cv2, torchvision)
```bash
# Solution: Make sure you're in groot environment
conda activate groot

# If packages still missing:
pip install av lmdb opencv-python
```

**Problem:** Running scripts from base conda environment
```bash
# Always verify environment first
which python  # Should show /home/jrobot/anaconda3/envs/groot/bin/python
conda env list  # Should show * next to groot
```

### Dataset Issues

**Problem:** `relative_stats.json not found`
```bash
# Solution: Generate relative stats (see Step 3.5 above)
# This is required for GR00T 1.6's relative action mode
```

**Problem:** `KeyError: 'new_embodiment'` when loading model
```bash
# Solution: Modality config must be imported BEFORE processor loads
# Make sure so101_config_1_6.py is imported first
```

**Problem:** Dimension mismatch errors
```bash
# Verify your modality.json matches so101_config_1_6.py:
# - state.single_arm: [0:5]
# - state.gripper: [5:6]
# - action.single_arm: [0:5]
# - action.gripper: [5:6]
```

### Training Issues

**Problem:** CUDA out of memory
```bash
# Solution: Reduce batch size
GLOBAL_BATCH_SIZE=8 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh

# Or even smaller:
GLOBAL_BATCH_SIZE=4 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh
```

**Problem:** Slow training speed
```bash
# Check GPU utilization
nvidia-smi

# If GPU not fully utilized, increase batch size
# If memory bound, reduce dataloader workers
DATALOADER_WORKERS=2 bash custom/scripts/ver1_6/train_groot_so101_1_6.sh
```

### Inference Issues

**Problem:** Model outputs NaN values
```bash
# Usually caused by:
# 1. Missing statistics files (stats.json, relative_stats.json)
# 2. Dimension mismatch between model and data
# 3. Corrupted checkpoint

# Debug: Run verification scripts first
python custom/scripts/ver1_6/verify_groot_dataset.py --dataset <path> --load-test
```

**Problem:** Actions look wrong / robot behaves erratically
```bash
# Check action representation settings:
# - Arm should use RELATIVE actions
# - Gripper should use ABSOLUTE actions
# See so101_config_1_6.py for correct ActionConfig settings
```

---

## Reference: Successful Verification Output (2025-12-18)

This is what a fully verified dataset looks like:

```
======================================================================
GR00T 1.6 Dataset Verification
======================================================================
Dataset: datasets/so101_pick_place_groot
======================================================================

[1/7] Checking metadata files...
  OK: info.json loaded successfully (dict)
  OK: modality.json loaded successfully (dict)
  OK: stats.json loaded successfully (dict)
  OK: episodes.jsonl loaded successfully (70)
  OK: tasks.jsonl loaded successfully (1)

[2/7] Checking info.json structure...
  OK: codebase_version: v2.1
  OK: total_episodes: 70
  OK: total_frames: 60271
  OK: fps: 30

[3/7] Checking modality.json structure...
  OK: Section 'state' found with keys: ['single_arm', 'gripper']
  OK:   single_arm: [0:5] (dim=5)
  OK:   gripper: [5:6] (dim=1)
  OK: Section 'action' found with keys: ['single_arm', 'gripper']
  OK: Section 'video' found with keys: ['head', 'wrist']
  OK: Annotation section found

[4/7] Checking episode consistency...
  OK: 70 episodes sequential [0, 69]

[5/7] Checking data dimensions...
  OK: State dimension: 6
  OK: Action dimension: 6

[6/7] Checking video files...
  OK: 140 video files (70 head + 70 wrist)
  OK: 640x480, AV1, 30fps

[7/7] Checking statistics file...
  OK: stats.json complete
  OK: relative_stats.json found

[Optional] Testing with LeRobotEpisodeLoader...
  OK: 70 episodes load correctly

======================================================================
RESULT: ALL CHECKS PASSED
Dataset is ready for GR00T 1.6 training!
======================================================================
```
