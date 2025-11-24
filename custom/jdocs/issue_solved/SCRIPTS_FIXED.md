# Training Scripts Fixed

**Date:** 2025-11-24
**Issue:** Invalid arguments causing training to fail silently

---

## Problem

All three training scripts contained invalid arguments that prevented GR00T training from executing:

```bash
--logging-steps 5      # ❌ Not supported by GR00T
--seed 1000            # ❌ Not supported by GR00T
--warmup-steps 500     # ❌ Not supported (use --warmup-ratio instead)
```

**Result:** Scripts exited with code 0 but training never ran.

---

## Fixed Scripts

### 1. `custom/scripts/train_groot_mini_mvp.sh`

**Removed:**
- `--logging-steps 5`
- `--seed 1000`

**Added:**
- `--gradient-accumulation-steps 1`
- `--warmup-ratio 0.05`

**Status:** ✅ Fixed

### 2. `custom/scripts/train_groot_so101_mvp.sh`

**Removed:**
- `--logging-steps 10`
- `--seed 1000`

**Added:**
- `--gradient-accumulation-steps 1`
- `--warmup-ratio 0.05`

**Status:** ✅ Fixed

### 3. `custom/scripts/train_groot_so101_full.sh`

**Removed:**
- `--logging-steps $LOGGING_STEPS`
- `--warmup-steps 500`
- `--seed 1000`
- `LOGGING_STEPS=100` variable declaration

**Added:**
- `--warmup-ratio 0.05`
(gradient-accumulation-steps was already present)

**Status:** ✅ Fixed

---

## What Changed

### Before (Broken)

```bash
python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 100 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --no-tune_diffusion_model \
    --save-steps 100 \
    --logging-steps 5 \          # ❌ Invalid
    --seed 1000 \                 # ❌ Invalid
    2>&1 | tee $OUTPUT_DIR/mini_mvp_training.log
```

**Error Output:**
```
╭─ Unrecognized options ──────────────────────────╮
│ Unrecognized options: --logging-steps --seed    │
│ ─────────────────────────────────────────────── │
│ For full helptext, run gr00t_finetune.py --help │
╰─────────────────────────────────────────────────╯
```

### After (Fixed)

```bash
python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 100 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --no-tune_diffusion_model \
    --save-steps 100 \
    --gradient-accumulation-steps 1 \     # ✅ Valid
    --warmup-ratio 0.05 \                  # ✅ Valid (replaces warmup-steps)
    2>&1 | tee $OUTPUT_DIR/mini_mvp_training.log
```

**Expected Output:**
- Training will actually execute
- Checkpoints will be saved
- Loss values will be logged
- GPU memory will be utilized

---

## Why These Arguments?

### --warmup-ratio 0.05

Replaces `--warmup-steps` with a ratio-based approach:
- **Mini-MVP (100 steps):** 0.05 × 100 = 5 warmup steps
- **MVP (500 steps):** 0.05 × 500 = 25 warmup steps
- **Full (10,000 steps):** 0.05 × 10,000 = 500 warmup steps

This is NVIDIA's default and scales automatically with training length.

### --gradient-accumulation-steps 1

Explicitly sets gradient accumulation (no accumulation):
- Effective batch size = batch_size × gradient_accumulation_steps
- With value 1: effective batch size = actual batch size
- Included for clarity and potential future tuning

### Removed --seed

GR00T's training script doesn't support manual seed setting. The framework uses its own randomization.

### Removed --logging-steps

GR00T uses its own logging strategy. Checkpoints are saved at `--save-steps` intervals, which is sufficient for monitoring.

---

## Next Steps

### 1. Re-run Mini-MVP Test

```bash
cd ~/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

**Expected Duration:** 5-10 minutes
**What to Check:**
- Training actually executes (not just warnings)
- Checkpoint saved at step 100
- GPU memory usage ~18-20GB
- Loss values in log (not just import warnings)

### 2. Verify Success

Check these files after training:

```bash
# Should have checkpoint directory
ls -la /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test/

# Expected files:
# - checkpoint-100/          ← Trained model
# - mini_mvp_training.log    ← Full training log
# - mini_mvp_inspection_report.txt
```

### 3. Validate Training Log

```bash
# Check log has actual training output
wc -l /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test/mini_mvp_training.log
# Should be 100+ lines (not just 15)

# Check for loss values
grep -i "loss" /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test/mini_mvp_training.log
```

### 4. If Mini-MVP Succeeds

- ✅ Pipeline validated
- ✅ Data format confirmed working
- ✅ VRAM requirements measured
- **Next:** Collect 40 more episodes (reach 50 total)
- **Then:** Run MVP training (500 steps)

---

## Expected Training Output

### Mini-MVP (100 steps)

```
Loading dataset from /home/jrobot/project/XLeRobot/jdocs/top_level/datasets
Found 10 episodes, 1500 frames
Loading GR00T-N1.5-3B model...
Initializing LoRA (rank=16)...
Starting training (100 steps)...

Step 5: loss=0.523
Step 10: loss=0.487
Step 15: loss=0.451
...
Step 95: loss=0.234
Step 100: loss=0.229

Saving checkpoint to .../checkpoint-100/
Training complete!
```

### Success Indicators

✅ **Training executes** (not just exits)
✅ **Loss values present** (decreasing is good but not critical for 100 steps)
✅ **Checkpoint saved** (`checkpoint-100/` directory exists)
✅ **GPU memory < 22GB** throughout training
✅ **No CUDA OOM errors**
✅ **No NaN losses**

---

## Troubleshooting

### Still Getting "Unrecognized options" Error

**Check:** Did you pull the latest scripts?
```bash
cd ~/project/Isaac-GR00T
git status
# Should show custom/ scripts as modified or committed
```

**Solution:** Make sure you're running the fixed scripts from `custom/scripts/`

### "CUDA out of memory"

**Solution 1:** Reduce batch size
```bash
# Edit script, change:
--batch-size 4
# To:
--batch-size 2
```

**Solution 2:** Install flash-attn (reduces VRAM ~20%)
```bash
conda activate groot
conda install -c nvidia cuda-toolkit=12.4 -y
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### "modality.json not found"

**Solution:** Script creates this automatically. If error persists:
```bash
ls -la /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/modality.json
```

Should exist. If not, re-run the script - it creates it in Step 3.

### Training Starts But Crashes Mid-Way

**Check logs for specific error:**
```bash
tail -50 /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test/mini_mvp_training.log
```

Common issues:
- Dataset corruption → Re-collect episodes
- VRAM spike → Reduce batch size
- Missing dependency → Reinstall gr00t package

---

## Summary

**Fixed:** ✅ All three training scripts corrected

**Removed Invalid Args:**
- `--logging-steps`
- `--seed`
- `--warmup-steps`

**Added Valid Args:**
- `--warmup-ratio 0.05`
- `--gradient-accumulation-steps 1`

**Ready To Run:** Yes! Execute mini-MVP test with:
```bash
cd ~/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

**Expected Outcome:** Training will actually execute and save checkpoints.

---

## Commit Suggestion

When committing these fixes:

```bash
cd ~/project/Isaac-GR00T
git add custom/scripts/
git commit -m "fix: Remove invalid arguments from GR00T training scripts

- Remove --logging-steps (not supported by tyro parser)
- Remove --seed (not in GR00T's ArgsConfig)
- Replace --warmup-steps with --warmup-ratio 0.05
- Add --gradient-accumulation-steps for clarity

Previous runs failed silently due to unrecognized options.
Training should now execute properly."

git push origin main
```
