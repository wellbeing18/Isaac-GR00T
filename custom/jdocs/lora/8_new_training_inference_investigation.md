# GR00T Training Configuration Investigation & Fix

**Date:** 2025-12-14
**Status:** Ready for retraining

---

## 1. Root Cause Identified

The model oscillation/vibration during inference was caused by **`--no-tune_diffusion_model`** in our training script.

### What We Actually Trained (Previous Broken Config)

| Component | Trained? | What it does |
|-----------|----------|--------------|
| Vision Encoder | L Frozen | Extracts image features |
| LLM Backbone | L Frozen | Language understanding |
| Projector |  Trained | Maps visioníaction features |
| **Diffusion/Action Head** | L **FROZEN** | Generates action trajectories |

**Result:** Only ~0.12% of parameters trained. The action generation model never learned SO-101 kinematics.

### Evidence of Failure

- Model MAE: **3.63∞**
- Baseline MAE (action := state): **1.41∞**
- **Model performed WORSE than "do nothing"**

---

## 2. Fix Applied

Removed `--no-tune_diffusion_model` from `custom/scripts/train_groot_mvp.sh`.

**Before:**
```bash
python scripts/gr00t_finetune.py \
    --lora-rank 16 \
    --no-tune_diffusion_model \  # ê PROBLEM: Freezes action head
    ...
```

**After:**
```bash
python scripts/gr00t_finetune.py \
    --lora-rank 16 \
    # (no --no-tune_diffusion_model)
    # Action head WILL be trained with LoRA
    ...
```

---

## 3. Comparison: Our Script vs NVIDIA Tutorial

### NVIDIA Tutorial Command (Working Reference)
```bash
python scripts/gr00t_finetune.py \
   --dataset-path /datasets/so101-table-cleanup/ \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir ~/so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam
```

### Our Script (After Fix)
```bash
python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 8000 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --save-steps 1000 \
    --gradient-accumulation-steps 2 \
    --warmup-ratio 0.05 \
    --dataloader_num_workers 4 \
    --report-to tensorboard
```

### Parameter-by-Parameter Comparison

| Parameter | NVIDIA Tutorial | Our Script | Notes |
|-----------|-----------------|------------|-------|
| `--batch-size` | **64** | 16 | Tutorial uses 4x larger |
| `--max-steps` | 10000 | 8000 | Similar |
| `--lora-rank` | **0** (default) | 16 | **Tutorial = full finetune, Ours = LoRA** |
| `--learning-rate` | 1e-4 (default) | 1e-4 | Same |
| `--tune-diffusion-model` | True (default) | True (fixed) | Same now |
| `--video-backend` | torchcodec (default) | torchvision_av | Should work |
| `--gradient-accumulation-steps` | 1 (default) | 2 | Effective batch = 32 |
| `--warmup-ratio` | 0.05 (default) | 0.05 | Same |
| `--dataloader_num_workers` | 12 (default) | 4 | Reduced for stability |

### Key Difference: LoRA vs Full Finetuning

| Config | Trainable Params | VRAM Required |
|--------|------------------|---------------|
| Tutorial (full finetune) | ~150M (action head + projector) | ~40-60GB (A100) |
| Ours (LoRA rank=16) | ~3.3M (LoRA adapters) | ~18-20GB (RTX 4090/5090) |

---

## 4. Training Options for 24GB VRAM

### Option A: LoRA with Action Head Trained (Current Script - Recommended First)

```bash
# Current train_groot_mvp.sh configuration
--lora-rank 16
--batch-size 16
--gradient-accumulation-steps 2  # effective batch = 32
--max-steps 8000
```

**Pros:**
- Low VRAM (~18-20GB)
- Fast training
- Should work for single-task

**Cons:**
- Less capacity than full finetune
- May underfit complex tasks

### Option B: Full Finetuning (Matches Tutorial)

```bash
# Remove --lora-rank or set to 0
--batch-size 8
--gradient-accumulation-steps 8  # effective batch = 64 (matches tutorial!)
--max-steps 10000
```

**Pros:**
- Matches NVIDIA tutorial exactly
- Maximum learning capacity (~150M params)

**Cons:**
- Slower training
- Larger checkpoint files

### Option C: Higher LoRA Rank (Compromise)

```bash
--lora-rank 32  # or 64
--batch-size 16
--gradient-accumulation-steps 2
```

**Pros:**
- More capacity than rank=16
- Still VRAM efficient

### VRAM Estimates

| Configuration | Estimated VRAM | Fits 24GB? |
|---------------|----------------|------------|
| LoRA rank=16, batch=16 | ~14-18GB |  |
| LoRA rank=32, batch=16 | ~16-20GB |  |
| Full finetune, batch=8 | ~12-16GB |  |
| Full finetune, batch=4 | ~10-12GB |  |
| Full finetune, batch=64 (tutorial) | ~40GB | L |

---

## 5. Recommended Training Plan

### Step 1: Try Current Script (Option A)

```bash
cd ~/project/Isaac-GR00T
bash custom/scripts/train_groot_mvp.sh
```

Expected:
- VRAM: ~18-20GB
- Time: ~1.5-2 hours for 8K steps
- Action head trained with LoRA

### Step 2: Evaluate

```bash
python custom/scripts/evaluate_groot_checkpoint.py \
    --checkpoint <output_dir>/checkpoint-8000 \
    --dataset /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place \
    --num-samples 300
```

**Key metric:** Model MAE should be **better than baseline MAE (1.41∞)**

### Step 3: Test on Robot

```bash
python custom/scripts/infer_groot_async.py \
    --model-path <output_dir>/checkpoint-8000 \
    --task "pick up the red cube and place it on the white plate" \
    --duration 30
```

### Step 4: If Still Failing, Try Option B

Modify `train_groot_mvp.sh`:
```bash
# Change these variables:
BATCH_SIZE=8
GRAD_ACCUM=8          # effective batch = 64
LORA_RANK=0           # disable LoRA = full finetuning
MAX_STEPS=10000
```

---

## 6. Summary of Investigation Findings

### Three AI Investigations Reviewed

| Investigator | Root Cause Identified | Solution Proposed |
|--------------|----------------------|-------------------|
| Gemini (doc 7) | "Blind policy" - vision frozen | `--tune-visual --lora-full-model` |
| Claude (doc 8) | Action head frozen | Remove `--no-tune_diffusion_model` |
| GPT-5 (doc 9) | Action head frozen, MAE worse than baseline | Remove flag, consider full finetune |

### Consensus

All three agree that `--no-tune_diffusion_model` was problematic. The disagreement is on whether vision tuning is needed:

- **Gemini:** Yes, tune vision for color discrimination
- **Claude/GPT-5:** No, action head is the priority; vision tuning is overkill for single-task

### Recommended Priority

1. **First:** Remove `--no-tune_diffusion_model`  Done
2. **Second:** If still failing, try full finetuning (Option B)
3. **Third:** If multi-task confusion, try vision LoRA (Gemini's suggestion)

---

## 7. Files Modified

- `custom/scripts/train_groot_mvp.sh` - Removed `--no-tune_diffusion_model`

## 8. Next Steps

1. Run training with fixed script
2. Evaluate checkpoint against baseline
3. Test on real robot
4. Document results
