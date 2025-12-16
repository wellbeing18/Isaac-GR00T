# GR00T Bad Inference Investigation: Root Cause Analysis

**Date:** 2025-12-13
**Investigator:** Claude
**Problem:** Finetuned GR00T model exhibits vibration/oscillation during inference despite good evaluation metrics (MAE 3.63°, Acc@10 93.5%)

---

## 1. Executive Summary

The root cause of the oscillation is **`--no-tune_diffusion_model`** in our training script, which **froze the action head entirely**. The model could only learn feature projections but never learned how to generate proper action trajectories for SO-101's kinematics.

**Solution:** Remove `--no-tune_diffusion_model` from the training script. This allows the diffusion/flow-matching action head to be trained with LoRA adapters.

---

## 2. Investigation Findings

### Finding 1: Training Configuration Mismatch

**NVIDIA Tutorial (working example):**
```bash
python scripts/gr00t_finetune.py \
   --dataset-path /datasets/so101-table-cleanup/ \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir ~/so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam
```

**Our training script:**
```bash
python scripts/gr00t_finetune.py \
   --dataset-path ... \
   --lora-rank 16 \
   --no-tune_diffusion_model \   # <-- THIS IS THE PROBLEM
   ...
```

The NVIDIA tutorial does NOT use `--no-tune_diffusion_model`. By adding this flag, we froze the entire action head.

### Finding 2: Understanding the Training Flags

From `gr00t_finetune.py --help`:

| Flag | Default | Effect |
|------|---------|--------|
| `--tune-diffusion-model` | **True** | Trains the action head |
| `--no-tune_diffusion_model` | - | **FREEZES** the action head |
| `--tune-visual` | False | Trains vision encoder |
| `--tune-llm` | False | Trains language model |
| `--tune-projector` | True | Trains projector layers |
| `--lora-rank` | 0 | 0=full finetuning, >0=LoRA |
| `--lora-full-model` | False | If True, applies LoRA to vision/LLM backbone |

**Key insight:** With `--no-tune_diffusion_model`, the action head is completely frozen regardless of LoRA settings. The model can only adjust projector layers, not learn action generation.

### Finding 3: Why Evaluation Looked Good But Inference Failed

With `--no-tune_diffusion_model`:
- **Projector layers trained:** Model learned to map vision+state features correctly
- **Action head frozen:** Model used pretrained action generation (for humanoids, not SO-101)

This explains the paradox:
- **Evaluation (open-loop):** Good MAE because projector learned feature mapping
- **Inference (closed-loop):** Oscillation because action head doesn't understand SO-101 kinematics

The pretrained action head was trained on humanoid robots (GR1, etc.) with completely different joint configurations. It cannot generate coherent trajectories for a 6-DOF arm.

### Finding 4: Log Evidence of Oscillation

From inference logs, model predictions oscillate between chunks:

```
Chunk 0: action[0] shoulder_lift = -95.7
Chunk 1: action[0] shoulder_lift = -94.8  (+0.9°)
Chunk 2: action[0] shoulder_lift = -93.8  (+1.0°)
...
Chunk 6: action[0] shoulder_lift = -90.0  (continued UP)
Chunk 7: action[0] shoulder_lift = -88.0  (continued UP)
Chunk 8: action[0] shoulder_lift = -90.3  (REVERSED!)
Chunk 9: action[0] shoulder_lift = -87.7  (reversed again)
```

The action head produces incoherent trajectories because it was never trained on SO-101 data.

---

## 3. Response to Gemini's Investigation

Gemini's investigation (document 7) proposed using `--tune-visual` and `--lora-full-model` to fix a "blind policy" issue. Here is my analysis:

### Where Gemini is Correct:
1. The model oscillates/vibrates - confirmed by logs
2. `--no-tune_diffusion_model` was problematic - freezing components was an issue

### Where Gemini is Incorrect:

#### Issue 1: "Vision blindness" is not the root cause

Gemini claims:
> "The pretrained GR00T vision encoder likely maps 'Red Cube' and 'Green Cube' to very similar embeddings"

This is incorrect because:
1. **Our task is single-task** - only "red cube to white plate", no green cube
2. **GR00T is a 3B parameter VLM** - it can absolutely distinguish a red cube
3. **The task description is explicit** - "pick up the red cube and place it on the white plate"

The pretrained vision encoder is more than capable of understanding this scene. The issue is the action head, not vision.

#### Issue 2: The comparison table is misleading

Gemini's table suggests NVIDIA tutorial uses "Full Finetuning" while we used "LoRA". This misses the point:

| What Matters | Tutorial | Our Script |
|--------------|----------|------------|
| Action head trained? | **Yes** (default) | **No** (frozen by flag) |
| LoRA vs Full | Full (default) | LoRA |

The critical difference is whether the action head is trained, not LoRA vs full finetuning.

#### Issue 3: `--lora-full-model` is overkill

From the help text:
> `--lora-full-model`: Whether to use the full model for LORA. If False, **only the action head will be trained**.

For a simple pick-and-place task with a clearly visible red cube:
- We do NOT need to tune the vision encoder
- We do NOT need to tune the LLM backbone
- We only need to train the **action head** for SO-101 kinematics

Adding `--tune-visual` and `--lora-full-model` would:
- Increase VRAM usage significantly
- Increase training time
- Risk overfitting the vision encoder to our small dataset
- Not address the actual problem (frozen action head)

---

## 4. Correct Solution

### The Fix (Already Applied)

Remove `--no-tune_diffusion_model` from `train_groot_mvp.sh`:

**Before:**
```bash
python scripts/gr00t_finetune.py \
    --lora-rank 16 \
    --no-tune_diffusion_model \  # WRONG - freezes action head
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

### Training Configurations Comparison

| Configuration | Action Head | Vision | VRAM | Use Case |
|---------------|-------------|--------|------|----------|
| **NVIDIA Tutorial** | Full finetune | Frozen | ~60GB | A100/H100 |
| **Our Fix (Option B)** | LoRA | Frozen | ~20GB | RTX 4090/5090 |
| Gemini's Proposal | LoRA | LoRA | ~30GB | Not needed |
| Our Old (Broken) | **Frozen** | Frozen | ~18GB | Doesn't work |

### Recommended Training Command

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place \
    --output-dir ~/outputs/groot_fixed \
    --max-steps 10000 \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 16 \
    --save-steps 1000 \
    --warmup-ratio 0.05 \
    --dataloader_num_workers 4 \
    --report-to tensorboard
```

Key points:
- `--lora-rank 16` - Use LoRA for VRAM efficiency
- NO `--no-tune_diffusion_model` - Action head will be trained
- NO `--tune-visual` - Vision encoder stays frozen (sufficient for this task)
- NO `--lora-full-model` - Only train action head, not backbone

---

## 5. Why This Should Work

With the corrected configuration:

1. **Action head trained with LoRA:** Model will learn SO-101's 6-DOF kinematics
2. **Projector trained:** Feature mapping from vision to action space
3. **Vision frozen:** Pretrained features are sufficient for red cube detection
4. **LLM frozen:** Task understanding is already good

The model will learn:
- How SO-101 joints move together
- What trajectories look like for pick-and-place
- The relationship between visual input and robot actions

---

## 6. Summary

| Aspect | Gemini's Diagnosis | My Diagnosis |
|--------|-------------------|--------------|
| Root Cause | Vision encoder can't distinguish colors | Action head was frozen |
| Solution | `--tune-visual --lora-full-model` | Remove `--no-tune_diffusion_model` |
| Complexity | High (tune backbone) | Low (just remove one flag) |
| VRAM Impact | Higher | Minimal |
| Risk | Overfitting vision to small dataset | None |

**The simplest solution is often correct.** We don't need to tune the vision backbone for a task with a clearly visible red cube. We just need to train the action head so the model can learn how to generate trajectories for SO-101.

---

## 7. Next Steps

1. ~~Remove `--no-tune_diffusion_model` from training script~~ **DONE**
2. Retrain model with corrected configuration
3. Evaluate on training data (expect similar MAE)
4. Test on real robot (expect smooth, coherent motion)

If the model still oscillates after retraining with the action head unfrozen, THEN we can consider:
- Increasing training steps
- Adjusting learning rate
- Adding `--tune-visual` (unlikely to be needed)

---

## 8. Response to GPT-5's Analysis (Document 9)

GPT-5's analysis is the most thorough and nuanced. Key points I agree with:

### Agreement 1: The Baseline Comparison is Critical

GPT-5 points out:
> "Your open-loop MAE **3.63°** is **worse than the action:=state baseline 1.41°**"

This is a crucial observation I missed. If the model's predictions are worse than simply predicting "stay where you are", the model hasn't learned useful behavior. This strongly supports the hypothesis that the action head wasn't being trained properly.

### Agreement 2: Prioritized Recommendations

GPT-5's prioritized approach is correct:
1. **Step 1:** Remove `--no-tune_diffusion_model` (what I proposed)
2. **Step 2:** Try no-LoRA baseline if Step 1 fails
3. **Step 3:** Vision LoRA only if task confusion persists
4. **Step 4:** Expand LoRA coverage if underfitting

This is more systematic than my "just remove the flag" approach.

### Refinement to My Analysis

GPT-5 correctly notes that with LoRA + `--no-tune_diffusion_model`:
- Some LoRA adapters may still train (on attention projections)
- But training is "severely constrained" with only ~0.12% trainable params
- The diffusion module is set to `.eval()` mode during training

This explains why we got reasonable-looking MAE numbers but poor inference - the model could learn *something* through the attention LoRA adapters, but not enough to generate coherent trajectories.

### Updated Recommendation

Based on GPT-5's analysis, the training priority should be:

**Option A (First Try):** Remove `--no-tune_diffusion_model`, keep LoRA
```bash
--lora-rank 16
# (no --no-tune_diffusion_model)
```

**Option B (If A fails):** No LoRA, full finetuning of action head
```bash
# (no --lora-rank, uses default 0)
# (no --no-tune_diffusion_model)
--batch-size 8  # Reduce for VRAM
--gradient-accumulation-steps 4
```

**Option C (If B fails):** Vision LoRA for multi-task disambiguation
```bash
--lora-rank 16
--lora-full-model
--tune-visual
```

### Key Metric to Track

GPT-5's suggestion to always compare against baseline:
```
Model MAE vs Baseline MAE (action := state)
```

If Model MAE > Baseline MAE, the model is not learning useful behavior.
