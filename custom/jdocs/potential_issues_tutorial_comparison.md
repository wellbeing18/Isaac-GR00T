# Potential Issues: Tutorial vs Our Implementation Comparison

**Date:** 2024-12-14
**Purpose:** Document differences between official NVIDIA GR00T tutorial and our custom implementation that may be causing inference issues.

**Reference Tutorial:** https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/3_0_new_embodiment_finetuning.md

---

## Table of Contents

1. [Summary of Findings](#summary-of-findings)
2. [Step 1: Dataset - ✅ MATCHES](#step-1-dataset)
3. [Step 2: Finetuning - ⚠️ KEY DIFFERENCE](#step-2-finetuning)
4. [Step 3: Open-Loop Evaluation - ✅ CORRECT](#step-3-open-loop-evaluation)
5. [Step 4: Deployment - ⚠️ MULTIPLE DIFFERENCES](#step-4-deployment)
6. [Action Items](#action-items)

---

## Summary of Findings

| Step | Component | Status | Issue |
|------|-----------|--------|-------|
| 1 | Dataset structure | ✅ Matches | None |
| 1 | modality.json | ✅ Correct | None |
| 2 | Finetuning script | ✅ Same script | None |
| 2 | **LoRA vs Full Finetuning** | ⚠️ **DIFFERENT** | We use LoRA, tutorial uses full |
| 3 | Open-loop evaluation | ✅ Correct | Minor implementation difference |
| 4 | Execution rate | ⚠️ Different | 30Hz vs 50Hz |
| 4 | Action horizon usage | ⚠️ Different | 16 vs 8 actions |
| 4 | Temporal ensembling | ⚠️ Different | We use it, tutorial doesn't |
| 4 | Fixed random seed | ⚠️ Different | We use seed=42, tutorial doesn't |

---

## Step 1: Dataset

### Official Tutorial
```bash
# Download dataset
huggingface-cli download --repo-type dataset youliangtan/so101-table-cleanup --local-dir ./demo_data/so101-table-cleanup

# Copy modality file for dual camera
cp examples/so100_dualcam__modality.json ./demo_data/so101-table-cleanup/meta/modality.json
```

### Our Implementation
- Dataset path: `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place`
- Structure: `data/`, `meta/`, `videos/` ✅
- modality.json: Present and correctly configured ✅

### Comparison

**Official `so100_dualcam__modality.json`:**
```json
{
    "video": {
        "front": { "original_key": "observation.images.front" },
        "wrist": { "original_key": "observation.images.wrist" }
    }
}
```

**Our `modality.json`:**
```json
{
    "video": {
        "front": { "original_key": "observation.images.head" },
        "wrist": { "original_key": "observation.images.left_wrist" }
    }
}
```

**Status:** ✅ **CORRECT** - Our keys match our actual video folder names (`observation.images.head`, `observation.images.left_wrist`).

---

## Step 2: Finetuning

### Official Tutorial Command
```bash
python scripts/gr00t_finetune.py \
   --dataset-path /datasets/so101-table-cleanup/ \
   --num-gpus 1 \
   --batch-size 64 \
   --output-dir ~/so101-checkpoints \
   --max-steps 10000 \
   --data-config so100_dualcam
```

### Our Command (from `train_groot_mvp.sh`)
```bash
python scripts/gr00t_finetune.py \
    --dataset-path $DATASET_PATH \
    --output-dir $OUTPUT_DIR \
    --num-gpus 1 \
    --max-steps 10000 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --lora-rank 64 \                    # <-- KEY DIFFERENCE
    --save-steps 1000 \
    --gradient-accumulation-steps 4 \
    --warmup-ratio 0.05 \
    --dataloader_num_workers 4 \
    --report-to tensorboard
```

### ⚠️ KEY DIFFERENCE: LoRA vs Full Finetuning

| Parameter | Tutorial (Default) | Ours | Impact |
|-----------|-------------------|------|--------|
| `--lora-rank` | **0** (no LoRA) | **64** | Fundamental training difference |
| `--batch-size` | 64 | 16 | Compensated with grad accum |
| `--gradient-accumulation-steps` | 1 | 4 | Effective batch = 64 ✅ |
| `--video-backend` | torchcodec | torchvision_av | Needed for AV1 codec |

#### What This Means

**Full Finetuning (Tutorial):**
- Trains ALL parameters in Projector + Diffusion Head
- ~60GB+ VRAM required
- Maximum learning capacity
- `new_embodiment` action head trained from scratch

**LoRA Finetuning (Ours):**
- Adds small adapter matrices (~2% of parameters)
- ~18-20GB VRAM required (fits on RTX 4090/5090)
- Reduced learning capacity
- May not have enough capacity for completely new embodiment

#### Why We Use LoRA
We have 24GB VRAM (RTX 4090/5090), which cannot fit full finetuning (~60GB+).

#### Potential Issue
The `new_embodiment` tag maps to projector index 31, which was **never pretrained**. LoRA might not have enough capacity to train this from scratch.

#### Possible Solutions
1. Try higher LoRA rank: `--lora-rank 128`
2. Enable full model LoRA: `--lora-full-model`
3. Use cloud GPU (H100/A100) for full finetuning

---

## Step 3: Open-Loop Evaluation

### Official Tutorial Command
```bash
python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path <YOUR_CHECKPOINT_PATH> \
   --data_config so100_dualcam \
   --dataset_path /datasets/so101-table-cleanup/ \
   --modality_keys single_arm gripper
```

### Our Command
```bash
python custom/scripts/eval_groot_openloop.py \
   --checkpoint <YOUR_CHECKPOINT_PATH> \
   --dataset /path/to/dataset \
   --data-config so100_dualcam \
   --embodiment-tag new_embodiment \
   --plot
```

### Comparison

| Aspect | Official `eval_policy.py` | Our `eval_groot_openloop.py` |
|--------|---------------------------|-------------------------------|
| Script | `scripts/eval_policy.py` | `custom/scripts/eval_groot_openloop.py` |
| LoRA support | ❌ No | ✅ Yes |
| Core function | `gr00t.utils.eval.calc_mse_for_single_trajectory` | Custom reimplementation |
| Video backend | `torchcodec` (default) | `torchvision_av` |

### Ground Truth Collection Difference

**Official (from single observation):**
```python
data_point = dataset.get_step_data(traj_id, step_count)
for j in range(action_horizon):
    gt_action = data_point[f"action.{key}"][j]  # Index into horizon
```

**Ours (fetch each step):**
```python
for h in range(action_horizon):
    gt_obs = dataset.get_step_data(traj_id, step + h)  # Fetch new obs
    gt_action = gt_obs[f"action.{key}"][0]  # Always index 0
```

**Status:** ✅ **FUNCTIONALLY EQUIVALENT** - Both approaches yield the same ground truth if the dataset is structured correctly.

**Why We Need Custom Script:** Official `eval_policy.py` does NOT support LoRA checkpoints. It will fail or produce incorrect results with LoRA adapters.

---

## Step 4: Deployment

### Official Tutorial Command
```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, ...}, front: {type: opencv, ...}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab pens and place into pen holder."
```

### Our Script
```bash
python custom/scripts/infer_groot_async.py \
    --model-path /path/to/checkpoint \
    --task "pick and place"
```

### Architecture Comparison

**Official: Client-Server**
```
┌─────────────────────┐         ┌─────────────────────┐
│   Robot Machine     │ Network │    GPU Server       │
│   (eval_lerobot.py) │ ◄─────► │ (Inference Server)  │
│                     │         │                     │
│  - Get observation  │         │  - Run GR00T model  │
│  - Execute actions  │         │  - Return actions   │
└─────────────────────┘         └─────────────────────┘
```

**Ours: Local Async (Producer-Consumer)**
```
┌─────────────────────────────────────────────────────┐
│              Single Machine with GPU                 │
│                                                     │
│  ┌─────────────────┐     ┌─────────────────┐       │
│  │ Producer Thread │     │ Consumer Thread │       │
│  │ (5-7 Hz)        │────►│ (30 Hz)         │       │
│  │ - Inference     │     │ - Execute       │       │
│  └─────────────────┘     └─────────────────┘       │
│            │                      │                 │
│            ▼                      ▼                 │
│  ┌─────────────────────────────────────────┐       │
│  │       Temporal Ensemble Buffer          │       │
│  └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### ⚠️ Key Parameter Differences

#### 1. Execution Rate

```python
# Official: 50Hz
time.sleep(0.02)  # 20ms between actions

# Ours: 30Hz
action_interval = 0.033  # 33ms between actions
```

**Impact:** Our robot moves through trajectory 40% slower.

**Question:** Is 30Hz correct because our dataset is 30Hz? Or should we match 50Hz?

#### 2. Action Horizon Usage

```python
# Official: Execute 8 actions, then get new prediction
action_horizon = 8
for i in range(action_horizon):
    robot.send_action(action_chunk[i])
    time.sleep(0.02)
# Total time per chunk: 8 × 20ms = 160ms, then refresh

# Ours: Execute all 16 actions with temporal ensembling
action_horizon = 16
# Uses ensemble buffer to average overlapping predictions
```

**Impact:**
- Official refreshes predictions every 160ms (at 50Hz)
- Ours uses older predictions for longer with ensembling

#### 3. Temporal Ensembling

```python
# Official: None
# Just execute actions directly from prediction

# Ours: Sliding window temporal ensembling
class TemporalEnsembleBuffer:
    """
    Maintains overlapping predictions and averages them.
    Reduces variance by sqrt(N) where N = overlapping predictions.
    """
```

**Impact:**
- Smoother actions but adds latency/complexity
- Standard approach in ACT/Diffusion Policy papers
- But official GR00T tutorial doesn't use it

#### 4. Fixed Random Seed

```python
# Official: No fixed seed
action_chunk = self.policy.get_action(obs_dict)

# Ours: Fixed seed every inference call
def get_action(self, ...):
    torch.manual_seed(42)  # Fixed seed
    return self.policy.get_action(obs_dict)
```

**Why We Added This:**
- Flow Matching action head uses `torch.randn()` for initial noise
- Without fixed seed, same input produces different outputs
- This caused "arm vibration" in our tests

**Potential Issue:**
- May prevent model from exploring diverse trajectories
- Official doesn't do this - maybe the randomness is intentional?

### Summary Table

| Feature | Official | Ours | Match? |
|---------|----------|------|--------|
| Architecture | Client-Server | Local Async | Different |
| Execution rate | 50Hz | 30Hz | ⚠️ Different |
| Actions per refresh | 8 | 16 | ⚠️ Different |
| Temporal ensembling | No | Yes | ⚠️ Different |
| Fixed random seed | No | Yes | ⚠️ Different |
| LoRA support | No | Yes | ✅ Ours better |
| Threading | Single (blocking) | Multi (async) | Different |

---

## Action Items

### High Priority (Try First)

1. **Match Execution Rate**
   ```python
   # Change from 30Hz to 50Hz
   action_interval = 0.02  # Instead of 0.033
   ```

2. **Match Action Horizon Usage**
   ```python
   # Execute only 8 actions before refresh
   action_horizon = 8  # Instead of 16
   ```

3. **Remove Fixed Seed (Test)**
   ```python
   # Comment out and see if behavior changes
   # torch.manual_seed(42)
   ```

### Medium Priority

4. **Disable Temporal Ensembling**
   ```bash
   # Already have flag for this
   python infer_groot_async.py --no-ensemble
   ```

5. **Try Higher LoRA Rank**
   ```bash
   # In training
   --lora-rank 128
   ```

### Low Priority (Requires More Resources)

6. **Full Finetuning on Cloud GPU**
   - Rent H100/A100 (80GB VRAM)
   - Remove `--lora-rank` flag
   - Compare results with LoRA version

---

## Questions to Investigate

1. **Is 30Hz correct for our dataset?**
   - Dataset recorded at 30Hz
   - But official deploys at 50Hz
   - Does execution rate need to match recording rate?

2. **Does temporal ensembling help or hurt?**
   - Standard in ACT/Diffusion Policy
   - But official GR00T doesn't use it
   - Need A/B test

3. **Is fixed seed causing issues?**
   - We added it to prevent vibration
   - But maybe randomness provides useful exploration?
   - Official doesn't use fixed seed

4. **Is LoRA sufficient for new embodiment?**
   - `new_embodiment` action head (projector 31) was never pretrained
   - LoRA adds adapters but base weights are random
   - May need full finetuning for best results

---

## Data Pipeline Investigation Results

**Date Added:** 2024-12-14

After comprehensive investigation of the data collection and conversion pipeline:

### Data Pipeline Status: ✅ VERIFIED WORKING

| Component | Status | Details |
|-----------|--------|---------|
| Data collection (30Hz) | ✅ Correct | `collect_xlerobot_data.py` uses ACTION_FPS=30 |
| Video encoding (AV1) | ✅ Correct | Cameras capture MJPG, saved as AV1 |
| Conversion to GR00T | ✅ Correct | Per-episode files, timestamps reset |
| GR00T loading test | ✅ Works | `LeRobotSingleDataset` loads successfully |
| Frame indexing | ✅ Correct | 0-based per episode |
| Video timestamps | ✅ Correct | start_time=0 for all videos |

### Warnings (Not Critical)

| Warning | Impact |
|---------|--------|
| Codebase version v3.0 (official uses v2.1) | GR00T reads paths from info.json, works fine |
| Video path structure different | `{video_key}/chunk/` vs `chunk/{video_key}/` - works fine |
| Extra parquet/video files (78 vs 70) | Harmless leftover files |

### Verification Command

```bash
conda run -n groot python custom/scripts/verify_dataset_for_groot.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place \
    --test-loading --verbose
```

### Conclusion

**The data pipeline is NOT the cause of inference issues.** Focus on deployment parameters (below) and LoRA capacity.

---

## Recommended Test Matrix

Based on the differences found, here's a systematic test plan:

### Test 1: Execution Rate (30Hz → 50Hz)

**Hypothesis:** Robot moves too slowly through trajectory at 30Hz.

```python
# In infer_groot_async.py
action_interval = 0.02  # Change from 0.033 to 0.02
```

### Test 2: Action Horizon (16 → 8)

**Hypothesis:** Using full 16-action horizon keeps stale predictions too long.

```bash
# Match official tutorial
python infer_groot_async.py --action-horizon 8
```

### Test 3: Disable Temporal Ensembling

**Hypothesis:** Ensembling adds latency and complexity that hurts performance.

```bash
python infer_groot_async.py --no-ensemble
```

### Test 4: Remove Fixed Seed

**Hypothesis:** Fixed seed limits action diversity; some randomness may be beneficial.

```python
# Comment out in infer_groot_async.py Gr00tLocalInference.get_action()
# torch.manual_seed(42)
```

### Test 5: Combined Official Settings

**Hypothesis:** All official parameters together work better.

```bash
python infer_groot_async.py \
    --action-horizon 8 \
    --action-interval 0.02 \
    --no-ensemble \
    # And remove fixed seed from code
```

### Test 6: Higher LoRA Rank (Training)

**Hypothesis:** LoRA rank 64 has insufficient capacity for new embodiment.

```bash
# Retrain with higher rank
python scripts/gr00t_finetune.py \
    --lora-rank 128 \
    --lora-full-model \
    ...
```

---

## Files Reference

| Component | Official | Ours |
|-----------|----------|------|
| Finetuning | `scripts/gr00t_finetune.py` | Same |
| Open-loop eval | `scripts/eval_policy.py` | `custom/scripts/eval_groot_openloop.py` |
| Deployment | `examples/SO-100/eval_lerobot.py` | `custom/scripts/infer_groot_async.py` |
| Training script | N/A | `custom/scripts/train_groot_mvp.sh` |
| modality.json | `examples/SO-100/so100_dualcam__modality.json` | `datasets_copy/left/pick_and_place/meta/modality.json` |
| Dataset verification | N/A | `custom/scripts/verify_dataset_for_groot.py` |
| Data pipeline docs | N/A | `custom/jdocs/lora/data_pipeline_investigation.md` |
