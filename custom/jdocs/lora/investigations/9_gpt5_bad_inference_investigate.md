### GR00T “Bad Inference” Investigation (GPT-5): Training Flags / LoRA Configuration

**Date**: 2025-12-14  
**Context**: New training run uses per-episode dataset (`datasets_copy/left/pick_and_place`), but real-robot inference still performs poorly.

### Inputs reviewed
- NVIDIA tutorial: `getting_started/3_0_new_embodiment_finetuning.md` ([link](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/getting_started/3_0_new_embodiment_finetuning.md))
- Gemini report: `custom/jdocs/lora/investigations/7_gemini_bad_inference_investigate.md`
- Claude report: `custom/jdocs/lora/investigations/8_claude_bad_inference_investigate.md`
- Code: `scripts/gr00t_finetune.py`, `gr00t/model/action_head/flow_matching_action_head.py`, `gr00t/utils/peft.py`
- Latest training log evidence (2025-12-13 run):
  - `tune_diffusion_model: False`
  - `lora_rank: 16`
  - `trainable params: 3,276,800 (0.1201%)`
  - open-loop eval: **MAE ≈ 3.63°**
  - dataset baseline: **MAE(action:=state) ≈ 1.41°** (full dataset)

---

## 1) What the flags actually do (confirmed from code)

### A. Default behavior in `scripts/gr00t_finetune.py`
- `tune_diffusion_model` default is **True**:

```76:112:scripts/gr00t_finetune.py
    tune_diffusion_model: bool = True
    ...
    lora_rank: int = 0  # 0 => no LoRA (full finetune for trainable modules)
```

So the NVIDIA tutorial command (which omits `--no-tune_diffusion_model` and omits `--lora-rank`) effectively runs:
- **tune_projector=True**
- **tune_diffusion_model=True**
- **lora_rank=0** (no PEFT LoRA)  
=> **full finetuning of the action head** (projector + diffusion head), with vision/LLM frozen by default.

### B. What `--no-tune_diffusion_model` changes
When `tune_diffusion_model=False`, the action head’s diffusion module is frozen:

```217:254:gr00t/model/action_head/flow_matching_action_head.py
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        ...
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
```

Implication:
- Even if the projector is trainable, the core diffusion policy is not being fully adapted to your dataset/robot dynamics.
- Additionally, the action head sets frozen modules to `.eval()` every training step when `tune_diffusion_model=False`, changing train-time behavior (dropout/etc.) for that submodule.

### C. Interaction with LoRA (PEFT)
If `lora_rank>0`, LoRA modules are added after the “trainable parameter” setup:

```339:346:scripts/gr00t_finetune.py
    if config.lora_rank > 0:
        model = get_lora_model(..., action_head_only=not config.lora_full_model)
```

And the repo’s `get_lora_model()` targets only attention projection Linear layers with names like `q_proj/v_proj/to_q/to_v/...`:

```17:44:gr00t/utils/peft.py
    if isinstance(module, torch.nn.Linear):
        if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
            target_modules.append(name)
```

Implication:
- With LoRA enabled, you are **not** adapting arbitrary layers (e.g., projectors/MLPs) unless they match these names.
- In practice, this can make training extremely parameter-limited (as evidenced by ~0.12% trainable params).

---

## 2) Evaluating Gemini vs Claude claims

### Claude (8) claim: “`--no-tune_diffusion_model` froze the action head entirely; finetuning becomes useless”
- **Partially correct**:
  - It does freeze the diffusion core (`action_head.model`) for full finetuning.
  - With LoRA, some adapters may still train, but training becomes **severely constrained** and can easily underfit.
- Evidence that current setting is not working well:
  - Your open-loop MAE **3.63°** is **worse than the action:=state baseline 1.41°**, which strongly suggests the policy is not learning the right mapping.

### Gemini (7) claim: “blind policy due to frozen vision; need Vision LoRA”
- **Plausible but not proven as the first-order blocker**.
- If your dataset is multi-task with ambiguous initial states (e.g. same home pose but different target locations), frozen vision can contribute to multimodality/oscillation.
- However, before tuning vision, the more immediate red flag is that the policy is not even beating a trivial baseline, which usually points to **insufficient action-head adaptation** (diffusion head / projector training capacity).

---

## 3) Recommendation (prioritized)

### Step 0 (sanity metrics): stop trusting MAE alone
Always report:
- **Baseline MAE(action:=state)** on the same evaluation sampling
- Closed-loop drift / error ratio (open-loop vs closed-loop)

If the model does not beat baseline, do not expect task success.

### Step 1 (most likely fix): remove `--no-tune_diffusion_model`
This aligns your run with the NVIDIA tutorial’s default (action head is trained).

If you keep LoRA:
- Prefer: **LoRA + tune_diffusion_model=True**
- Avoid forcing the diffusion module into always-eval mode during training.

### Step 2 (match tutorial as a reference): run **one** “no-LoRA” baseline training
Use:
- `--lora-rank 0` (omit flag)
- keep defaults (tune_projector=True, tune_diffusion_model=True)
- reduce batch size / increase grad accumulation to fit VRAM

This is the closest reproduction of the official tutorial behavior ([link](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/getting_started/3_0_new_embodiment_finetuning.md)).

### Step 3 (if still oscillating / task confusion): try Vision LoRA
Use `custom/scripts/train_groot_vision_lora.sh`:
- `--lora-full-model --tune-visual`

This is higher VRAM/time, but addresses the “ambiguous initial state requires vision discrimination” failure mode described by Gemini.

### Step 4 (if LoRA remains underpowered): expand LoRA coverage
Given `gr00t/utils/peft.py` currently targets only attention projections, consider:
- adding projector/MLP modules to LoRA targets (or keeping select projector layers fully trainable)

This is often necessary when “LoRA-on-attention-only” underfits small robotics datasets.

---

## 4) Concrete “next run” suggestions

**A. Minimal change (likely improvement):** same as your `train_groot_mvp.sh` but remove `--no-tune_diffusion_model`.

**B. Reference run (tutorial-like):**
- remove LoRA (`--lora-rank` omitted)
- keep defaults (diffusion model trained)
- smaller batch size + grad accumulation to stay within VRAM

**C. If multi-task ambiguity persists:** run `train_groot_vision_lora.sh` (Vision LoRA).









