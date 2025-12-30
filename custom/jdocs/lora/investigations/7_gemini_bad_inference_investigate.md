# GR00T "Bad Inference" Investigation: Vibration & Oscillation

**Date:** 2025-12-13
**Investigator:** Gemini
**Problem:** Finetuned GR00T model exhibits "tight vibration" and barely moves, despite low validation loss.

---

## 1. Executive Summary

The "vibration" is caused by **Model Oscillation**, where the policy rapidly alternates between opposing actions (e.g., "Go Left" then "Go Right") at 30Hz.

This oscillation stems from a **"Blind Policy"**:
The model's Vision Encoder was frozen during training and failed to learn the visual distinction between "Red Cube" (Go Left) and "Green Cube" (Go Right). Since both scenarios share the same initial robot state ("Home"), the model sees them as identical and outputs a multimodal average or flips between modes, resulting in vibration.

**Solution:** Retrain using **Vision LoRA**. We must unfreeze the vision backbone using LoRA adapters (`--lora-full-model --tune-visual`) so the model can learn to distinguish the targets.

---

## 2. Investigation Timeline & Findings

### Phase 1: Symptom Analysis (Vibration)
*   **Observation:** Robot arm vibrates in place.
*   **Diagnosis Script:** `infer_groot_stable.py` with "Plan Locking" (executing 16-step chunks).
*   **Log Evidence:** Even with Plan Locking, the model predicted opposing trajectories every 0.5 seconds:
    ```
    Plan A: Delta +10.29 deg (Move Left)
    Plan B: Delta -3.69 deg (Reverse/Move Right)
    Plan C: Delta +1.82 deg (Reverse again)
    ```
*   **Conclusion:** The model is not just "noisy"; it is **conflicted**. It has two valid hypotheses about what to do and cannot decide.

### Phase 2: Eliminating Other Causes
We systematically ruled out other potential causes:
1.  **Normalization/OOD:** `check_normalization.py` confirmed the robot's "Home" state is within the training distribution (Z-scores < 2.0).
2.  **Data Quality:** `analyze_dataset_smoothness.py` showed the training data is smooth (Jerk ~0.05 deg/s³), so the model is not overfitting to shaky teleoperation.
3.  **Action Consistency:** `analyze_action_consistency.py` found that for the "Home" state, the training data contains **inconsistent actions** (high variance). This confirms the "Home" state is ambiguous: the human went Left in some episodes and Right in others.

### Phase 3: The "Blind Policy" Discovery
*   **The Missing Link:** The ambiguity (Left vs Right) should be resolved by **Vision**. The image shows where the red cube is.
*   **Why Vision Failed:**
    *   Our training script (`train_groot_mvp.sh`) used `--lora-rank 16` and `--no-tune_diffusion_model` (default).
    *   **Critical Flaw:** By default, LoRA only applies to the **Action Head**. The **Vision Transformer (Backbone) was FROZEN**.
    *   The pretrained GR00T vision encoder (trained on generic data) likely maps "Red Cube" and "Green Cube" to very similar embeddings (e.g., "Small Object").
    *   Without tuning the vision encoder, the policy effectively sees the **same input** for both tasks.
*   **Result:** Same Input (Home State + Ambiguous Vision) -> Randomly sampled Output (Left or Right) -> Oscillation.

---

## 3. Comparison with NVIDIA Tutorial

The user noted the NVIDIA tutorial works well. Why?

| Configuration | Tutorial (Implicit) | Our Old Script (`mvp`) | Our New Fix (`vision_lora`) |
| :--- | :--- | :--- | :--- |
| **LoRA Rank** | `0` (Disabled) | `16` | `16` |
| **Action Head** | **Full Finetuning** | LoRA | LoRA |
| **Vision Backbone** | Frozen | Frozen | **LoRA (Tuned)** |
| **VRAM Usage** | High (A100 class) | Low (24GB friendly) | Low (24GB friendly) |
| **Result** | Works for generic objects | **Fails for color-specific tasks** | **Should Work** |

*   **Tutorial:** Uses Full Finetuning on the Action Head. The pretrained vision features are sufficient for simple tasks like "Pick Apple" (where class features are distinct).
*   **Our Case:** "Red vs Green" requires distinguishing color/texture fine-grained details that the frozen backbone glosses over. We **must** tune the vision encoder.

---

## 4. Recommendations

### Immediate Fix: Vision LoRA Training
We must run a new training job that applies LoRA to the vision backbone.

**Script:** `custom/scripts/train_groot_vision_lora.sh`
**Key Flags:**
```bash
--lora-rank 16 \
--lora-full-model \  <-- Applies LoRA to Backbone (Vision/LLM)
--tune-visual        <-- Unfreezes Vision gradients
```

### Interim Mitigation: One-Euro Filter
While waiting for training, use the `infer_groot_stable.py` script with the added **One-Euro Filter**.
*   This smooths out the oscillation by averaging the "Left/Right" fighting into a stable (though potentially wrong) path.
*   It allows testing the pipeline but won't solve the core "blindness".

### Future Data Collection
*   Ensure the "Home" position always has the target object clearly visible in the center crop (224x224).
*   Avoid "hovering" at the home position without movement; start the task immediately to reduce ambiguous data points.











