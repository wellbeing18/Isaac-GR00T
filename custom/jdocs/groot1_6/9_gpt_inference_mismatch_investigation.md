# GR00T 1.6 (checkpoint-45000) — Trace-based Investigation for 2025-12-22 Runs

This report analyzes the two new inference traces you provided:

- `outputs/inference_traces/trace_20251222_113045` (duration 60s, config says `action_horizon=4`)
- `outputs/inference_traces/trace_20251222_115129` (duration 120s, config says `action_horizon=4`)

You asked specifically to reason from the **input → inference → output loop** and use **trace facts** (not assumptions) to explain:

1) why `trace_20251222_113045` appears to “finish one cube then sit back”  
2) why `trace_20251222_115129` “behaves really bad / looks blind”

---

## Executive summary (what the traces prove vs what they suggest)

- **Proven (important): your action-horizon change is now real.**  
  In both traces, `inference_triggered=true` occurs at steps **0,4,8,12,…** (not 0,16,32,…).  
  This matches the `config.json` setting of `action_horizon: 4` and is consistent with the improved success rate you observed.

- **Proven: the system is still not running at 30Hz.**  
  Both traces report ~**18.5–18.8Hz effective rate**, because each 4-step cycle is dominated by inference latency.  
  This matters because even with `horizon=4`, the policy effectively replans every ~**209–211ms**, not every ~132ms.

- **Trace_113045 “sit back” is consistent with “post-place completion behavior”.**  
  Around `step 152` the policy commands a strong **gripper opening** (gripper jumps from ~6.6 to ~20.9→26.1 within one 4-step chunk), while joint targets look like a “place/release” motion. After that, subsequent chunks show relatively small deltas and the arm stays in the same region.  
  This is consistent with the model treating “one successful place” as task completion and not re-initiating search for another cube.

- **Trace_115129 does not show “grasp intent”.**  
  The gripper stays near **~27 down to ~24** for a long time; there is no abrupt close/open cycle like in trace_113045. The arm still moves, often driving wrist_flex near its upper limit (~100+), suggesting it is “doing something” but not entering a grasp phase.

- **Head-camera-is-ignored cannot be proven from these traces alone.**  
  The traces confirm both cameras are captured (capture timing) and that the observation includes both keys (`head`, `wrist` in the scripts), but they do not contain any “camera contribution” metric.  
  Also: in this analysis environment, the saved `images/*.jpg` in both traces could not be decoded, so I cannot use the saved images as evidence of what was in head vs wrist views.

---

## 1) Sanity check: is `action_horizon=4` really applied?

### Trace evidence (both runs)

`trace_20251222_113045/trace.jsonl` and `trace_20251222_115129/trace.jsonl` both show:

- `step 0`: `inference_triggered=true`
- `step 4`: `inference_triggered=true`
- `step 8`: `inference_triggered=true`
- … continuing every 4 steps

This matches the `config.json` for both runs (`action_horizon: 4`).

### Timing evidence

From `analysis_report.json`:

- `trace_20251222_113045`
  - `time_between_inferences_ms.mean ≈ 211ms`
  - `observation_age_ms.mean ≈ 134ms`
  - `effective_rate_hz ≈ 18.5Hz`

- `trace_20251222_115129`
  - `time_between_inferences_ms.mean ≈ 209ms`
  - `observation_age_ms.mean ≈ 132ms`
  - `effective_rate_hz ≈ 18.8Hz`

Interpretation (still trace-grounded):

- Even with `horizon=4`, you’re effectively executing ~4 actions over ~0.21s, i.e. **~52ms per action step** on average.
- That’s slower than the dataset’s nominal 30Hz. The model may tolerate it, but it increases scene change per action and can amplify “miss” behaviors.

---

## 2) Trace_20251222_113045: why it “places one cube then sits back”

### What the trace shows at the key moment (step ~152)

The most obvious “phase change” in this run occurs around `step 144–168`, where the gripper target increases rapidly.

From `trace_20251222_113045/inference.log`:

- At `[INF 144]` (t≈9.02s), the chunk ends with `Gripper … -> 4.1`
- At `[INF 148]` (t≈9.21s), the chunk ends with `Gripper … -> 15.1`
- At `[INF 152]` (t≈9.39s), the chunk ends with `Gripper … -> 26.1`

At `[INF 152]`, the script logs:

- **State now**: `[..., gripper ≈ 6.6]`
- **Action[0] gripper ≈ 20.9**
- **Action[3] gripper ≈ 26.1**

This is a large, directed gripper change within one short-horizon chunk, consistent with “release/open” at the destination.

You can also see the same in `trace_20251222_113045/trace.jsonl`:

- `step 152` is an inference step (`inference_triggered=true`) with `action_buffer[..., gripper]` values around **20.9→23.2→24.9→26.1**

### What happens after that (why it looks like “sit back”)

Immediately after the large gripper-opening chunk:

- subsequent inferences (`INF 156`, `INF 160`, `INF 164`, `INF 168`, …) show:
  - joint deltas are relatively small compared to the approach phase
  - gripper stays in the mid-20s (open) instead of returning to the low values seen at the start

This pattern matches “task reached the place location and is no longer pursuing a grasp target”.

### Likely explanation (grounded + explicit where it is a hypothesis)

**Trace-grounded observation**: The policy transitions into a “place/release + settle” regime and does not re-initiate a new approach-to-block regime afterward (within the run window you highlighted).

**Hypothesis (very plausible)**: This is not “blindness”, it’s **task semantics**:

- The dataset has only one task string (`datasets/so101_pick_place_groot_augmented/meta/tasks.jsonl` contains a single task: “pick up the blocks and place them on the plate”).
- If demonstrations typically end after placing one block (common for teleop episodes), the model will learn that “after a successful place, stop / retract / idle”, not “repeat for remaining blocks”.

This hypothesis fits your symptom: “it placed the first cube, then sits back while other cubes remain” — because “multiple remaining cubes” is not necessarily represented as “continue the task” in training.

---

## 3) Trace_20251222_115129: why it “behaves really bad / looks blind”

### What the trace proves (early behavior)

From `trace_20251222_115129/inference.log` at the very beginning:

- `[INF 0]` state: `[shoulder_pan≈-0.2, shoulder_lift≈-7.2, elbow≈46.1, wrist_flex≈68.9, wrist_roll≈-2.4, gripper≈27.3]`
- The predicted 4-step trajectories repeatedly:
  - adjust wrist_flex upward (e.g., `+3–4.5 deg` per chunk early on)
  - keep gripper approximately constant (~27)

The gripper line stays around **27 → 26 → 25 → 24** across many inferences and does not show the kind of decisive open/close transition seen in trace_113045.

### What the trace proves (later behavior around step ~150)

In `trace_20251222_115129/trace.jsonl` around `step 150`:

- `joint_states` includes wrist_flex around **100.44** (very high; at/above typical max in your dataset stats).
- gripper is still around **24.25**, and action gripper stays around **24.2–24.3**.

So the arm is moving into extreme joint configurations without ever executing a “grasp” behavior, which can look like “it’s doing random bad stuff”.

### Likely explanation categories (what traces suggest, not prove)

Based on trace facts (no scene images available), the most consistent interpretations are:

- **Category A (perception / visibility)**: the policy does not reach a state where it believes a grasp is possible (gripper never closes).  
  This can happen if:
  - blocks are not visible in the *dominant* visual stream used for grasp decisions (often wrist),
  - the camera extrinsics/layout differ from training (objects appear in unexpected regions),
  - or one stream is degraded (blur/overexposure) and the model relies on the other stream in training.

- **Category B (distribution shift / recovery)**: the policy drifts toward joint limits while searching, and without explicit recovery behaviors it keeps pushing into extremes.  
  The wrist_flex approaching ~100 is a concrete example of “near-limit behavior”.

---

## 4) “Head camera doesn’t play any effect” — what can we conclude from traces?

### What traces confirm

- The software is attempting to capture both cameras each inference step (capture duration is low and stable).
- The model is being called on the dual-camera observation format (per the scripts).

### What traces do **not** provide

There is no direct trace field like:

- “attention weight per camera”
- “vision embedding norm per view”
- “action delta if one camera were blacked out”

So **these two traces alone cannot prove** that head is ignored.

### Practical note from this investigation

In this environment, `outputs/inference_traces/.../images/step_XXXX_{head,wrist}.jpg` files could not be decoded (tool reported “corrupted or unsupported format”). That prevents using the saved images as evidence to confirm the “wrist sees empty but head sees blocks” hypothesis.

If you can open those JPGs locally and confirm their content, we can make a much stronger trace+image-grounded argument.

---

## 5) Trace-grounded next checks (high leverage)

### Check A: confirm “multi-block continuation” is unsupported by training

Goal: determine whether demonstrations typically end after one pick-place.

Fast ways:

- inspect a few training episodes and count grasp/release cycles (manual video scrub is enough)
- or compute “number of gripper open/close cycles per episode” from the dataset actions

If episodes mostly show **one** pick-place, then trace_113045’s “stop after one” is expected.

### Check B: isolate head vs wrist contribution (camera ablation)

This is the cleanest way to validate your head-camera hypothesis:

- Run a short trace where head frames are replaced with black (or constant) but wrist is real
- Run another where wrist is black but head is real
- Compare predicted `action_buffer` (especially shoulder_pan direction and approach behavior)

If wrist-black causes near-identical actions to baseline, head is being ignored (or uninformative).

### Check C: validate starting pose / joint limits vs training distribution

In trace_115129, wrist_flex reaches ~100+ and the policy never grasps.  
Once near limits, behavior can degrade. If your runtime scene frequently drives “search” into joint limits, you likely need:

- explicit recovery data (how to back off and re-acquire)
- or a safety controller to prevent limit-hugging while the policy is unsure

---

## Bottom line

- Your new traces provide strong evidence that **shorter horizon is actually applied now**, and the timing metrics support why success rate improved.
- `trace_20251222_113045` shows a clear “place/release then settle” phase (big gripper opening + small subsequent deltas), which is consistent with the model treating “one cube placed” as completion rather than “repeat for multiple cubes”.
- `trace_20251222_115129` shows motion without grasp intent (gripper nearly constant) and drift toward extreme joint configurations, which is consistent with perception/visibility issues and/or lack of recovery — but does not, by itself, prove head-camera is ignored.










