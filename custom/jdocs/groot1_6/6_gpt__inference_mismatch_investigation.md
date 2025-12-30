# GR00T 1.6 Inference Mismatch Investigation (GPT)

## Executive summary (what I believe is the *primary* mismatch)

Your traces strongly suggest the “reduced `action_horizon`” setting is **not actually being applied at runtime**. Even when a run is configured with `action_horizon: 2` or `4`, the system still performs inference **every 16 control steps** (i.e., executes a full 16-step action chunk before re-planning). This is enough by itself to explain:

- **Bad grasp timing / “grip in the air”**: actions are executed with **hundreds of milliseconds of visual staleness**.
- **Missed blocks / end near the block**: motion continues while the policy is “blind” (open-loop within the chunk).
- **Runaway / never recover / swing away**: errors compound inside a 16-step chunk and can drive joints far outside the training distribution before the next inference.

This finding also explains why open-loop evaluation looks good while real closed-loop behavior is bad: the open-loop eval never exposes compounding closed-loop errors or real-time latency.

## Key finding #1: trace “configured horizon” ≠ “executed horizon”

### Evidence from your own saved traces

In `outputs/inference_traces/trace_20251221_214617/config.json`, the run records:

- `action_horizon: 4`

but the computed analysis (`outputs/inference_traces/trace_20251221_214617/analysis_report.json`) shows inference is triggered at **step 0, 16, 32, 48, ...**, which implies the system is still executing **16 actions per inference**:

- `num_inferences: 79` over `total_steps: 1251`
- `time_between_inferences_ms.mean ≈ 622ms` (close to 16×33ms + overhead)
- slow loops are listed at `step: 0, 16, 32, 48, ...` (these are the inference steps)

This same pattern appears in other traces where the config claims a different horizon:

- `trace_20251221_214755/config.json` records `action_horizon: 2`, yet inference happens at step multiples of 16 as well.

### Direct evidence inside the trace lines (joint states show OOD drift)

In `outputs/inference_traces/trace_20251221_214755/trace.jsonl`, the joint state reaches \(\approx -117^\circ\) shoulder pan (far outside typical training ranges) during the rollout:

- Example lines containing shoulder pan around \(-117^\circ\):
  - `step 581`: `joint_states[0] = -114.1978`, `action_executed[0] = -117.6436`
  - `step 584`: `joint_states[0] = -117.0110`
  - `step 589`: `joint_states[0] = -117.6264`
  - `step 590`: `joint_states[0] = -117.8022`

And then at `step 592` the trace shows `inference_triggered: true` again, matching the “every 16 steps” cadence (592 is divisible by 16).

## Key finding #2: observation staleness is large (and bigger than intended), because inference is blocking and the loop runs ~25Hz

From `outputs/inference_traces/trace_20251221_214617/analysis_report.json`:

- **Effective rate**: ~25Hz (not 30Hz)
- **Inference time**: mean ~115ms, max ~482ms
- **Observation age at action execution**:
  - mean ~343ms
  - max ~959ms

Even if your *conceptual* control design is “30Hz, horizon=8”, the actual measured system behavior is closer to:

- “~25Hz loop” and
- “execute 16 actions before next perception/inference”

which makes the “image staleness” and “can’t grasp” symptoms very plausible.

## Key finding #3: open-loop evaluation does not test the failure mode you see on the robot

`custom/scripts/ver1_6/eval_openloop_1_6.py` performs inference every `ACTION_HORIZON` steps **on dataset observations**, and then compares predicted actions to dataset actions.

Critically, it does **not**:

- feed the model’s predicted actions back to generate the next observation/state,
- include real camera/robot latency,
- include timing jitter / control-rate collapse (30Hz → 25Hz),
- include compounding closed-loop error.

So a good MAE/MSE in open-loop is compatible with poor closed-loop behavior.

## Important context: your checkpoint *does* use relative-action decoding + dataset stats

The finetuned checkpoint’s processor config (`outputs/groot_1_6_augmented_20251220_124942/checkpoint-45000/processor_config.json`) indicates:

- **Action config (new_embodiment)**: `single_arm` is `RELATIVE`, `gripper` is `ABSOLUTE`
- **Processor setting**: `use_relative_action: true`

and the dataset provides matching relative-action statistics:

- `datasets/so101_pick_place_groot_augmented/meta/relative_stats.json`
- `datasets/so101_pick_place_groot_augmented/meta/stats.json`

This is *good* in principle (it means training/inference are consistent in how actions are normalized/decoded), but it makes the system **more sensitive** to stale observations: the “absolute” action targets for a chunk are computed from a reference state at inference time, then executed open-loop while the true state drifts.

## Highest-probability root cause (prioritized)

### Root cause A (highest confidence): execution horizon is effectively 16, regardless of your “ACTION_HORIZON=1/2/4/8” intention

Your traces show that changing `ACTION_HORIZON` in the scripts/config is not changing “how many actions you execute before re-inferencing”.

This means that any conclusions drawn from “we tried horizon=2/4 and it still fails” should be re-interpreted as:

> “we still executed 16-step chunks, but we wrote a different number to `config.json`.”

### Root cause B (high confidence): blocking inference causes stale vision + lower effective control rate

The combination of:

- inference mean ~115ms (sometimes > 400ms),
- occasional camera capture stalls (hundreds of ms),
- and executing 16 actions per inference,

predictably yields ~300–600ms of staleness for many executed actions in a chunk, which directly impacts grasp timing and precise placement.

### Root cause C (medium confidence): OOD drift + lack of recovery

Your trace shows the system can reach shoulder_pan \(\approx -118^\circ\). Once you’re that far OOD, the model likely extrapolates poorly, and without explicit recovery training or constraints, it may never return.

This can be a *secondary* effect that becomes rare/benign once Root cause A/B are fixed, or it can remain an issue even after fixing A/B.

## Secondary hypotheses worth checking (lower confidence, but cheap to verify)

- **Camera stream mismatch**: “head” vs “wrist” swapped physically (still “uses both cameras”, but semantics are wrong).
- **OpenCV buffering**: frames returned by `cap.read()` may be old if the buffer isn’t drained (worsens staleness beyond what timestamps imply).
- **Dataset mismatch in evaluation**: ensure open-loop eval uses the same dataset split & augmentation regime as training (`so101_pick_place_groot_augmented` vs `so101_pick_place_groot`), otherwise it can mislead diagnostics.
- **Action safety constraints**: joint limits/clipping absent; once the policy starts drifting, nothing prevents runaway.

## Concrete verification checklist (fast and decisive)

### Check 1: confirm actual executed horizon in traces

For any trace directory, confirm inference steps happen at `0, 16, 32, ...` by inspecting:

- `analysis_report.json` → `anomalies.slow_loops[*].step` (inference steps cluster there)
- `analysis_report.json` → `timing.time_between_inferences_ms.mean` (≈ 16×interval + overhead)
- `trace.jsonl` → entries where `inference_triggered: true` (their `step` should jump by your intended horizon)

### Check 2: implement a true “execution_horizon” and re-run a trace

Goal: force “replan every N steps” (e.g., N=8), regardless of the model’s predicted horizon (16).

Success criteria in the resulting trace:

- `time_between_inferences_ms.mean` drops roughly by ~2× when going from 16→8
- `observation_age_ms.mean` drops materially
- `inference_triggered` steps show `0, 8, 16, 24, ...` (not `0, 16, 32, ...`)

### Check 3: verify closed-loop stability after fixing horizon

If the robot still “swings away” after true replanning is applied:

- check whether drift begins right after an inference (suggesting a perception/semantics issue),
- or whether it’s a slow accumulation (suggesting lack of recovery / insufficient training diversity).

## Bottom line

Before digging deeper into augmentation correctness or subtle modality/stats mismatches, I would treat this as the highest-leverage diagnosis:

> **Your system is still effectively running 16-step open-loop chunks between perception updates**, producing large observation staleness and enabling OOD drift. The “horizon reduction” knob appears not to be operational in the real execution loop.




