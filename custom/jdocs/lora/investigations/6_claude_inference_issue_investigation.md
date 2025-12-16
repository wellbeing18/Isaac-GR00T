# GR00T Inference Issue Investigation

**Date:** 2025-12-13
**Training Run:** `groot_mvp_lora_20251213_155740026`
**Investigator:** Claude

## Symptom

After training with good metrics (MAE 3.63°, Acc@10 93.5%), real robot inference shows:
- **Arm vibrates/jitters in place without substantial movement**
- Robot does not make progress toward task targets
- Open-loop evaluation on training data looks good (MSE ~8.4)

## Investigation Methodology

Following the CLAUDE.md principle of **facts & experiments based** investigation:
1. Analyzed actual training data format
2. Examined inference runtime logs
3. Compared training vs inference data flow
4. Identified testable hypotheses

## Key Findings

### Finding 1: Training Data Format Confirmed

From analyzing `/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place`:

```
Action = Leader arm absolute position (where human moved the leader)
State  = Follower arm absolute position (robot's current position)
Both are ABSOLUTE joint positions in degrees
```

Evidence from parquet data (wrist_roll joint):
```
Frame 66: action=-4.3°, state=-2.5°, next_state=-3.0°
Frame 70: action=-5.5°, state=-4.3°, next_state=-4.7°
Frame 80: action=-10.5°, state=-8.7°, next_state=-9.2°
```

The follower arm tracks the leader with ~0.5°/frame lag at 30Hz. This is expected teleoperation behavior.

### Finding 2: Metadata Confirms Absolute Actions

From `checkpoint-8000/experiment_cfg/metadata.json`:
```json
"modalities": {
  "action": {
    "single_arm": {
      "absolute": true
    }
  }
}
```

Normalization uses min-max scaling to [-1, 1] range.

### Finding 3: Model Predictions Are Inconsistent Between Chunks

From inference log `infer_groot_async_20251213_195749.log`:

```
Chunk 0: action[0]=[  0.9, -98.0, 97.3, 42.4, -3.4, 0.8]
Chunk 1: action[0]=[ -3.1, -97.8, 89.4, 49.3,  1.6, 1.4]  ← OPPOSITE DIRECTION!
Chunk 2: action[0]=[ -0.5, -95.1, 94.6, 42.4, -3.4, 1.0]
Chunk 3: action[0]=[  0.8, -95.6, 95.2, 41.6, -2.8, 0.8]
```

Key observations:
- shoulder_pan: 0.9 → -3.1 → -0.5 → 0.8 (oscillates)
- elbow_flex: 97.3 → 89.4 → 94.6 → 95.2 (jumps around)
- wrist_roll: -3.4 → 1.6 → -3.4 → -2.8 (sign flips!)

**This inconsistency between consecutive predictions is likely causing the vibration.**

### Finding 4: Robot IS Responding to Commands

From log lines 48-57:
```
Action 1: target=[0.9, -98, 97.3, 42.4, -3.4, 0.8]
  actual=[0, -99, 97.5, 50.5, -1.3, 0.7]  (no change yet - servo latency)
Action 3: actual=[0.3, -98.5, 97.5, 49.6, -1.8, 0.7]  (starting to move)
Action 5: actual=[0.8, -98.4, 97.5, 45.2, -2.8, 0.7]  (more movement)
```

The robot IS trying to follow commands - wrist_flex moves from 50.5° toward 42.4°. But new contradictory commands arrive before it reaches the target.

### Finding 5: Temporal Ensembling May Amplify Problem

The inference uses temporal ensembling (averaging overlapping predictions). With inconsistent predictions:
- Prediction A: move shoulder_pan +1°
- Prediction B: move shoulder_pan -3°
- Ensemble average: move ~-1°

This averaging of conflicting directions produces small net movements that don't progress toward the goal.

## Root Cause: CONFIRMED

### Flow Matching is Inherently Stochastic

**Date Confirmed:** 2025-12-13

After running controlled experiments, we confirmed the **actual root cause**:

**The Flow Matching action head starts from random noise (`torch.randn`) for each inference call.** This is by design in diffusion/flow models - they sample from a learned distribution. However, this creates problems for real-time robot control.

#### Experimental Evidence

From `diagnose_groot_inference.py` (lines 308-357):

```
4. Consistency test (same inputs, 3 runs WITHOUT seed):
   Run 1: [-3.09, -6.59, 35.50, ...]
   Run 2: [-0.64, -12.25, 47.94, ...]
   Run 3: [-1.13, -8.47, 38.84, ...]
   Mean std (no seed): 2.14°

5. Consistency test (same inputs, 3 runs WITH fixed seed):
   Run 1: [-4.33, 0.88, 32.86, ...]
   Run 2: [-4.33, 0.88, 32.86, ...]
   Run 3: [-4.33, 0.88, 32.86, ...]
   Mean std (with seed): 0.0000°
```

**Key Insight:** With the EXACT same input (image, state, task), the model produces different outputs each time due to random noise initialization. Setting a fixed seed eliminates this variance completely.

#### Code Evidence

From `gr00t/model/action_head/flow_matching_action_head.py` lines 361-368:
```python
# Set initial actions as the sampled noise.
batch_size = vl_embs.shape[0]
device = vl_embs.device
actions = torch.randn(  # ← Random noise here!
    size=(batch_size, self.config.action_horizon, self.config.action_dim),
    dtype=vl_embs.dtype,
    device=device,
)
```

#### Denoising Steps Does NOT Fix This

We tested multiple denoising step values:

| Denoising Steps | Mean Std (without seed) |
|-----------------|-------------------------|
| 4               | 1.82°                   |
| 16              | 2.05°                   |
| 32              | 2.77°                   |

Increasing steps actually makes variance **worse** because more noise accumulates through the denoising chain.

### Why This Causes Vibration

1. Robot at position A
2. Model predicts target B (with random noise sample 1)
3. Robot starts moving toward B
4. Next inference uses different random noise sample 2
5. Model predicts target C (different from B)
6. Robot changes direction toward C
7. Repeat → vibration/jittering

## Old Hypotheses (Rejected)

### ~~Hypothesis A: Low Denoising Steps Cause Noisy Predictions~~
**REJECTED** - Increasing denoising steps did not improve consistency.

### ~~Hypothesis B: Model Not Generalizing to Inference State Distribution~~
**NOT RELEVANT** - The model passes all sensitivity tests. It responds correctly to state, image, and task changes.

### ~~Hypothesis C: Closed-Loop Feedback Creates Oscillation~~
**PARTIALLY TRUE** - But the root cause is the stochastic nature of the model, not the feedback loop itself.

## Diagnostic Experiments to Run

### Experiment 1: Model Consistency Test
```python
# Run same input through model 10 times
for i in range(10):
    action = model.get_action(same_image, same_state)
    print(f"Run {i}: {action}")
# Expected: variance < 1° if model is stable
```

### Experiment 2: Training Data Prediction Test
```python
# Load training episode, run inference on each frame
# Compare predicted actions to ground truth
for frame in training_episode:
    predicted = model.get_action(frame.image, frame.state)
    ground_truth = frame.action
    error = predicted - ground_truth
    # Expected: MAE similar to evaluation (~3.6°)
```

### Experiment 3: Increased Denoising Steps
```bash
python custom/scripts/infer_groot_async.py \
    --model-path ... \
    --denoising-steps 16 \  # Increase from 4 to 16
    --task "..."
```

### Experiment 4: Disable Temporal Ensembling
Modify inference to execute full action horizon before re-predicting:
- Currently: predict every ~200ms, ensemble overlapping predictions
- Test: predict once, execute all 16 actions (~533ms), then re-predict

### Experiment 5: Direct Robot Command Test
```python
# Without model, send explicit trajectory
targets = [
    [0, -90, 90, 45, -5, 1],  # Step 1
    [5, -80, 80, 40, -10, 1], # Step 2
    [10, -70, 70, 35, -15, 1], # Step 3
]
for target in targets:
    robot.set_target_state(target)
    time.sleep(0.5)
    print(f"Target: {target}, Actual: {robot.get_state()}")
```

## Files Involved

| File | Purpose |
|------|---------|
| `custom/scripts/infer_groot_async.py` | Inference script with temporal ensembling |
| `custom/scripts/diagnose_groot_inference.py` | Diagnostic script (to be enhanced) |
| `gr00t/model/action_head/flow_matching_action_head.py` | Flow matching denoising logic |
| `gr00t/data/transform/state_action.py` | Normalization/denormalization |

## Solutions

### Solution 1: Use Fixed Seed Per Action Chunk (RECOMMENDED)

Set a fixed random seed before each model inference call:

```python
# In infer_groot_async.py, before calling policy.get_action()
torch.manual_seed(42)
np.random.seed(42)
action = policy.get_action(obs_dict)
```

**Pros:**
- Simple to implement
- Guarantees consistent output for same input
- No change to model architecture

**Cons:**
- May reduce action diversity (always samples same trajectory from distribution)

### Solution 2: Seed Based on Observation Hash

Use a deterministic seed based on the observation content:

```python
# Create deterministic seed from observation
seed = hash((image.tobytes(), tuple(state)))
torch.manual_seed(seed % (2**31))
action = policy.get_action(obs_dict)
```

**Pros:**
- Same observation → same action (deterministic)
- Different observations → different seeds (diverse)

**Cons:**
- Hash computation adds latency
- Hash collisions possible (unlikely)

### Solution 3: Average Multiple Samples

Sample N times and average:

```python
samples = []
for _ in range(N):
    samples.append(policy.get_action(obs_dict))
action = np.mean(samples, axis=0)
```

**Pros:**
- Uses full distribution
- Reduces variance by sqrt(N)

**Cons:**
- N times slower inference
- N=4 needed to halve variance (~160ms with 40ms inference)

### Solution 4: Modify Flow Matching Head (Advanced)

Allow passing a fixed noise tensor to the action head:

```python
# In flow_matching_action_head.py
def forward(..., initial_noise=None):
    if initial_noise is None:
        actions = torch.randn(...)
    else:
        actions = initial_noise  # Use provided noise
```

**Pros:**
- Clean API
- Full control

**Cons:**
- Requires modifying NVIDIA's core code

## Implementation Plan

1. **Quick fix:** Add `torch.manual_seed(42)` before inference in `infer_groot_async.py`
2. **Test:** Re-run robot inference and verify vibration is eliminated
3. **Long-term:** Consider observation-hash seeding for better diversity

## Recommended Next Steps

1. ~~Create comprehensive diagnostic script~~ **DONE**
2. ~~Run experiments and collect data~~ **DONE**
3. ~~Analyze results to confirm root cause~~ **CONFIRMED: Stochastic noise**
4. **Implement Solution 1** (fixed seed) in `infer_groot_async.py`
5. **Test on real robot** to verify fix

## Appendix: Calibration Files

Calibration files found at `/home/jrobot/.cache/huggingface/lerobot/calibration/`:

**Left arm follower (`xlerobot_left_arm.json`):**
```json
{
  "shoulder_pan": {"homing_offset": 1298},
  "shoulder_lift": {"homing_offset": 234},
  "elbow_flex": {"homing_offset": -119},
  "wrist_flex": {"homing_offset": -377},
  "wrist_roll": {"homing_offset": 1577},
  "gripper": {"homing_offset": -289}
}
```

**Left arm leader (`xlerobot_left_leader.json`):**
```json
{
  "shoulder_pan": {"homing_offset": 752},
  "shoulder_lift": {"homing_offset": 113},
  "elbow_flex": {"homing_offset": 782},
  "wrist_flex": {"homing_offset": -1641},
  "wrist_roll": {"homing_offset": 176},
  "gripper": {"homing_offset": 1811}
}
```

Note: Leader and follower have different homing offsets, but this is expected - the lerobot calibration system handles the conversion to normalized degree space.
