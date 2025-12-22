# GR00T 1.6 Inference Investigation

## Problem Statement

Open-loop evaluation shows good performance (MAE ~2.28°), but closed-loop robot inference performs poorly with:
- **Misplace issues** - Object placed in wrong location
- **Swing issues** - Arm swings unexpectedly
- **Pick-up-in-air issues** - Gripper closes before reaching object

This gap between open-loop and closed-loop performance suggests the issue is in the inference/execution pipeline, not the model itself.

## Investigation Questions

1. Is there a timing mismatch between inference frequency and robot execution?
2. Is there a camera view vs arm action synchronization issue?
3. What is the actual latency between observation capture and action execution?
4. Are there unexpected delays or gaps in the control loop?

---

## Current Inference Architecture

### Control Loop Flow (from `infer_groot_so101_1_6.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE LOOP (30Hz target)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 0 (inference step):                                       │
│  ├─ Capture images (head + wrist cameras)     ~30-50ms          │
│  ├─ Read robot state (6 DOF)                  ~5-10ms           │
│  ├─ Run model inference                       ~40-50ms          │
│  ├─ Get action buffer [16 actions]                              │
│  ├─ Execute action[0]                         ~5-10ms           │
│  └─ Sleep to maintain 30Hz                                      │
│                                                                 │
│  Steps 1-15 (execution steps):                                  │
│  ├─ Execute action[idx] from buffer           ~5-10ms           │
│  └─ Sleep to maintain 30Hz                                      │
│                                                                 │
│  Step 16 (buffer empty → new inference):                        │
│  └─ Repeat from Step 0                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Timing Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target action rate | 30 Hz | 33.3ms per step |
| Action horizon | 16 steps | ~533ms lookahead |
| Inference frequency | ~1.9 Hz | Every 16 steps |
| Denoising steps | 4 | Flow matching iterations |

### Potential Issues

1. **Observation Latency**: By the time action[15] executes, the observation is ~500ms old
2. **Inference Time Variance**: If inference takes longer than expected, loop falls behind
3. **No Stochastic Seed**: Flow matching uses random noise, causing prediction variance
4. **Sequential Camera Capture**: Head then wrist, not simultaneous

---

## Tracing Solution

### Script 1: `infer_groot_so101_trace.py`

A modified inference script that records detailed traces for investigation.

#### Trace Data Structure

```python
trace_entry = {
    # Step identification
    "step": int,                        # Step number (0, 1, 2, ...)
    "inference_triggered": bool,        # True every ~16 steps
    "action_idx_in_buffer": int,        # Which action from buffer (0-15)

    # High-resolution timestamps (relative to start)
    "t_loop_start": float,              # Loop iteration start
    "t_capture_start": float,           # Camera capture start
    "t_capture_end": float,             # Camera capture end
    "t_state_read": float,              # Robot state read complete
    "t_inference_start": float,         # Model inference start (if triggered)
    "t_inference_end": float,           # Model inference end (if triggered)
    "t_action_sent": float,             # Action sent to robot
    "t_loop_end": float,                # Loop iteration end

    # Input data
    "joint_states": [6 floats],         # Current robot state
    "task_description": str,            # Text prompt

    # Inference data (only when triggered)
    "action_buffer": [[16, 6] floats],  # Full 16-step prediction

    # Execution data
    "action_executed": [6 floats],      # Action sent to robot
    "action_delta": [6 floats],         # action - current_state

    # Derived metrics
    "loop_duration_ms": float,
    "inference_duration_ms": float,
    "capture_duration_ms": float,
}
```

#### Output Structure

```
outputs/inference_traces/trace_YYYYMMDD_HHMMSS/
├── trace.jsonl           # All trace entries (JSON lines format)
├── summary.json          # Statistics and metrics
├── config.json           # Run configuration
└── images/               # Images saved on inference steps
    ├── step_0000_head.jpg
    ├── step_0000_wrist.jpg
    ├── step_0016_head.jpg
    ├── step_0016_wrist.jpg
    └── ...
```

#### Usage

```bash
# Basic usage
python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint outputs/groot_1_6_augmented_*/checkpoint-45000 \
    --task "pick the red cube and place it on the plate" \
    --duration 30

# With custom output directory
python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint <path> \
    --task "pick the red cube" \
    --duration 60 \
    --output-dir outputs/inference_traces

# Dry run (no robot)
python custom/scripts/ver1_6/infer_groot_so101_trace.py \
    --checkpoint <path> \
    --task "pick the red cube" \
    --dry-run
```

### Script 2: `analyze_inference_trace.py`

Post-run analysis and visualization of trace data.

#### Features

1. **Timing Analysis**
   - Loop duration statistics (mean, std, min, max, percentiles)
   - Inference duration breakdown
   - Camera capture timing
   - Time between inferences

2. **Synchronization Analysis**
   - Observation age when action executes
   - Gaps in control loop
   - Action interval consistency

3. **Action Analysis**
   - Action magnitude over time
   - Delta between consecutive actions
   - Gripper state transitions
   - Action buffer utilization

4. **Visualization**
   - Timeline plot showing all components
   - Action trajectory plots per joint
   - Loop duration over time
   - Histogram of timing distributions

5. **Anomaly Detection**
   - Flag steps with unusual timing (>2 std)
   - Flag sudden action jumps
   - Identify dropped frames or gaps

#### Usage

```bash
# Analyze a trace
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_YYYYMMDD_HHMMSS

# With visualization
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_YYYYMMDD_HHMMSS \
    --show-plots

# Save plots to files
python custom/scripts/ver1_6/analyze_inference_trace.py \
    --trace-dir outputs/inference_traces/trace_YYYYMMDD_HHMMSS \
    --save-plots
```

---

## Expected Insights

The traces should reveal:

### 1. Actual Control Frequency
- Is the 30Hz target being maintained?
- How much variance in loop timing?
- Are there periodic slowdowns?

### 2. Inference Impact
- How long does each inference take?
- Does inference time vary significantly?
- How does inference affect loop timing?

### 3. Observation Freshness
- How old is the observation when each action executes?
- Is there drift accumulation over the action horizon?

### 4. Action Consistency
- Are predictions stable across inferences?
- Are there sudden jumps in action values?
- How does gripper timing correlate with arm position?

### 5. Synchronization
- Are camera and state reads aligned?
- Is there latency in camera capture?
- Are actions executed at consistent intervals?

---

## Investigation Workflow

1. **Collect Traces**
   ```bash
   python custom/scripts/ver1_6/infer_groot_so101_trace.py \
       --checkpoint <best_checkpoint> \
       --task "pick the red cube and place it on the plate" \
       --duration 30
   ```

2. **Analyze Traces**
   ```bash
   python custom/scripts/ver1_6/analyze_inference_trace.py \
       --trace-dir outputs/inference_traces/trace_* \
       --show-plots
   ```

3. **Review Images**
   - Check saved images at inference steps
   - Correlate with action predictions
   - Look for camera/arm synchronization issues

4. **Identify Root Cause**
   - Compare timing against expected values
   - Look for anomalies in the data
   - Correlate timing issues with bad behaviors

5. **Implement Fixes**
   - Based on findings, implement targeted fixes
   - Re-run traces to verify improvements

---

## Related Files

| File | Description |
|------|-------------|
| `custom/scripts/ver1_6/infer_groot_so101_1_6.py` | Current inference script (base) |
| `custom/scripts/ver1_6/infer_groot_so101_trace.py` | Tracing inference script |
| `custom/scripts/ver1_6/analyze_inference_trace.py` | Trace analysis script |
| `custom/cfgs/so101_hardware.yaml` | Hardware configuration |
| `eval_outputs/checkpoint_45000/` | Open-loop evaluation results |

---

## Previous Investigation Findings

From earlier investigations (`custom/jdocs/lora/investigations/`):

1. **Stochastic Noise**: Flow matching uses random noise initialization, causing ~2° variance between predictions with same input
2. **Action Horizon**: 16-step predictions mean observations can be ~500ms old
3. **Temporal Ensembling**: Averaging overlapping predictions can reduce variance (implemented in async version)

These findings should be kept in mind when analyzing new traces.
