# GR00T SO101 Inference Issue Investigation

**Date**: 2025-12-06
**Status**: Active Investigation
**Problem**: Training metrics and evaluation results look good, but real robot inference performance is poor.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Comparison: Evaluation vs Inference](#pipeline-comparison)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Diagnostic Experiments](#diagnostic-experiments)
5. [Findings and Evidence](#findings-and-evidence)
6. [Recommendations](#recommendations)

---

## Executive Summary

### The Problem
After 5K step LoRA finetuning:
- **Training loss**: 0.0287 (looks good)
- **Evaluation MAE**: < 10° (looks good)
- **Open-loop MSE**: 45.04 (looks good)
- **Real robot inference**: Poor performance (jerky, inaccurate movements)

### Key Finding
**The evaluation pipeline and inference pipeline are fundamentally different**, leading to a false sense of model quality.

| Aspect | Evaluation (Open-Loop) | Real Inference (Closed-Loop) |
|--------|------------------------|------------------------------|
| State source | Ground truth from dataset | Actual robot state (with errors) |
| Error accumulation | None (each step independent) | Compounds over time |
| Camera input | Pre-recorded (perfect) | Real-time (potential corruption) |
| Timing | Unbounded | Real-time constraints |

---

## Pipeline Comparison

### Evaluation Pipeline (Open-Loop)

```mermaid
flowchart TB
    subgraph Dataset["Dataset (Ground Truth)"]
        D1[/"video.front (recorded)"/]
        D2[/"video.wrist (recorded)"/]
        D3[/"state.single_arm (recorded)"/]
        D4[/"action.single_arm (ground truth)"/]
    end

    subgraph Model["GR00T Model"]
        M1["Vision Encoder"]
        M2["Language Encoder"]
        M3["State Encoder"]
        M4["Cross-Attention Fusion"]
        M5["Flow Matching Action Head<br/>(4 denoising steps)"]
    end

    subgraph Eval["Evaluation"]
        E1["Compare predicted vs GT"]
        E2["Calculate MSE/MAE"]
    end

    D1 --> M1
    D2 --> M1
    D3 --> M3
    M1 --> M4
    M2 --> M4
    M3 --> M4
    M4 --> M5
    M5 --> |"predicted action"| E1
    D4 --> E1
    E1 --> E2

    style Dataset fill:#90EE90,stroke:#228B22
    style Model fill:#87CEEB,stroke:#4169E1
    style Eval fill:#FFB6C1,stroke:#DC143C
```

**Key characteristic**: Each evaluation step uses **fresh ground-truth data** from the dataset. Errors do not propagate.

### Real Inference Pipeline (Closed-Loop)

```mermaid
flowchart TB
    subgraph Cameras["Real Cameras (30 FPS)"]
        C1["Head Camera<br/>(USB, idx=4)"]
        C2["Wrist Camera<br/>(USB, idx=6)"]
    end

    subgraph Robot["SO101 Robot"]
        R1["Read Joint Encoders"]
        R2["Dynamixel Motors"]
    end

    subgraph Model["GR00T Model"]
        M1["Vision Encoder"]
        M2["Language Encoder"]
        M3["State Encoder"]
        M4["Cross-Attention Fusion"]
        M5["Flow Matching Action Head"]
    end

    subgraph Execution["Action Execution"]
        E1["Extract 12 actions"]
        E2["Temporal Ensembling"]
        E3["Execute @ 30Hz"]
    end

    C1 --> |"RGB (480x640)"| M1
    C2 --> |"RGB (480x640)"| M1
    R1 --> |"6D state"| M3
    M1 --> M4
    M2 --> M4
    M3 --> M4
    M4 --> M5
    M5 --> |"16 actions predicted"| E1
    E1 --> E2
    E2 --> E3
    E3 --> |"send_action()"| R2
    R2 --> |"actual position<br/>(with error)"| R1

    style Cameras fill:#FFD700,stroke:#B8860B
    style Robot fill:#DDA0DD,stroke:#8B008B
    style Model fill:#87CEEB,stroke:#4169E1
    style Execution fill:#98FB98,stroke:#228B22
```

**Key characteristic**: The robot's actual state (which includes execution errors) feeds back into the next inference cycle. **Errors compound**.

---

## Timing Comparison

### Evaluation Timing (No Constraints)

```mermaid
sequenceDiagram
    participant D as Dataset
    participant M as Model
    participant E as Evaluator

    Note over D,E: Step 0
    D->>M: Load observation[0]
    M->>M: Inference (~100-300ms)
    M->>E: predicted_action[0]
    D->>E: ground_truth[0]
    E->>E: Calculate error

    Note over D,E: Step 16 (next inference point)
    D->>M: Load observation[16]
    M->>M: Inference (~100-300ms)
    M->>E: predicted_action[16]
    D->>E: ground_truth[16]
    E->>E: Calculate error

    Note right of E: No time pressure<br/>Each step independent
```

### Real Inference Timing (Real-Time Constraints)

```mermaid
sequenceDiagram
    participant C as Cameras
    participant R as Robot
    participant M as Model
    participant A as Action Executor

    Note over C,A: Chunk 0 (t=0ms)
    C->>M: Capture frames (~5ms)
    R->>M: Read state (~2ms)
    M->>M: Inference (~150ms)
    M->>A: 16 predicted actions

    loop Execute 12 actions
        A->>R: send_action[i]
        Note over A,R: Sleep 33ms
    end
    Note over A: Total exec: 396ms

    Note over C,A: Chunk 1 (t=~550ms)
    C->>M: Capture NEW frames
    R->>M: Read ACTUAL state
    Note right of R: State includes<br/>accumulated error!
    M->>M: Inference
    M->>A: 16 predicted actions
```

**Critical timing differences**:

| Metric | Evaluation | Real Inference |
|--------|------------|----------------|
| Inference time | Unbounded | Must complete before next observation |
| Observation source | Dataset (perfect) | Real cameras (potential lag/corruption) |
| State feedback | None | Every 550ms (approximate) |
| Control frequency | N/A | ~1.8 Hz effective |

---

## Error Accumulation Analysis

### Why Open-Loop MSE Looks Good But Closed-Loop Fails

```mermaid
flowchart LR
    subgraph OpenLoop["Open-Loop Evaluation"]
        O1["Step 0: GT state"] --> O2["Predict"] --> O3["Error: 5°"]
        O4["Step 1: GT state"] --> O5["Predict"] --> O6["Error: 6°"]
        O7["Step 2: GT state"] --> O8["Predict"] --> O9["Error: 4°"]
        O10["Average MSE: ~25"]
    end

    subgraph ClosedLoop["Closed-Loop (Real)"]
        C1["Step 0: GT state"] --> C2["Predict"] --> C3["Error: 5°"]
        C3 --> C4["Step 1: State + 5° error"] --> C5["Predict"] --> C6["Error: 8°<br/>(compounded)"]
        C6 --> C7["Step 2: State + 13° error"] --> C8["Predict"] --> C9["Error: 15°<br/>(diverging)"]
        C10["MSE grows exponentially!"]
    end

    style O10 fill:#90EE90
    style C10 fill:#FF6B6B
```

### Mathematical Model of Error Accumulation

In closed-loop control with imperfect predictions:

```
state[t+1] = state[t] + action_predicted[t] + noise[t]
error[t+1] = error[t] + prediction_error[t]

If prediction_error has variance σ²:
After N steps: total_error ~ √N × σ (random walk)
```

With MSE=45 (σ ≈ 6.7°), after 100 steps:
- Expected drift: √100 × 6.7° ≈ **67°**

This explains why the robot drifts significantly during inference!

---

## Identified Issues

### Issue 1: Camera Corruption During Inference (CRITICAL)

**Evidence**: `eval_images/img_01198.jpg` shows severe horizontal black banding

```mermaid
flowchart LR
    subgraph Normal["Normal Operation"]
        N1["Camera"] --> N2["USB Bus"] --> N3["CPU Buffer"] --> N4["Model"]
    end

    subgraph Inference["During Inference"]
        I1["Camera"] --> I2["USB Bus<br/>(CONGESTED)"] --> I3["CPU Buffer<br/>(STARVED)"] --> I4["Model<br/>(GPU BUSY)"]
        I5["GPU Memory<br/>Pressure"] -.-> I2
        I6["DMA Contention"] -.-> I3
    end

    style I2 fill:#FF6B6B
    style I3 fill:#FF6B6B
```

**Hypothesis**: USB bandwidth contention or GPU/CPU resource starvation during inference causes partial frame capture.

**Diagnostic Script**: `diagnose_camera_sync.py`

### Issue 2: Timing Mismatch (HIGH)

**Current implementation** (`infer_groot_so101.py`):
- Action horizon: 12 actions
- Action interval: 33ms (30 Hz)
- Total execution: 12 × 33ms = **396ms**
- Effective control frequency: ~**1.8 Hz**

**NVIDIA official** (`eval_lerobot.py`):
- Action horizon: 8 actions
- Action interval: 20ms (50 Hz)
- Total execution: 8 × 20ms = **160ms**
- Effective control frequency: ~**4.5 Hz**

```mermaid
gantt
    title Action Execution Timing Comparison
    dateFormat X
    axisFormat %L

    section Current
    Inference    :0, 150
    Action 1-4   :150, 282
    Action 5-8   :282, 414
    Action 9-12  :414, 546
    Next Obs     :546, 550

    section NVIDIA
    Inference    :0, 150
    Action 1-4   :150, 230
    Action 5-8   :230, 310
    Next Obs     :310, 315
```

### Issue 3: Per-Joint Error Distribution (HIGH)

From open-loop evaluation (`eval_trajectories_traj0.png`):

| Joint | MSE | Status |
|-------|-----|--------|
| shoulder_pan | 10.02° | Acceptable |
| shoulder_lift | **62.90°** | HIGH |
| elbow_flex | **94.86°** | CRITICAL |
| wrist_flex | 26.96° | Moderate |
| wrist_roll | 23.10° | Acceptable |
| gripper | 52.41° | HIGH |

**Observation**: `elbow_flex` and `shoulder_lift` have disproportionately high errors.

```mermaid
pie title Per-Joint MSE Distribution
    "shoulder_pan" : 10
    "shoulder_lift" : 63
    "elbow_flex" : 95
    "wrist_flex" : 27
    "wrist_roll" : 23
    "gripper" : 52
```

### Issue 4: Denoising Steps (MEDIUM)

Current: 4 denoising steps (fast but noisy)
Recommended: 16+ for smooth motion

```mermaid
flowchart LR
    subgraph D4["4 Denoising Steps"]
        A1["Noise"] --> A2["Step 1"] --> A3["Step 2"] --> A4["Step 3"] --> A5["Step 4<br/>(rough)"]
    end

    subgraph D16["16 Denoising Steps"]
        B1["Noise"] --> B2["..."] --> B3["Step 16<br/>(smooth)"]
    end

    style A5 fill:#FFB6C1
    style B3 fill:#90EE90
```

---

## Diagnostic Experiments

### Experiment 1: Camera Corruption Diagnosis

**Objective**: Prove/disprove camera corruption is caused by GPU load

**Method**:
1. Capture 100 frames with NO GPU load
2. Capture 100 frames WITH GPU inference running
3. Measure corruption rate in both conditions

**Tool**: `custom/scripts/diagnose_camera_sync.py`

**Expected output**:
```
Condition: No GPU load
  Frames captured: 100
  Corrupt frames: 0
  Capture latency: 33±2ms

Condition: With GPU inference
  Frames captured: 100
  Corrupt frames: 15
  Capture latency: 33±45ms  <-- High variance indicates problem
```

### Experiment 2: Closed-Loop Simulation

**Objective**: Quantify error accumulation rate using dataset

**Method**:
1. Start with ground-truth state at step 0
2. Run model inference
3. Use PREDICTED action to compute next state (instead of GT)
4. Repeat for N steps
5. Compare final state drift vs open-loop MSE

**Tool**: `custom/scripts/diagnose_closed_loop_sim.py`

**Expected output**:
```
Open-loop MSE: 45.04
Closed-loop after 50 steps:
  State drift (degrees): [15.2, 42.8, 67.3, 23.1, 18.4, 31.2]
  Total drift: 198.0°

This confirms error accumulation is the primary issue.
```

### Experiment 3: Timing Analysis

**Objective**: Identify timing bottlenecks in inference pipeline

**Method**:
1. Instrument each stage with microsecond timestamps
2. Run 100 inference cycles
3. Generate latency distribution

**Tool**: `custom/scripts/diagnose_timing_analysis.py`

**Expected output**:
```
Stage                    Mean (ms)   Std (ms)   Max (ms)
─────────────────────────────────────────────────────────
Camera capture (head)      5.2        2.1        45.3
Camera capture (wrist)     4.8        1.9        38.7
State read                 0.3        0.1         1.2
Model inference          148.3       12.4       215.6
Action extraction          0.1        0.0         0.3
Action send                1.2        0.4         3.8
─────────────────────────────────────────────────────────
Total loop               160.1       14.2       265.1

BOTTLENECK: Model inference (93% of total time)
ISSUE: Camera capture has HIGH variance (max 45ms)
```

### Experiment 4: Denoising Steps Impact

**Objective**: Measure action smoothness vs denoising steps

**Method**:
1. Run open-loop evaluation with steps=[4, 8, 16]
2. Measure MSE and action variance

**Command**:
```bash
# Test different denoising steps
for steps in 4 8 16; do
    python eval_groot_openloop.py \
        --checkpoint /path/to/checkpoint \
        --denoising-steps $steps \
        --save-plot denoising_${steps}.png
done
```

---

## Data Flow Diagram: Full Pipeline

```mermaid
flowchart TB
    subgraph Input["Real-Time Input"]
        I1["Head Camera<br/>640×480 @ 30fps"]
        I2["Wrist Camera<br/>640×480 @ 30fps"]
        I3["Robot Encoders<br/>6 joints"]
        I4["Task Description<br/>String"]
    end

    subgraph Preprocess["Preprocessing"]
        P1["RGB uint8 → float"]
        P2["Crop 95% center"]
        P3["Resize to 224×224"]
        P4["Normalize [-1, 1]"]
        P5["State normalize<br/>(stats.json)"]
    end

    subgraph Model["GR00T Model (~150ms)"]
        M1["Vision Tower<br/>(ViT)"]
        M2["Language Tower<br/>(LLM)"]
        M3["State Encoder"]
        M4["Cross-Attention<br/>Projector"]
        M5["Flow Matching<br/>Action Head"]
    end

    subgraph Output["Action Output"]
        O1["16 actions predicted<br/>(horizon)"]
        O2["Extract 12 actions"]
        O3["Temporal Ensemble<br/>(EMA α=0.8)"]
        O4["Denormalize<br/>(stats.json)"]
    end

    subgraph Execution["Execution (~400ms)"]
        E1["Action 0"]
        E2["Sleep 33ms"]
        E3["Action 1"]
        E4["..."]
        E5["Action 11"]
        E6["Sleep 33ms"]
    end

    subgraph Feedback["Feedback Loop"]
        F1["Robot executes"]
        F2["New state<br/>(with error)"]
        F3["Back to Input"]
    end

    I1 --> P1
    I2 --> P1
    P1 --> P2 --> P3 --> P4 --> M1
    I3 --> P5 --> M3
    I4 --> M2
    M1 --> M4
    M2 --> M4
    M3 --> M4
    M4 --> M5
    M5 --> O1 --> O2 --> O3 --> O4
    O4 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6
    E6 --> F1 --> F2 --> F3
    F3 -.-> I1
    F3 -.-> I3

    style Input fill:#FFD700
    style Preprocess fill:#87CEEB
    style Model fill:#98FB98
    style Output fill:#DDA0DD
    style Execution fill:#FFB6C1
    style Feedback fill:#FF6B6B
```

---

## Recommendations

### Immediate Actions (Quick Fixes)

1. **Increase denoising steps to 16**
   ```bash
   python infer_groot_so101.py --denoising-steps 16
   ```

2. **Match NVIDIA timing: 8 actions × 20ms**
   ```bash
   python infer_groot_so101.py --action-horizon 8 --action-interval 0.02
   ```

3. **Always use `--go-home-first`** to start from known state

### Medium-Term Fixes

4. **Fix camera corruption**
   - Use separate USB controllers for cameras
   - Implement frame validation before inference
   - Add camera buffering/queue

5. **Implement proper temporal ensembling**
   - Current: simple EMA with α=0.8
   - Better: ACT-style chunking with weighted averaging

### Long-Term Improvements

6. **Closed-loop training** (if possible)
   - Train with simulated error injection
   - Make model robust to state perturbations

7. **State estimation**
   - Use Kalman filter to smooth state readings
   - Detect and reject outlier states

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `custom/jdocs/lora/3_inference_issue_investigation_20251206.md` | This document |
| `custom/scripts/diagnose_camera_sync.py` | Camera corruption diagnosis |
| `custom/scripts/diagnose_timing_analysis.py` | Pipeline timing analysis |
| `custom/scripts/diagnose_closed_loop_sim.py` | Error accumulation simulation |

---

## Appendix: Reference Links

- [XLeRobot VLA Training Guide](https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html)
- [GR00T N1.5 SO101 Tuning Blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [NVIDIA Isaac-GR00T Getting Started](https://github.com/NVIDIA/Isaac-GR00T/tree/main/getting_started)
- [LeRobot x NVIDIA Healthcare](https://huggingface.co/blog/lerobotxnvidia-healthcare)
