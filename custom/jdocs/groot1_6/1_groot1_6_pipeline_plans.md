# GR00T 1.6 Pipeline Rebuild for SO-101

## Overview
Rebuild the complete GR00T pipeline for SO-101 robot after upgrading from GR00T N1.5 to N1.6. Scripts will be created in `custom/scripts/ver1_6/`.

## Design Principles
1. **Use official GR00T 1.6 scripts as much as possible** - Wrap official APIs, minimize custom code
2. **External config files** - Complex configurations in separate YAML/JSON
3. **Key hyperparameters at script top** - Easy tuning without digging through code
4. **Selective parameter freezing** - GR00T 1.6 does NOT use LoRA; uses parameter freezing instead
5. **NEW_EMBODIMENT tag** - Required for custom robot configurations
6. **Comprehensive comments** - Document decisions, considerations, and rationale

---

## Critical Technical Findings (from source code analysis)

### GR00T 1.6 Architecture
- **Backbone**: Eagle-Block2A-2B-v2 (Qwen3-1.7B + SigLIP2)
- **Action Head**: 32-layer DiT (doubled from N1.5's 16 layers)
- **Inference**: Flow matching with 4 denoising steps (not diffusion's 100+)
- **Total params**: ~3B, but only ~214M trainable by default (7%)

### Finetuning Strategy (NOT LoRA)
GR00T 1.6 uses **selective parameter freezing**, not LoRA:
```python
tune_llm: bool = False           # Freeze Qwen3-1.7B (saves ~1.7B params)
tune_visual: bool = False        # Freeze SigLIP2 vision encoder
tune_projector: bool = True      # Train ~14M params (state/action encoders)
tune_diffusion_model: bool = True # Train ~200M params (32-layer DiT)
```

**Default trainable**: ~214M params = ~7% of model = fits 24GB VRAM

### VRAM Requirements (RTX 5090 24GB)
- Model loading: ~7-8 GB
- Training batch_size=4: ~18-20GB peak
- Training batch_size=16: ~20-22GB peak
- Safe margin with batch_size=16

### Data Pipeline (Two Related Systems)
1. **`modality.json`** (dataset metadata): Defines array slicing
   ```json
   {"state": {"single_arm": {"start": 0, "end": 5}}}
   ```
2. **`ModalityConfig`** (Python code): Defines temporal sampling
   ```python
   ModalityConfig(delta_indices=[0], modality_keys=["single_arm"])
   ```
These must be consistent - `modality_keys` in Python must match keys in `modality.json`.

---

## Source Data
- **Location**: `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`
- **Format**: LeRobot v3.0
- **Episodes**: 70
- **Robot**: SO-101 (6-DOF: 5 arm + 1 gripper)
- **Cameras**: `observation.images.head` + `observation.images.left_wrist`, 640x480, 30fps, AV1
- **Hardware config**: `custom/cfgs/so101_hardware.yaml`

---

## Scripts to Create

### Script 1: Data Conversion (`convert_lerobot_v3_to_groot_1_6.py`)

**Purpose**: Convert LeRobot v3 dataset to GR00T 1.6 compatible format

**Strategy**: Use official `convert_v3_to_v2.py` functions, add GR00T-specific `modality.json`

**Key Configuration Block**:
```python
# ============================================================================
# KEY CONFIGURATION
# ============================================================================
INPUT_DATASET = "/home/jrobot/project/XLeRobot/datasets/left/pick_and_place"
OUTPUT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"

# SO-101 Robot Configuration (must match modality.json and ModalityConfig)
ARM_JOINTS = 5      # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DIMS = 1    # single gripper position
TOTAL_STATE_DIM = 6 # ARM_JOINTS + GRIPPER_DIMS
FPS = 30
# ============================================================================
```

**Steps**:
1. Import functions from official `scripts/lerobot_conversion/convert_v3_to_v2.py`
2. Convert v3 → v2.1 format (per-episode parquet + video files)
3. Generate `meta/modality.json` for SO-101 dual-camera setup
4. Generate `meta/tasks.jsonl` from task metadata
5. Verify output structure

**External Config**: `custom/cfgs/so101_modality.json`
```json
{
  "state": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "action": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "video": {
    "head": {"original_key": "observation.images.head"},
    "wrist": {"original_key": "observation.images.left_wrist"}
  },
  "annotation": {
    "human.action.task_description": {"original_key": "task_index"}
  }
}
```

**Reference**: `scripts/lerobot_conversion/convert_v3_to_v2.py`

---

### Script 2: Data Verification (`verify_groot_dataset.py`)

**Purpose**: Verify converted dataset is valid for GR00T 1.6

**Checks**:
- modality.json schema validity and key consistency
- Episode count matches parquet/video files
- State/action dimensions match modality.json slicing
- Video dimensions (640x480) and codec compatibility
- stats.json completeness for normalization
- Timestamp monotonicity in parquet files
- Try loading with official `LeRobotEpisodeLoader` to validate format

**Reference**: `gr00t/data/dataset/lerobot_episode_loader.py`

---

### Script 3: SO-101 Modality Config (`so101_config_1_6.py`)

**Purpose**: Define modality configuration for GR00T 1.6 training

**Critical**: Keys must exactly match `modality.json` in dataset

```python
#!/usr/bin/env python3
"""
SO-101 Modality Configuration for GR00T 1.6.

CRITICAL: modality_keys must exactly match keys in meta/modality.json:
  - video: "head", "wrist" (maps to observation.images.head, observation.images.left_wrist)
  - state: "single_arm", "gripper" (indices 0-5, 5-6)
  - action: "single_arm", "gripper" (indices 0-5, 5-6)

Decision Notes:
- delta_indices=[0] for video/state: Use current frame only
- delta_indices=list(range(16)) for action: 16-step prediction horizon
- ActionRepresentation.RELATIVE for arm: Predicts delta from current state
- ActionRepresentation.ABSOLUTE for gripper: Direct position target
- ActionType.NON_EEF: Joint space control (not end-effector)
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
)

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
ACTION_HORIZON = 16  # Steps predicted per inference (NVIDIA default)
# ============================================================================

so101_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "wrist"],  # Must match modality.json video keys
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],  # Must match modality.json state keys
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],  # Must match modality.json action keys
        action_configs=[
            # Arm: relative actions (delta from current state)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Gripper: absolute position target
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

**Reference**: `examples/SO100/so100_config.py`

---

### Script 4: Zero-Shot Verification (`verify_zeroshot_1_6.py`)

**Purpose**: Test GR00T N1.6 base model inference before finetuning

**Key Configuration Block**:
```python
# ============================================================================
# KEY CONFIGURATION
# ============================================================================
BASE_MODEL = "nvidia/GR00T-N1.6-3B"
DATASET_PATH = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0"
NUM_TEST_EPISODES = 3
# ============================================================================
```

**Steps**:
1. Load modality config (registers NEW_EMBODIMENT)
2. Load base model with `Gr00tPolicy`
3. Load sample episodes from converted dataset
4. Run inference and verify:
   - Output shape: (B, action_horizon, action_dim) = (1, 16, 6)
   - Output in physical units (denormalized)
   - No NaN/Inf values
5. Report inference timing (should be ~40-50ms on RTX 5090)

**Reference**: `scripts/deployment/standalone_inference_script.py`, `getting_started/policy.md`

---

### Script 5: Training Script (`train_groot_so101_1_6.sh`)

**Purpose**: Finetune GR00T N1.6 on SO-101 data (24GB VRAM)

**Key Insight**: GR00T 1.6 does NOT use LoRA. It uses selective parameter freezing:
- `tune_llm=False`: Freeze 1.7B param language model
- `tune_visual=False`: Freeze vision encoder
- `tune_projector=True`: Train ~14M params
- `tune_diffusion_model=True`: Train ~200M params

```bash
#!/bin/bash
# ===========================================================================
# GR00T 1.6 Finetuning for SO-101
# ===========================================================================
# GR00T 1.6 uses SELECTIVE PARAMETER FREEZING, not LoRA.
# Default config freezes backbone (~2.8B params), trains action head (~214M params).
# This fits comfortably in 24GB VRAM.
#
# Decision Notes:
# - tune_llm=False, tune_visual=False: Freeze VLM backbone (default)
# - tune_projector=True, tune_diffusion_model=True: Train action processing (default)
# - Batch size 16: Safe for 24GB VRAM (~20GB peak)
# - Learning rate 1e-4: NVIDIA recommended for finetuning
# - Action horizon 16: Standard for smooth trajectory prediction
# ===========================================================================

set -x -e

# ===========================================================================
# KEY HYPERPARAMETERS - Modify these for tuning
# ===========================================================================
# Model
BASE_MODEL="nvidia/GR00T-N1.6-3B"
EMBODIMENT_TAG="NEW_EMBODIMENT"

# Dataset
DATASET_PATH="/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"
MODALITY_CONFIG="custom/scripts/ver1_6/so101_config_1_6.py"

# Training
MAX_STEPS=10000           # MVP: 500, Full: 10000
LEARNING_RATE=1e-4        # NVIDIA recommended
GLOBAL_BATCH_SIZE=16      # Safe for 24GB VRAM
WARMUP_RATIO=0.05
WEIGHT_DECAY=1e-5

# Parameter Freezing (GR00T 1.6 default - no flags needed, these are defaults)
# tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True

# Checkpointing
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=5
OUTPUT_DIR="outputs/groot_1_6_so101"

# Data Augmentation
COLOR_JITTER="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

# Resources
NUM_GPUS=1
DATALOADER_WORKERS=4
# ===========================================================================

export NUM_GPUS=$NUM_GPUS

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base_model_path $BASE_MODEL \
    --dataset_path $DATASET_PATH \
    --modality_config_path $MODALITY_CONFIG \
    --embodiment_tag $EMBODIMENT_TAG \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    --use_wandb \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params $COLOR_JITTER \
    --dataloader_num_workers $DATALOADER_WORKERS
```

**MVP Mode**: Set `MAX_STEPS=500` for quick validation (~30 min)

**VRAM Notes**:
- Default config trains ~214M params, needs ~18-20GB
- If OOM: reduce `GLOBAL_BATCH_SIZE` to 8
- Monitor with `nvidia-smi` during training

**Reference**: `examples/SO100/finetune_so100.sh`, `gr00t/experiment/launch_finetune.py`

---

### Script 6: Open-Loop Evaluation (`eval_openloop_1_6.py`)

**Purpose**: Evaluate finetuned model on dataset trajectories (no robot needed)

**Key Configuration Block**:
```python
# ============================================================================
# KEY CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = "outputs/groot_1_6_so101/checkpoint-10000"
DATASET_PATH = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0"

# Evaluation settings
NUM_TRAJECTORIES = 5      # Number of episodes to evaluate
STEPS_PER_TRAJ = 150      # Steps per trajectory
ACTION_HORIZON = 16       # Must match training
# ============================================================================
```

**Strategy**: Use official `gr00t/eval/open_loop_eval.py` as base

**Metrics**:
- MSE/MAE per joint group (single_arm, gripper)
- Trajectory visualization (predicted vs ground truth)
- Interpretation: MSE < 0.01 good, < 0.05 acceptable, > 0.1 poor

**Reference**: `gr00t/eval/open_loop_eval.py`

---

### Script 7: Robot Inference (`infer_groot_so101_1_6.py`)

**Purpose**: Run finetuned model on real SO-101 robot

**Key Configuration Block**:
```python
# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Model
CHECKPOINT_PATH = "outputs/groot_1_6_so101/checkpoint-10000"
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"
DEVICE = "cuda:0"

# Inference Settings
ACTION_HORIZON = 16       # Actions per inference (must match training)
ACTION_INTERVAL = 0.033   # 30Hz execution rate (1/30 seconds)
NUM_DENOISING_STEPS = 4   # DiT denoising iterations (NVIDIA default)

# Hardware Config (external file)
HARDWARE_CONFIG = "custom/cfgs/so101_hardware.yaml"

# Task
DEFAULT_TASK = "pick the red cube from the table"

# Recording
RECORD_IMAGES = False     # Save images to eval_images/
MAX_DURATION = 60.0       # Maximum run duration in seconds
# ============================================================================
```

**Execution Flow**:
```python
# 1. Load modality config (registers NEW_EMBODIMENT)
# 2. Load hardware config from YAML
# 3. Initialize robot connection (serial port)
# 4. Initialize cameras (head + wrist)
# 5. Load Gr00tPolicy with checkpoint
# 6. Main loop:
#    a. Capture images from both cameras
#    b. Read robot state (6 DOF)
#    c. Format observation dict (batch dim, temporal dim)
#    d. Run policy.get_action(observation)
#    e. Execute action chunk at 30Hz
#    f. Repeat until duration exceeded
```

**Observation Format** (critical for correct inference):
```python
observation = {
    "video": {
        "head": np.array(...),   # (B=1, T=1, H=480, W=640, C=3) uint8
        "wrist": np.array(...),  # (B=1, T=1, H=480, W=640, C=3) uint8
    },
    "state": {
        "single_arm": np.array(...),  # (B=1, T=1, D=5) float32
        "gripper": np.array(...),      # (B=1, T=1, D=1) float32
    },
    "language": {
        "annotation.human.action.task_description": [["pick the red cube"]]
    }
}
```

**Action Output**:
```python
action, info = policy.get_action(observation)
# action = {
#   "single_arm": (B=1, horizon=16, D=5) float32 in physical units
#   "gripper": (B=1, horizon=16, D=1) float32 in physical units
# }
```

**Reference**: `getting_started/policy.md`, `custom/scripts/ver1_5/infer_groot_simple.py`

---

## Directory Structure

```
custom/scripts/ver1_6/
├── convert_lerobot_v3_to_groot_1_6.py   # Data conversion
├── verify_groot_dataset.py              # Data verification
├── so101_config_1_6.py                  # Modality config (Python)
├── verify_zeroshot_1_6.py               # Zero-shot test
├── train_groot_so101_1_6.sh             # Training script
├── eval_openloop_1_6.py                 # Open-loop evaluation
└── infer_groot_so101_1_6.py             # Robot inference

custom/cfgs/
├── so101_hardware.yaml                  # Hardware config (existing)
└── so101_modality.json                  # GR00T modality mapping (new)

datasets/so101_pick_place_groot/         # Converted dataset
├── data/chunk-000/
│   └── episode_XXXXXX.parquet
├── videos/chunk-000/
│   ├── head/episode_XXXXXX.mp4
│   └── wrist/episode_XXXXXX.mp4
└── meta/
    ├── modality.json                    # Copied from so101_modality.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── info.json
    └── stats.json
```

---

## Execution Order

1. **Data Conversion**: `python convert_lerobot_v3_to_groot_1_6.py`
   - Input: `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`
   - Output: `datasets/so101_pick_place_groot/`

2. **Data Verification**: `python verify_groot_dataset.py`
   - Validates converted dataset format

3. **Zero-Shot Test**: `python verify_zeroshot_1_6.py`
   - Tests base model with converted data (before training)

4. **Training**: `bash train_groot_so101_1_6.sh`
   - MVP first (MAX_STEPS=500), then full (10k steps)

5. **Open-Loop Evaluation**: `python eval_openloop_1_6.py`
   - Check MSE metrics on dataset trajectories

6. **Robot Inference**: `python infer_groot_so101_1_6.py`
   - Deploy on real SO-101 robot

---

## Critical Reference Files

### GR00T 1.6 Official
- `examples/SO100/so100_config.py` - Modality config example
- `examples/SO100/finetune_so100.sh` - Training script example
- `gr00t/experiment/launch_finetune.py` - Training entry point
- `gr00t/configs/finetune_config.py` - All training parameters
- `gr00t/eval/open_loop_eval.py` - Official evaluation
- `gr00t/policy/gr00t_policy.py` - Policy inference API
- `scripts/lerobot_conversion/convert_v3_to_v2.py` - Official data conversion
- `gr00t/data/dataset/lerobot_episode_loader.py` - Data loading logic
- `gr00t/data/state_action/state_action_processor.py` - Normalization

### Existing 1.5 Scripts (for reference patterns)
- `custom/scripts/ver1_5/convert_lerobot_v3_to_groot.py`
- `custom/scripts/ver1_5/infer_groot_simple.py`
- `custom/scripts/ver1_5/eval_groot_openloop.py`

### Configuration
- `custom/cfgs/so101_hardware.yaml` - Hardware config (cameras, robot ports)

---

## Reference Example: Actual Execution (2025-12-18)

This section documents the actual execution steps performed, including issues encountered and solutions.

### Dataset Verification Run

**Command:**
```bash
conda activate groot
cd /home/jrobot/project/Isaac-GR00T
python custom/scripts/ver1_6/verify_groot_dataset.py --dataset datasets/so101_pick_place_groot --load-test
```

**Initial Issues & Fixes:**

1. **Missing `relative_stats.json`** - Required for relative action support
   ```bash
   # Generate relative_stats.json
   python -c "
   import sys
   sys.path.insert(0, '.')

   # Import modality config to register NEW_EMBODIMENT
   import importlib.util
   spec = importlib.util.spec_from_file_location('so101_config', 'custom/scripts/ver1_6/so101_config_1_6.py')
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)

   from gr00t.data.stats import generate_rel_stats
   from gr00t.data.embodiment_tags import EmbodimentTag

   generate_rel_stats('datasets/so101_pick_place_groot', EmbodimentTag.NEW_EMBODIMENT)
   "
   ```

2. **Missing pip packages** (if running in wrong environment):
   ```bash
   pip install av lmdb opencv-python  # May be needed if not in groot env
   ```

**Successful Output:**
```
======================================================================
Verification Summary
======================================================================

  RESULT: ALL CHECKS PASSED
  Dataset is ready for GR00T 1.6 training!
======================================================================
```

### Zero-Shot Verification

**Important Discovery:** The `Gr00tPolicy` class loads modality_configs from the pretrained model's `processor_config.json`. For NEW_EMBODIMENT, you must override the processor's modality_configs during loading.

**Key Code Pattern for Custom Embodiment:**
```python
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from transformers import AutoModel, AutoProcessor

# 1. Import your modality config to register NEW_EMBODIMENT
import importlib.util
spec = importlib.util.spec_from_file_location('config', 'custom/scripts/ver1_6/so101_config_1_6.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 2. Load model
model = AutoModel.from_pretrained("nvidia/GR00T-N1.6-3B")
model.eval()
model.to(device="cuda:0", dtype=torch.bfloat16)

# 3. Load processor WITH modality_configs override
modality_configs_override = {
    EmbodimentTag.NEW_EMBODIMENT.value: MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
}
processor = AutoProcessor.from_pretrained(
    "nvidia/GR00T-N1.6-3B",
    modality_configs=modality_configs_override  # Critical!
)
processor.eval()

# 4. Now processor.get_modality_configs()["new_embodiment"] works
modality_config = processor.get_modality_configs()["new_embodiment"]
```

### Final Verification Results (2025-12-18)

| Check | Status | Details |
|-------|--------|---------|
| Metadata Files | PASS | info.json, modality.json, stats.json, episodes.jsonl, tasks.jsonl |
| info.json Structure | PASS | v2.1, 70 episodes, 60,271 frames, 30fps |
| modality.json | PASS | state[0:5]=single_arm, state[5:6]=gripper |
| Episode Consistency | PASS | 70 episodes sequential [0-69] |
| Data Dimensions | PASS | State dim=6, Action dim=6 |
| Video Files | PASS | 140 videos, 640x480, AV1, 30fps |
| Statistics | PASS | stats.json complete |
| Relative Stats | PASS | relative_stats.json generated |
| LeRobotLoader Test | PASS | 70 episodes load correctly |
| Model Loading | PASS | GR00T N1.6 loads with NEW_EMBODIMENT override (6.37s) |

### Key Learnings

1. **MODALITY_CONFIGS registration timing**: The modality config Python file must be imported BEFORE using the processor, but the processor itself loads configs from saved JSON - requires override pattern.

2. **relative_stats.json is mandatory**: GR00T 1.6 uses relative actions by default (`use_relative_action=True`). Without `relative_stats.json`, action normalization will fail.

3. **Correct conda environment**: Always ensure `conda activate groot` before running scripts. Base environment lacks critical packages.

4. **Model size reference**:
   - Download: ~6.57 GB
   - VRAM for loading: ~7-8 GB
   - VRAM for inference: ~8-10 GB
   - VRAM for training (batch=16): ~20-22 GB
