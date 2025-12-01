# SO101 GR00T Inference Script

Run finetuned GR00T models on the SO101 robot arm with dual camera support.

## Prerequisites

1. **groot conda environment** with Isaac-GR00T installed
2. **LeRobot** installed in groot env (see Installation below)
3. **Finetuned GR00T checkpoint** (e.g., from `train_groot_mini_mvp.sh`)

## Installation

LeRobot v0.4.1 is already installed in the groot environment. After installing lerobot, restore gr00t dependencies:

```bash
conda activate groot

# If lerobot not yet installed:
conda install -c conda-forge evdev  # Fixes kernel 6.14 build issue
pip install -e /home/jrobot/project/lerobot --no-deps

# Restore gr00t-compatible dependency versions
pip install accelerate==1.2.1 av==12.3.0 gymnasium==1.0.0 "numpy<2.0.0" pyarrow==14.0.1 wandb==0.18.0

# Restore torch/flash-attn for gr00t
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# Download and install flash-attn wheel:
cd /tmp && wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "/tmp/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

## Usage

### Quick Start (Bash Wrapper)

```bash
cd /home/jrobot/project/Isaac-GR00T

# Run with defaults
./custom/scripts/infer_groot_so101.sh

# Specify task and execution length
./custom/scripts/infer_groot_so101.sh --task "grasp object" --actions-to-execute 50

# Show all options
./custom/scripts/infer_groot_so101.sh --help
```

### Direct Python Execution

```bash
conda activate groot
cd /home/jrobot/project/Isaac-GR00T

python custom/scripts/infer_groot_so101.py \
    --model-path /home/jrobot/project/XLeRobot/outputs/groot_mini_mvp_test \
    --task "grasp object" \
    --actions-to-execute 100
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | `.../groot_mini_mvp_test` | Path to finetuned checkpoint |
| `--data-config` | `so100_dualcam` | Data config for dual cameras |
| `--embodiment-tag` | `new_embodiment` | Embodiment tag used during training |
| `--task` | `grasp object` | Task description (must match training data) |
| `--port` | `/dev/ttyACM0` | Serial port for robot motor bus |
| `--head-cam-idx` | `4` | Head camera device index |
| `--wrist-cam-idx` | `6` | Wrist camera device index |
| `--action-horizon` | `12` | Actions to execute per inference (of 16 predicted) |
| `--actions-to-execute` | `100` | Total action chunks to run |
| `--denoising-steps` | `4` | Diffusion denoising steps |
| `--calibrate` | flag | Force robot recalibration |
| `--no-display` | flag | Disable camera preview |
| `--record-imgs` | flag | Save images to `eval_images/` folder |

## Task Vocabulary

The task description must match what was used during training. Check your dataset:

```bash
cat /home/jrobot/project/XLeRobot/jdocs/top_level/datasets_groot/meta/tasks.jsonl
```

Current task: `"grasp object"`

## Troubleshooting

### Camera Not Found
- Verify camera indices: `ls /dev/video*`
- Test cameras: `v4l2-ctl --list-devices`

### Robot Connection Issues
- Check USB connection to Dynamixel motors
- Verify motor IDs match SO101 configuration
- Try `--calibrate` flag to recalibrate

### Model Loading Errors
- Ensure checkpoint path contains `experiment_cfg/metadata.json`
- Verify embodiment tag matches training config

## Files

- `infer_groot_so101.py` - Main Python inference script
- `infer_groot_so101.sh` - Bash wrapper with environment setup
- `README_infer_groot_so101.md` - This file
