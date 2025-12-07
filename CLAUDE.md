# Isaac-GR00T Project

NVIDIA's Isaac GR00T N1.5 - an open foundation model (3B parameters) for generalist humanoid robot reasoning and skills.

## Project Structure

- `gr00t/` - Core library (model, data, training)
- `scripts/` - Official NVIDIA scripts (inference, finetuning, evaluation)
- `getting_started/` - Tutorials and notebooks
- `examples/` - Reference implementations for different robots (SO-100, Libero, RoboCasa)
- `custom/` - **User's custom scripts and documentation**
  - `custom/scripts/` - Training, inference, and evaluation scripts for SO-101
  - `custom/jdocs/` - Documentation and guides
  - `custom/cfgs/` - Custom configurations

## Rules to Follow

### Git Commits
- Never include "Claude Code" or AI attribution in commit messages

### Investigation & Research Methodology

When debugging or investigating issues (especially inference problems), follow these principles:

1. **Facts & Experiments Based** - Never rely on assumptions alone
   - Every hypothesis must be validated with concrete data
   - Create experiments/scripts that generate logs or analysis data specific to assumptions
   - Build probes to expose internal mechanisms (black box → white box)
   - Iterate step-by-step toward the true problem based on evidence

2. **Process Visualization**
   - Generate detailed process diagrams (mermaid) mirroring each inference step
   - Collect and visualize inputs/outputs/logs at each step
   - Use the complete picture to narrow down to the actual issue location

3. **Systematic Probing**
   - Create diagnostic tools/scripts for specific hypotheses
   - Log intermediate values at key pipeline stages
   - Compare expected vs actual behavior with data

4. **Open-Loop vs Closed-Loop Understanding**
   - Understand why evaluation (open-loop) may look good while real inference (closed-loop) fails
   - The difference often lies in: model output → robot execution synchronization
   - Check timing, action chunking, state feedback loops

5. **Reference Documentation**
   - XLeRobot VLA docs: https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html
   - GR00T getting started: https://github.com/NVIDIA/Isaac-GR00T/tree/main/getting_started
   - SO-101 tuning guide: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning


## Key Concepts

### EmbodimentTag
Defines the robot type for action head selection:
- `GR1` - Humanoid with dexterous hands (joint space)
- `OXE_DROID` - Single arm with EEF control (delta)
- `AGIBOT_GENIE1` - Humanoid with grippers (joint space)
- `NEW_EMBODIMENT` - Custom/untrained embodiment

### Data Format
Uses LeRobot-compatible data schema with `modality.json` metadata. Datasets contain (video, state, action) triplets.

### LoRA Finetuning
LoRA adapters reduce VRAM from 60GB+ to ~18-20GB. Training saves adapters separately (`adapter_config.json` + `adapter_model.safetensors`).

## Common Commands

```bash
# Activate environment
conda activate groot

# Training (LoRA finetuning)
bash custom/scripts/train_groot_mvp.sh

# Inference
python custom/scripts/infer_groot_so101.py --model_path <checkpoint>

# Evaluation
python custom/scripts/evaluate_groot_checkpoint.py --checkpoint <path>

# Data conversion (LeRobot v3 to GR00T format)
python custom/scripts/convert_lerobot_v3_to_groot.py --input <path> --output <path>

# Combine multiple datasets
python custom/scripts/combine_groot_datasets.py --datasets <paths> --output <path>
```

## Hardware

- Training tested on: H100, L40, RTX 4090, RTX 5090, A6000
- Inference tested on: RTX 3090, RTX 4090, Jetson
- LoRA finetuning: RTX 4090/5090 with 24GB VRAM sufficient

## Current Work

Working on LoRA finetuning for SO-101 robot arm. See `custom/jdocs/lora/` for detailed guides.

**Current Issue:** Training metrics (loss) and evaluation (MAE) look good, but actual robot inference performs poorly. Investigation ongoing - see `custom/jdocs/lora/3_inference_issue_investigation_20251206.md`.

## Files to Ignore

- `eval_images/` - Evaluation output images
- `outputs/` - Training outputs
- `wandb/` - Weights & Biases logs
- `*.log` - Log files
