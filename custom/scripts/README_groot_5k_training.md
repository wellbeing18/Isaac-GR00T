# GR00T LoRA Training Scripts

This directory contains scripts for GR00T LoRA finetuning with comprehensive monitoring and evaluation.

## Key Changes (2025-12-01)

### Critical Fix: LoRA Adapter Loading

**Problem**: GR00T training saves LoRA adapters separately (`adapter_config.json` + `adapter_model.safetensors`), but the standard `Gr00tPolicy` doesn't load them. This is identical to the Pi0.5 PEFT loading issue.

**Solution**: Updated `infer_groot_so101.py` with:
- `is_lora_checkpoint()` - Detects LoRA checkpoints
- `load_groot_with_lora()` - Loads base model, applies PEFT adapter, merges weights
- Auto-detection in `Gr00tLocalInference` class

### Training Scripts

| Script | Steps | Duration | Purpose |
|--------|-------|----------|---------|
| `train_groot_mini_mvp.sh` | 100 | ~5 min | Quick validation before full run |
| `train_groot_mvp.sh` | 5000 | ~1 hour | Full training with best checkpoint selection |

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `verify_groot_training_setup.py` | Pre-training verification (dataset, model, config) |
| `evaluate_groot_checkpoint.py` | Post-training MAE evaluation per checkpoint |
| `diagnose_groot_inference.py` | Inference behavior diagnosis |
| `combine_groot_datasets.py` | Combine multiple datasets for multi-task training |
| `convert_lerobot_v3_to_groot.py` | Convert LeRobot v3 dataset to GR00T format |
| `infer_groot_so101.py` | Inference with LoRA adapter loading support |

## Quick Start

### Prerequisites
- GR00T dataset already converted to GR00T format at `/home/jrobot/project/XLeRobot/datasets_groot`
- If you have a new LeRobot v3 dataset, convert it first (see Data Pipeline below)

### 1. Quick Validation (Optional but Recommended)
```bash
cd /home/jrobot/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

### 2. Run Full 5K Training
```bash
cd /home/jrobot/project/Isaac-GR00T
bash custom/scripts/train_groot_mvp.sh
```

This script automatically:
- Verifies dataset before training
- Trains for 5000 steps with checkpoints every 500 steps
- Evaluates all checkpoints (MAE)
- Selects best checkpoint (lowest MAE) and creates `best/` symlink
- Runs inference diagnosis on best checkpoint
- Generates comprehensive training report

### 3. Run Inference with Best Checkpoint
```bash
python custom/scripts/infer_groot_so101.py \
    --model-path /path/to/output/best \
    --task "pick red_cube from center"
```

## Data Pipeline

### When is Data Conversion Needed?

**You DON'T need conversion if:**
- Dataset already exists at `/home/jrobot/project/XLeRobot/datasets_groot`
- Dataset has `meta/modality.json`, `meta/tasks.jsonl`, `meta/info.json`

**You DO need conversion if:**
- You have a new LeRobot v3 dataset (e.g., from data collection)
- You want to combine multiple datasets for multi-task training

### Single Dataset Conversion
```bash
# Convert LeRobot v3 → GR00T format
python custom/scripts/convert_lerobot_v3_to_groot.py \
    --input /path/to/lerobot_dataset \
    --output /path/to/groot_dataset \
    --task "pick red_cube from center"
```

### Multi-Task Dataset Combination
```bash
# Combine multiple GR00T datasets
python custom/scripts/combine_groot_datasets.py \
    --datasets /path/to/task_a /path/to/task_b /path/to/task_c \
    --output /path/to/combined_dataset
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 5000 | ~50-60 minutes |
| Batch Size | 4 | Conservative for memory |
| Learning Rate | 1e-4 | 4x higher than default (critical!) |
| LoRA Rank | 16 | Standard for VLMs |
| Save Steps | 500 | 10 checkpoints total |
| VRAM | ~18-20GB | With --no-tune_diffusion_model |

## Expected Results

| Metric | Target |
|--------|--------|
| Training Loss | < 0.3 |
| Overall MAE | < 10 degrees |
| Per-Joint MAE | < 15 degrees |
| Acc@5 degrees | > 50% |
| Acc@10 degrees | > 80% |

## Output Structure

After running `train_groot_mvp.sh`:

```
outputs/groot_mvp_lora_TIMESTAMP/
├── best -> checkpoint-XXXX       # Symlink to best checkpoint (lowest MAE)
├── best_info.txt                 # MAE comparison for all checkpoints
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
├── ...
├── checkpoint-5000/
├── evaluation_results.json       # MAE metrics for all checkpoints
├── diagnosis_results.json        # Inference behavior test results
├── training.log                  # Full training log
├── training_report.txt           # Comprehensive summary
└── runs/                         # TensorBoard logs
```

## Troubleshooting

### Model Not Responding to Inputs
Run diagnosis:
```bash
python custom/scripts/diagnose_groot_inference.py \
    --checkpoint /path/to/checkpoint \
    --dataset /path/to/dataset
```

### High MAE on Specific Joints
Check per-joint MAE in evaluation results. Common issues:
- shoulder_lift and elbow_flex often have higher error
- May need more training data for those poses

### Task Description Mismatch
Ensure inference task matches training:
- Training: Check `datasets_groot/meta/tasks.jsonl`
- Inference: Use `--task` argument with exact match

## Files

```
custom/scripts/
├── train_groot_mini_mvp.sh          # Quick validation (100 steps, ~5 min)
├── train_groot_mvp.sh               # Full training (5K steps, ~1 hour) with best checkpoint
├── infer_groot_so101.py             # Inference with LoRA support
├── verify_groot_training_setup.py   # Pre-training verification
├── evaluate_groot_checkpoint.py     # Post-training evaluation
├── diagnose_groot_inference.py      # Inference diagnosis
├── combine_groot_datasets.py        # Multi-task dataset combination
├── convert_lerobot_v3_to_groot.py   # LeRobot v3 → GR00T conversion
└── README_groot_5k_training.md      # This file
```
