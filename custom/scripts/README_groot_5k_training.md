# GR00T 5K LoRA Training Scripts

This directory contains scripts for GR00T LoRA finetuning with comprehensive monitoring and evaluation.

## Key Changes (2025-12-01)

### Critical Fix: LoRA Adapter Loading

**Problem**: GR00T training saves LoRA adapters separately (`adapter_config.json` + `adapter_model.safetensors`), but the standard `Gr00tPolicy` doesn't load them. This is identical to the Pi0.5 PEFT loading issue.

**Solution**: Updated `infer_groot_so101.py` with:
- `is_lora_checkpoint()` - Detects LoRA checkpoints
- `load_groot_with_lora()` - Loads base model, applies PEFT adapter, merges weights
- Auto-detection in `Gr00tLocalInference` class

### New Scripts

| Script | Purpose |
|--------|---------|
| `verify_groot_training_setup.py` | Pre-training verification (dataset, model, config) |
| `evaluate_groot_checkpoint.py` | Post-training MAE evaluation per checkpoint |
| `diagnose_groot_inference.py` | Inference behavior diagnosis |
| `combine_groot_datasets.py` | Combine multiple datasets for multi-task training |

### Updated Scripts

| Script | Changes |
|--------|---------|
| `train_groot_mini_mvp.sh` | 5000 steps, integrated evaluation and diagnosis |
| `infer_groot_so101.py` | LoRA adapter loading support |
| `convert_lerobot_v3_to_groot.py` | Better task description handling |

## Quick Start

### 1. Verify Dataset
```bash
python custom/scripts/verify_groot_training_setup.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot
```

### 2. Run 5K Training
```bash
cd /home/jrobot/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

### 3. Evaluate Checkpoints
```bash
python custom/scripts/evaluate_groot_checkpoint.py \
    --training-dir /path/to/output \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot
```

### 4. Run Inference
```bash
python custom/scripts/infer_groot_so101.py \
    --model-path /path/to/checkpoint \
    --task "pick red_cube from center"
```

## Data Pipeline

### Single Dataset
```
LeRobot v3 Dataset → convert_lerobot_v3_to_groot.py → GR00T Dataset
```

### Multi-Task Dataset
```
Task A Dataset ─┐
Task B Dataset ─┼─ combine_groot_datasets.py → Combined GR00T Dataset
Task C Dataset ─┘
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
| Overall MAE | < 10° |
| Per-Joint MAE | < 15° |
| Acc@5° | > 50% |
| Acc@10° | > 80% |

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
├── train_groot_mini_mvp.sh          # Main training script (5K steps)
├── infer_groot_so101.py             # Inference with LoRA support
├── verify_groot_training_setup.py   # Pre-training verification
├── evaluate_groot_checkpoint.py     # Post-training evaluation
├── diagnose_groot_inference.py      # Inference diagnosis
├── combine_groot_datasets.py        # Multi-task dataset combination
├── convert_lerobot_v3_to_groot.py   # LeRobot v3 → GR00T conversion
└── README_groot_5k_training.md      # This file
```
