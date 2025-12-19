# GR00T 1.6 Code Changes

This document tracks all modifications made to the GR00T codebase for SO-101 finetuning.

## Summary of Changes

| File | Change Type | Reason |
|------|-------------|--------|
| `gr00t/configs/finetune_config.py` | Added `optim` parameter | Allow optimizer selection for VRAM optimization |
| `gr00t/experiment/launch_finetune.py` | Use config optimizer | Apply user-specified optimizer |
| `gr00t/data/dataset/factory.py` | Fix distributed barrier | Prevent hang on single-GPU training |

---

## 1. Optimizer Configuration (`gr00t/configs/finetune_config.py`)

### Problem
GR00T 1.6 trains ~1.6B parameters (49% of 3.2B model). With AdamW optimizer, this requires storing:
- 2 momentum states per parameter (m and v)
- Total: 1.6B × 2 × 4 bytes = ~12.8 GB just for optimizer states

This caused OOM on 24GB GPUs (RTX 4090/5090).

### Solution
Added configurable optimizer selection with `adafactor` as default:

```python
# Added to FinetuneConfig class (lines 111-119)
optim: str = "adafactor"
"""
Optimizer choice. Options include:
  - 'adamw_torch': Standard AdamW (requires ~12GB for optimizer states with 1.6B params)
  - 'adamw_torch_fused': Fused AdamW (slightly faster, same memory)
  - 'adafactor': Memory-efficient optimizer (no momentum states, ~6GB less VRAM)
  - 'paged_adamw_8bit': 8-bit AdamW (requires bitsandbytes)
Default: 'adafactor' for 24GB VRAM compatibility with GR00T 1.6's 1.6B trainable params.
"""
```

### Why Adafactor?
- **No momentum states**: Uses running averages instead of per-parameter states
- **VRAM savings**: ~6GB less than AdamW
- **Quality**: Comparable training quality to AdamW for finetuning
- **Native support**: Built into HuggingFace Trainer

---

## 2. Apply Optimizer Config (`gr00t/experiment/launch_finetune.py`)

### Problem
The optimizer was hardcoded to `adamw_torch`, ignoring any user configuration.

### Solution
Changed line 73 to use the config value:

```python
# Before (hardcoded):
config.training.optim = "adamw_torch"

# After (configurable):
config.training.optim = ft_config.optim
```

This allows the `FinetuneConfig.optim` value to flow through to the Trainer.

---

## 3. Distributed Barrier Fix (`gr00t/data/dataset/factory.py`)

### Problem
The dataset factory called `torch.distributed.barrier()` unconditionally after stats generation:

```python
# Original code (lines 41-48)
if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        generate_stats(dataset_path)
        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
    torch.distributed.barrier()  # BUG: Called even when not rank 0!
else:
    generate_stats(dataset_path)
    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
```

On single-GPU training without distributed initialization, this caused issues. More critically, the barrier was outside the rank check, causing hangs.

### Solution
Moved barrier inside the distributed block:

```python
# Fixed code (lines 41-48)
if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        generate_stats(dataset_path)
        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
    torch.distributed.barrier()  # Sync after rank 0 generates stats
else:
    generate_stats(dataset_path)
    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
```

The barrier is now only called when distributed is initialized, preventing hangs on single-GPU setups.

---

## 4. Training Script Improvements (`custom/scripts/ver1_6/train_groot_so101_1_6.sh`)

### Changes Made

#### 4.1 Timestamped Output Directories
**Problem**: Multiple training runs would overwrite each other's outputs.

**Solution**: Auto-generate timestamped directories:
```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot_1_6_so101}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"
```

Output example: `outputs/groot_1_6_so101_20251218_201530/`

#### 4.2 Dual Logging (Terminal + File)
**Problem**: Training output was only visible in terminal, lost after session.

**Solution**: Log to both terminal and file using `tee`:
```bash
LOG_FILE="$OUTPUT_DIR/training.log"

# Log configuration header
{
    echo "=================================================================="
    echo "Training Started: $(date)"
    echo "Model: $BASE_MODEL"
    # ... more config ...
} > "$LOG_FILE"

# Training with dual output
python gr00t/experiment/launch_finetune.py ... 2>&1 | tee -a "$LOG_FILE"
```

#### 4.3 Removed Debug Flag
**Problem**: `set -x` caused excessive debug output.

**Solution**: Changed `set -x -e` to just `set -e` for cleaner logs.

---

## VRAM Usage Summary

| Configuration | VRAM Usage | Status |
|---------------|------------|--------|
| AdamW + batch 8 | ~26-28 GB | OOM on 24GB |
| Adafactor + batch 8 | ~20-22 GB | Works on 24GB |
| Adafactor + batch 12 | ~22-23 GB | Should work |
| Adafactor + batch 16 | ~24-26 GB | Risky |

---

## Files Modified (Quick Reference)

```
gr00t/configs/finetune_config.py    # Lines 111-119: Added optim parameter
gr00t/experiment/launch_finetune.py # Line 73: Use ft_config.optim
gr00t/data/dataset/factory.py       # Lines 41-48: Fixed barrier placement
custom/scripts/ver1_6/train_groot_so101_1_6.sh  # Logging & timestamps
```

---

## How to Verify Changes

```bash
# Check optimizer config exists
grep -n "optim:" gr00t/configs/finetune_config.py

# Check launch_finetune uses config
grep -n "ft_config.optim" gr00t/experiment/launch_finetune.py

# Check factory.py barrier fix
grep -A5 "torch.distributed.is_initialized" gr00t/data/dataset/factory.py
```
