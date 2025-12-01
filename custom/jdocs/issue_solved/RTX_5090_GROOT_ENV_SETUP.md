# RTX 5090 GR00T Environment Setup Guide

**Date:** 2025-11-29
**Status:** SOLVED
**GPU:** NVIDIA GeForce RTX 5090 Laptop GPU (Blackwell, sm_120)

---

## Overview

This document records all environment issues encountered when setting up GR00T training on RTX 5090 (Blackwell architecture) and their solutions.

---

## Issue 1: PyTorch sm_120 (Blackwell) Support

### Problem
RTX 5090 uses Blackwell architecture with compute capability sm_120 (12.0). Standard PyTorch releases (2.5.x) don't include sm_120 support.

**Error:**
```
CUDA error: no kernel image is available for execution on the device
```

**Verification:**
```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
# Missing sm_120 means RTX 5090 not supported
```

### Solution
Use PyTorch 2.9.1+cu128 or later which includes sm_120 support.

```bash
conda activate groot

# Install PyTorch 2.9.1 with CUDA 12.8
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Verify sm_120 support
python -c "import torch; print(torch.cuda.get_arch_list())"
# Should show: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

---

## Issue 2: flash-attn Compatibility

### Problem
Pre-built flash-attn wheels are not available for torch 2.9.1+cu128. Attempting to use older wheels results in ABI incompatibility.

**Error:**
```
undefined symbol: _ZNK3c106SymInt6sym_neERKS0_
```

### Solution
Build flash-attn from source for sm_120 only (faster build, ~60 minutes).

**Step 1: Install matching CUDA nvcc**
```bash
conda activate groot

# Install CUDA 12.8 nvcc to match torch's CUDA version
conda install -c nvidia cuda-nvcc=12.8.93 -y

# Verify nvcc version matches torch CUDA
nvcc --version  # Should show: Cuda compilation tools, release 12.8
python -c "import torch; print(f'torch CUDA: {torch.version.cuda}')"  # Should show: 12.8
```

**Step 2: Build flash-attn for Blackwell only**
```bash
# Build for sm_120 only (saves ~3x build time)
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
```

**Important:** The build takes ~60 minutes. The wheel is cached at `~/.cache/pip/wheels/` for future reinstalls.

---

## Issue 3: TensorFlow CUDA Conflict

### Problem
TensorFlow 2.15.0 was installed and conflicted with PyTorch CUDA libraries, causing `std::bad_alloc` errors during import.

**Error:**
```
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
```

**With TensorFlow warnings:**
```
Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

### Solution
Remove TensorFlow (not needed for GR00T training).

```bash
conda activate groot

# Check if TensorFlow is installed
pip show tensorflow

# Remove TensorFlow and related packages
pip uninstall tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem -y

# Verify removal
pip list | grep -i tensor
# Should only show: safetensors, tensorboard (tensorboard is OK)
```

---

## Issue 4: torchcodec Version Incompatibility

### Problem
torchcodec 0.5 (installed by lerobot) is incompatible with torch 2.9.1, causing `std::bad_alloc` during import.

**Error:**
```
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
```

(Crash happens during `import torchcodec`)

### Solution
Upgrade torchcodec to 0.8.1.

```bash
conda activate groot

# Upgrade torchcodec
pip install torchcodec==0.8.1

# Verify
python -c "import torchcodec; print('torchcodec OK')"
```

**Note:** This will show lerobot dependency conflicts (torch<2.8.0, torchcodec<0.6.0) but GR00T training works correctly.

---

## Issue 5: torchvision Deprecation Warnings

### Problem
torchvision 0.22+ shows excessive deprecation warnings for video decoding, cluttering training logs.

**Warning:**
```
UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24. We recommend that you migrate to TorchCodec...
```

### Solution
Add warning filters in training scripts:

```bash
# In bash script, add before running Python:
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io,ignore::UserWarning:torchvision"

# When running Python:
python -W "ignore::UserWarning:torchvision" scripts/gr00t_finetune.py ...

# Or filter output:
python scripts/gr00t_finetune.py 2>&1 | grep -v "torchvision/io/_video_deprecation_warning"
```

---

## Complete Environment Setup

### Working Configuration (as of 2025-11-29)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10.x | Required by GR00T |
| torch | 2.9.1+cu128 | Has sm_120 support |
| torchvision | 0.24.1+cu128 | Matches torch |
| flash-attn | 2.8.3 | Built for sm_120 |
| torchcodec | 0.8.1 | Compatible with torch 2.9 |
| cuda-nvcc | 12.8.93 | From conda, matches torch |
| transformers | 4.51.3 | GR00T requirement |

### Full Setup Script

```bash
#!/bin/bash
# Complete GR00T environment setup for RTX 5090

# Create and activate environment
conda create -n groot python=3.10 -y
conda activate groot

# Install CUDA nvcc 12.8 from conda
conda install -c nvidia cuda-nvcc=12.8.93 -y

# Install PyTorch 2.9.1 with sm_120 support
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install GR00T base
cd ~/project/Isaac-GR00T
pip install -e .[base]

# Build flash-attn for Blackwell (takes ~60 minutes)
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation

# Upgrade torchcodec for compatibility
pip install torchcodec==0.8.1

# Remove TensorFlow if present (causes CUDA conflicts)
pip uninstall tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem -y 2>/dev/null || true

# Verify setup
echo "=== Environment Verification ==="
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'sm_120 support: {\"sm_120\" in torch.cuda.get_arch_list()}')"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
python -c "from gr00t.model import Gr00tPolicy; print('GR00T: OK')"
```

### Verification Commands

```bash
# Check all components
conda activate groot

# PyTorch and CUDA
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'CUDA arch list: {torch.cuda.get_arch_list()}')"

# flash-attn
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"

# GR00T imports
python -c "from gr00t.model import Gr00tPolicy; print('Gr00tPolicy: OK')"
python -c "from gr00t.experiment.runner import TrainRunner; print('TrainRunner: OK')"

# nvcc version
nvcc --version
```

---

## Troubleshooting

### "no kernel image is available for execution on the device"
- **Cause:** PyTorch doesn't have sm_120 support
- **Fix:** Install torch 2.9.1+cu128 or later

### "std::bad_alloc" on import
- **Cause 1:** TensorFlow CUDA conflict → Uninstall TensorFlow
- **Cause 2:** torchcodec incompatibility → Upgrade to 0.8.1

### "undefined symbol" when importing flash-attn
- **Cause:** ABI mismatch between flash-attn wheel and torch
- **Fix:** Build flash-attn from source with matching CUDA

### flash-attn build fails with "CUDA version mismatch"
- **Cause:** nvcc version doesn't match torch's CUDA version
- **Fix:** Install matching cuda-nvcc from conda (e.g., `cuda-nvcc=12.8.93` for torch+cu128)

### Training hangs or runs very slowly
- **Cause:** Missing flash-attn (falls back to slow attention)
- **Fix:** Ensure flash-attn is properly installed and importable

---

## Related Files

- Training script: `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_mini_mvp.sh`
- Previous RTX 5090 doc: `/home/jrobot/project/Isaac-GR00T/custom/jdocs/issue_solved/RTX_5090_COMPATIBILITY_ISSUE.md`
- Setup recovery guide: `/home/jrobot/project/Isaac-GR00T/custom/jdocs/lora/SETUP_RECOVERY_GUIDE.md`

---

## Changelog

- **2025-11-29:** Initial documentation of full RTX 5090 setup with torch 2.9.1+cu128
