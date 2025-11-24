# RTX 5090 GPU Compatibility Issue

**Date:** 2025-11-24
**Severity:** ⚠️ CRITICAL - May prevent training or cause errors

---

## Problem

Your system has an **NVIDIA GeForce RTX 5090 Laptop GPU** with CUDA compute capability **sm_120 (12.0)**.

However, PyTorch 2.5.1+cu124 in the `groot` environment was compiled for older architectures:
- sm_50 (Maxwell)
- sm_60 (Pascal)
- sm_70 (Volta)
- sm_75 (Turing)
- sm_80 (Ampere)
- sm_86 (Ampere)
- sm_90 (Hopper)

**Missing:** sm_120 (Blackwell - RTX 5090)

---

## What This Means

### Potential Issues

1. **Training may fail** - CUDA kernels won't run optimally
2. **Reduced performance** - Fallback to slower code paths
3. **Flash-attention incompatible** - May not work at all
4. **Random errors** - Unpredictable behavior during training

### Warning from PyTorch

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not
compatible with the current PyTorch installation.

The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70
sm_75 sm_80 sm_86 sm_90.
```

---

## Why This Happened

**RTX 5090 is VERY new:**
- Launched in Q4 2024/Q1 2025
- Based on Blackwell architecture (compute capability 12.0)
- PyTorch 2.5.1 was released **before** RTX 5090 existed
- Pre-compiled PyTorch binaries don't include sm_120 support

---

## Solutions

### Option 1: Use PyTorch Nightly (Recommended for RTX 5090)

PyTorch nightly builds include sm_120 support:

```bash
conda activate groot

# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install PyTorch nightly with CUDA 12.4
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify sm_120 support
python -c "import torch; print(torch.cuda.get_arch_list())"
# Should include 'sm_120'
```

**Pros:**
- ✅ Full RTX 5090 support
- ✅ Optimal performance
- ✅ Flash-attention compatibility

**Cons:**
- ⚠️ Nightly builds may have bugs
- ⚠️ GR00T tested with stable PyTorch 2.5.1
- ⚠️ May need to reinstall if Isaac-GR00T breaks

### Option 2: Try Training Anyway (Test First)

The warning may not prevent training - PyTorch might fall back to compatible code:

```bash
# Just proceed with current setup
bash custom/scripts/train_groot_mini_mvp.sh

# Watch for errors like:
# - "no kernel image is available for execution on the device"
# - CUDA errors during training
# - Crashes during model initialization
```

**Pros:**
- ✅ No changes needed
- ✅ Might work (PyTorch has fallbacks)

**Cons:**
- ❌ May crash during training
- ❌ Reduced performance (~30-50% slower)
- ❌ flash-attn likely won't work

### Option 3: Compile PyTorch from Source (Advanced)

Build PyTorch with sm_120 support:

```bash
# This takes 1-2 hours and requires significant disk space
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"
python setup.py install
```

**Pros:**
- ✅ Full control
- ✅ Guaranteed sm_120 support

**Cons:**
- ❌ Very time-consuming (1-2 hours compile time)
- ❌ Complex build process
- ❌ High disk space requirement (~50GB)

### Option 4: Downgrade GPU Driver (Not Recommended)

Force CUDA to report older compute capability - **DO NOT DO THIS**

---

## Recommended Approach

### Step 1: Test Current Setup First

```bash
cd ~/project/Isaac-GR00T
bash custom/scripts/train_groot_mini_mvp.sh
```

**Watch for:**
- Does it crash immediately?
- Does model loading work?
- Does training start?

### Step 2: If It Fails

Install PyTorch nightly:

```bash
conda activate groot
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

# Reinstall Isaac-GR00T dependencies
cd ~/project/Isaac-GR00T
pip install -e .[base]

# Reinstall flash-attn if needed
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### Step 3: Verify

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Supported architectures: {torch.cuda.get_arch_list()}')
print(f'sm_120 supported: {\"sm_120\" in str(torch.cuda.get_arch_list())}')
"
```

Should show `sm_120 supported: True`

---

## Impact on Training

### Without sm_120 Support

- **Model loading:** May work (CPU operations)
- **Data loading:** Should work (CPU operations)
- **Forward pass:** May fail (CUDA kernels)
- **Backward pass:** May fail (CUDA kernels)
- **Flash-attention:** Definitely won't work

### With sm_120 Support

- ✅ All operations work
- ✅ Full GPU utilization
- ✅ Optimal performance
- ✅ Flash-attention works

---

## Known Issues with RTX 5090

### 1. PyTorch Versions

- **PyTorch 2.5.1 and earlier:** NO sm_120 support
- **PyTorch 2.6.0 (upcoming):** Will include sm_120
- **PyTorch nightly:** Already has sm_120

### 2. CUDA Versions

- **CUDA 12.4+** recommended for RTX 5090
- Your system: CUDA 12.4 ✅
- PyTorch: cu124 ✅

### 3. Flash-attention

Flash-attention 2.7.1 may not have sm_120 kernels:
- May need to compile from source
- Or skip flash-attention (use `--no-tune_diffusion_model` flag)

---

## Monitoring During Training

Watch for these errors:

### CUDA Kernel Errors

```
RuntimeError: no kernel image is available for execution on the device
```
→ **Solution:** Install PyTorch nightly

### Performance Issues

```
Training is very slow (< 1 step/second)
```
→ **Cause:** Fallback to CPU or slow code path
→ **Solution:** Install PyTorch nightly

### Flash-attention Errors

```
AttributeError: module 'flash_attn' has no attribute 'flash_attn_func'
```
→ **Temporary solution:** Remove flash-attn, continue without it

---

## Quick Decision Tree

```
Can you afford to wait 10 minutes to test?
    ↓
   YES → Try current setup first
    |      ↓
    |    Does training start and run?
    |      ↓           ↓
    |     YES         NO
    |      ↓           ↓
    |   Continue    Install PyTorch nightly
    |                  (Option 1)
    |
   NO → Install PyTorch nightly immediately
         (Option 1)
```

---

## What We're Doing

**Current plan:**
1. ✅ Fixed modality.json format
2. ✅ Fixed training script validation
3. ⚠️ **Next:** Test if training works despite GPU warning
4. If fails → Install PyTorch nightly

**Why test first?**
- PyTorch may have fallbacks that work
- Avoid unnecessary environment changes
- Nightly builds may break other things

---

## Summary

**Problem:** RTX 5090 (sm_120) not supported by PyTorch 2.5.1

**Impact:** May prevent training or reduce performance

**Solution:** Install PyTorch nightly (has sm_120 support)

**Current Status:** Testing with current setup first

**Next Steps:**
1. Run mini-MVP test
2. If it fails with CUDA errors → Install PyTorch nightly
3. If it works → Continue with current setup

---

## References

- PyTorch CUDA compatibility: https://pytorch.org/get-started/locally/
- PyTorch nightly builds: https://pytorch.org/get-started/locally/#start-locally
- CUDA compute capabilities: https://developer.nvidia.com/cuda-gpus
- RTX 5090 specs: Compute capability 12.0 (sm_120)
