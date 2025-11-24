# GR00T Environment Setup Recovery Guide

**Last Updated:** 2025-11-24
**Status:** Setup incomplete - flash-attn installation failed

---

## 🔧 Quick Fix (Updated)

**Problem:** `CUDA_HOME environment variable is not set` when installing flash-attn

**Easiest Solution:**
```bash
conda activate groot

# Install CUDA toolkit with nvcc via conda
conda install -c nvidia cuda-toolkit=12.4 -y

# Verify nvcc is available
which nvcc

# Install flash-attn
pip install --no-build-isolation flash-attn==2.7.1.post4

# Verify
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
```

**Time Required:** 10-15 minutes (including compilation)

**Alternative:** Skip flash-attn for now and run mini-MVP test → See Option 2 below

---

## Current Setup Status

### ✅ Completed Steps

1. **Isaac-GR00T Repository**
   - ✅ Cloned to: `~/project/Isaac-GR00T`
   - ✅ Location verified and accessible

2. **Conda Environment**
   - ✅ Environment created: `groot`
   - ✅ Python 3.10 installed
   - ✅ Activated successfully

3. **Base Dependencies**
   - ✅ PyTorch 2.5.1 installed
   - ✅ Transformers 4.51.3 installed
   - ✅ Isaac-GR00T base package installed (editable mode)

### ❌ Incomplete Step

4. **flash-attn Installation** - FAILED
   - ❌ Error: `CUDA_HOME environment variable is not set`
   - ❌ Error: `nvcc was not found`
   - Status: Installation script exited before completing

---

## Error Details

### Error Message

```
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
fatal: not a git repository (or any of the parent directories): .git
nvcc was not found. Are you sure your environment has nvcc available?
```

### What This Means

- **CUDA_HOME not set:** The flash-attn compilation needs to know where CUDA is installed
- **nvcc not found:** NVIDIA's CUDA compiler (nvcc) is not in system PATH
- **Consequence:** flash-attn cannot be compiled from source

### Why flash-attn Matters

- **Performance:** flash-attn provides 2-4x faster attention computation
- **Memory:** Reduces VRAM usage by ~20-30%
- **Training Speed:** Significantly speeds up GR00T training
- **Required?** Unclear - may be optional for basic training

---

## Recovery Options

Choose the option that best fits your situation:

### Option 1: Fix CUDA_HOME and Install flash-attn (Recommended)

**Best For:** Production use, full training runs

#### Step 1: Locate CUDA Installation

```bash
# Activate groot environment first
conda activate groot

# Check if PyTorch knows where CUDA is (CORRECTED command)
python -c "import torch.utils.cpp_extension as cpp_ext; print(f'CUDA_HOME: {cpp_ext.CUDA_HOME}')"

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# If CUDA_HOME is None, search for CUDA manually
# Option 1: Search system-wide
find /usr -name "nvcc" 2>/dev/null
find /opt -name "nvcc" 2>/dev/null
find /usr/local -name "nvcc" 2>/dev/null

# Option 2: Check conda environment for CUDA toolkit
conda list | grep cuda
ls $CONDA_PREFIX/bin/ | grep nvcc

# Option 3: Check for cudatoolkit package
conda list cudatoolkit
```

#### Step 2: Install CUDA Toolkit (If Not Found)

If the searches above found nothing, you have two options:

**Option A: Install via Conda (Recommended - Easier)**

```bash
conda activate groot

# Install CUDA toolkit with nvcc into conda environment
conda install -c nvidia cuda-toolkit=12.4 -y

# Verify installation
which nvcc
# Should show: ~/anaconda3/envs/groot/bin/nvcc

nvcc --version
# Should show: Cuda compilation tools, release 12.4
```

**Option B: Install System-Wide CUDA Toolkit**

```bash
# Check nvidia-smi for supported CUDA version
nvidia-smi
# Look for "CUDA Version: XX.X" in top right

# Download matching CUDA toolkit (example for 12.4)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install
sudo sh cuda_12.4.0_550.54.14_linux.run

# Follow prompts:
# - Accept license
# - Install toolkit + samples
# - Let it create /usr/local/cuda-12.4 symlink
```

#### Step 3: Set CUDA_HOME

**If you used Option A (conda):**

```bash
# CUDA_HOME should auto-set to conda environment
conda activate groot
python -c "import torch.utils.cpp_extension as cpp_ext; print(f'CUDA_HOME: {cpp_ext.CUDA_HOME}')"
# Should show: ~/anaconda3/envs/groot or similar

# If still None, set manually:
export CUDA_HOME=$CONDA_PREFIX
echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
```

**If you used Option B (system-wide):**

```bash
# Set CUDA_HOME to system installation
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Replace `/usr/local/cuda-12.4` with your actual CUDA path!**

#### Step 4: Verify CUDA Setup

```bash
# Check nvcc is found
which nvcc
# Should return: ~/anaconda3/envs/groot/bin/nvcc (conda install)
#           or: /usr/local/cuda-12.4/bin/nvcc (system install)

# Check CUDA version
nvcc --version
# Should match your PyTorch CUDA version (12.x)

# Verify in Python (CORRECTED command)
conda activate groot
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch.utils.cpp_extension as cpp_ext; print(f'CUDA_HOME: {cpp_ext.CUDA_HOME}')"
# CUDA_HOME should NOT be None anymore!
```

#### Step 5: Install flash-attn

```bash
conda activate groot
cd ~/project/Isaac-GR00T

# Install flash-attn (takes 5-10 minutes to compile)
pip install --no-build-isolation flash-attn==2.7.1.post4

# Verify installation
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"
```

#### Step 6: Verify Complete Setup

```bash
# Run full environment check
conda activate groot
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
python -c "import gr00t; print('GR00T: OK')"
```

**✅ If all three imports succeed, your setup is complete!**

---

### Option 2: Skip flash-attn for Now (Quick Start)

**Best For:** Testing, validation, mini-MVP run

#### When to Use This

- You want to run mini-MVP test immediately
- You're unsure about CUDA installation
- You can test if GR00T works without flash-attn
- You'll install flash-attn later for full training

#### What You'll Lose

- Training will be 2-4x slower
- VRAM usage will be higher (~22-24GB vs ~18-20GB)
- May not fit batch size 8 (drop to 4 if needed)

#### How to Proceed

**No additional installation needed!** Your environment is already functional:

```bash
# Verify what you have
conda activate groot
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gr00t; print('GR00T: OK')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

#### Adjust Mini-MVP Script

If you get CUDA OOM errors during training without flash-attn:

```bash
# Edit mini-MVP script
nano /home/jrobot/project/XLeRobot/scripts/train_groot_mini_mvp.sh

# Change line 205:
# FROM: --batch-size 4 \
# TO:   --batch-size 2 \
```

---

## Next Steps After Setup Complete

### 1. Add ISAAC_GROOT_ROOT to Environment (Recommended)

```bash
echo 'export ISAAC_GROOT_ROOT=$HOME/project/Isaac-GR00T' >> ~/.bashrc
source ~/.bashrc
```

### 2. Run Mini-MVP Test

```bash
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

**What This Does:**
- Validates GR00T pipeline with your 10 episodes
- Creates modality.json if needed
- Runs 100 training steps (~5-10 minutes)
- Generates inspection report

**Expected Duration:**
- With flash-attn: 5-8 minutes
- Without flash-attn: 10-15 minutes

### 3. Review Results

```bash
# Check inspection report
cat outputs/groot_mini_mvp_test/mini_mvp_inspection_report.txt

# Check training log
less outputs/groot_mini_mvp_test/mini_mvp_training.log

# Look for:
# - ✅ Training completed 100 steps
# - ✅ Loss values (not NaN)
# - ✅ No CUDA OOM errors
# - ✅ Checkpoint saved
```

### 4. What to Share for Validation

Share these files:
1. `outputs/groot_mini_mvp_test/mini_mvp_inspection_report.txt`
2. Last 50 lines of `outputs/groot_mini_mvp_test/mini_mvp_training.log`
3. Any error messages

---

## Common Issues and Solutions

### Issue 1: "ImportError: No module named 'gr00t'"

**Solution:**
```bash
conda activate groot
cd ~/project/Isaac-GR00T
pip install -e .[base]
```

### Issue 2: "CUDA out of memory" During Training

**Solution 1:** Reduce batch size
```bash
# Edit script: train_groot_mini_mvp.sh
# Change: --batch-size 4
# To:     --batch-size 2
```

**Solution 2:** Install flash-attn (reduces VRAM by ~20%)
```bash
# Follow Option 1 above
```

### Issue 3: "modality.json not found"

**Solution:** The mini-MVP script auto-creates this, but if you need to create manually:

```bash
# The script creates this automatically at:
# /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/modality.json

# If you need to verify it exists:
ls -la /home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/modality.json
```

### Issue 4: "Dataset v3 format not compatible"

**Warning:** You may see a warning about LeRobot v3 format.

**For Mini-MVP:** Continue anyway - script handles this

**For Full Training:** If mini-MVP fails with format errors:
```bash
cd /home/jrobot/project/lerobot
conda activate lerobot

python scripts/convert_dataset_v3_to_v2.py \
  --input /home/jrobot/project/XLeRobot/jdocs/top_level/datasets \
  --output /home/jrobot/project/XLeRobot/jdocs/top_level/datasets_v2

# Then update DATASET_PATH in training scripts to point to datasets_v2
```

---

## Environment Management Reference

### Quick Commands

```bash
# List all environments
conda env list

# Activate groot environment
conda activate groot

# Check active environment
echo $CONDA_DEFAULT_ENV

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import gr00t; print('GR00T OK')"
python -c "import flash_attn; print('flash-attn OK')"  # May fail if not installed

# Check GPU
nvidia-smi

# Check Python location (should be in groot env)
which python
```

### Environment Locations

```
~/anaconda3/envs/
├── lerobot/          ← For Pi0.5, LeRobot (PyTorch 2.9.0)
└── groot/            ← For GR00T (PyTorch 2.5.1)

~/project/
├── lerobot/          ← LeRobot codebase
├── Isaac-GR00T/      ← GR00T codebase (this repo)
└── XLeRobot/         ← Your project
```

---

## Full Project Documentation

For complete guides, see:

1. **Environment Management:**
   - `/home/jrobot/project/XLeRobot/jdocs/top_level/ENVIRONMENT_MANAGEMENT.md`
   - `/home/jrobot/project/XLeRobot/jdocs/top_level/ENVIRONMENT_SETUP_SUMMARY.md`

2. **LoRA Training Guide:**
   - `/home/jrobot/project/XLeRobot/jdocs/top_level/lora/COMPLETE_LORA_GUIDE.md`

3. **Training Scripts:**
   - Mini-MVP: `/home/jrobot/project/XLeRobot/scripts/train_groot_mini_mvp.sh`
   - MVP: `/home/jrobot/project/XLeRobot/scripts/train_groot_so101_mvp.sh`
   - Full: `/home/jrobot/project/XLeRobot/scripts/train_groot_so101_full.sh`

---

## Decision Tree

```
Start Here
    ↓
Do you have time to fix CUDA setup? (10-30 min)
    ↓                                    ↓
   YES                                  NO
    ↓                                    ↓
Follow Option 1                      Follow Option 2
(Install flash-attn)                 (Skip for now)
    ↓                                    ↓
    └────────────┬────────────────────────┘
                 ↓
         Run Mini-MVP Test
                 ↓
         Review Results
                 ↓
    Did it work?
    ↓          ↓
   YES        NO
    ↓          ↓
Collect    Share logs
more data  for debug
    ↓
Run Full MVP
(500 steps)
```

---

## Quick Start (TL;DR)

**If you want to skip flash-attn and run mini-MVP now:**

```bash
conda activate groot
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

**If training fails with OOM error:**
```bash
# Edit script to reduce batch size
nano scripts/train_groot_mini_mvp.sh
# Change: --batch-size 4
# To:     --batch-size 2
```

**If you want to properly install flash-attn first:**

```bash
conda activate groot

# Find CUDA (CORRECTED command)
python -c "import torch.utils.cpp_extension as cpp_ext; print(f'CUDA_HOME: {cpp_ext.CUDA_HOME}')"

# If CUDA_HOME is None, install CUDA toolkit via conda (easiest):
conda install -c nvidia cuda-toolkit=12.4 -y
which nvcc  # Verify nvcc is now available

# Install flash-attn
pip install --no-build-isolation flash-attn==2.7.1.post4

# Verify
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"

# Then run mini-MVP
cd /home/jrobot/project/XLeRobot
bash scripts/train_groot_mini_mvp.sh
```

---

## Getting Help

If you're stuck:

1. **Share your error logs:**
   - Copy error messages
   - Include output of environment checks

2. **Check these files:**
   - Setup script: `/home/jrobot/project/XLeRobot/scripts/setup_groot_env.sh`
   - Environment guide: `/home/jrobot/project/XLeRobot/jdocs/top_level/ENVIRONMENT_MANAGEMENT.md`

3. **Useful diagnostic commands:**
   ```bash
   conda env list
   which python
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   echo $CUDA_HOME
   which nvcc
   ```

---

## Summary

**Current Status:**
- ✅ Isaac-GR00T cloned
- ✅ `groot` conda environment created
- ✅ Base dependencies installed
- ❌ flash-attn not installed (optional but recommended)

**Recommended Path:**
1. Try Option 2 (skip flash-attn) to run mini-MVP now
2. If it works: collect more data, worry about flash-attn later
3. If you get OOM errors: install flash-attn (Option 1)

**Goal:** Validate pipeline with mini-MVP test, then proceed to full MVP with 50 episodes.

Good luck! 🚀
