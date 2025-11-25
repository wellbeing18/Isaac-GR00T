# Follow-up Fixes and Improvements

**Date:** 2025-11-24
**Purpose:** Address remaining issues and improvements for GR00T training

---

## 1. Add Parquet Splitting to Conversion Script

### Issue
The `convert_lerobot_v3_to_groot.py` script converts metadata but doesn't split parquet files, which is **critical** for GR00T.

### Solution
Add a new function to split parquet files. Insert this into `/home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py`:

```python
def split_parquet_files(dataset_path: Path) -> None:
    """
    Split consolidated parquet file into per-episode files with reset indices.

    This is CRITICAL for GR00T - it expects:
    - Per-episode parquet files (episode_000.parquet, episode_001.parquet, etc.)
    - Each episode with 0-based indices (not global indices)

    Args:
        dataset_path: Path to dataset root
    """
    print("\n[BONUS] Splitting parquet files...")

    try:
        import pandas as pd
    except ImportError:
        print("  ⚠️  Warning: pandas not installed, skipping parquet splitting")
        print("     Install with: pip install pandas pyarrow")
        return

    data_dir = dataset_path / 'data' / 'chunk-000'
    consolidated_file = data_dir / 'file-000.parquet'

    if not consolidated_file.exists():
        print(f"  ℹ️  No consolidated file found at {consolidated_file}")
        print(f"     Checking if files are already split...")

        # Check if already split
        episode_files = list(data_dir.glob('episode_*.parquet'))
        if episode_files:
            print(f"  ✅ Found {len(episode_files)} per-episode parquet files")
            print(f"     Files already split - no action needed")
            return
        else:
            print(f"  ⚠️  No parquet files found!")
            return

    print(f"  📂 Reading: {consolidated_file.name}")
    df = pd.read_parquet(consolidated_file)

    total_rows = len(df)
    episodes = sorted(df['episode_index'].unique())

    print(f"     Total rows: {total_rows}")
    print(f"     Episodes: {len(episodes)}")

    # Split by episode
    split_count = 0
    for episode_idx in episodes:
        episode_df = df[df['episode_index'] == episode_idx].copy()

        # CRITICAL: Reset index to 0-based for each episode
        episode_df = episode_df.reset_index(drop=True)

        output_file = data_dir / f'episode_{episode_idx:03d}.parquet'
        episode_df.to_parquet(output_file)

        split_count += 1
        if split_count <= 3 or split_count == len(episodes):
            print(f"  ✅ Created: {output_file.name} ({len(episode_df)} frames)")

    if split_count > 3:
        print(f"     ... (+ {split_count - 3} more files)")

    # Optionally backup the original consolidated file
    backup_consolidated = data_dir / 'file-000.parquet.original'
    if not backup_consolidated.exists():
        shutil.copy2(consolidated_file, backup_consolidated)
        print(f"  📦 Backed up: {backup_consolidated.name}")

    print(f"\n  💡 TIP: You can delete {consolidated_file.name} to save space")
    print(f"          The per-episode files contain all the data")
```

### Integration
Add this call to the `main()` function after line 391:

```python
def main():
    # ... existing code ...

    try:
        # Run all conversions
        convert_modality_json(dataset_path, args.robot_type, args.dual_camera)
        fix_stats_json(dataset_path)
        generate_episodes_jsonl(dataset_path)
        generate_tasks_jsonl(dataset_path, args.task_description)
        split_parquet_files(dataset_path)  # <-- ADD THIS LINE

        # Validate
        success = validate_conversion(dataset_path)
        # ... rest of code ...
```

### Why This Matters
- GR00T's dataloader **requires** per-episode parquet files
- Each episode **must** have 0-based indices (reset_index is critical)
- Without this, you get `KeyError: 40` when trying to access frames

---

## 2. Suppress TorchVision Video Decoding Warnings

### Issue
Training logs are cluttered with hundreds of deprecation warnings:

```
/home/jrobot/anaconda3/envs/groot/lib/python3.10/site-packages/torchvision/io/_video_deprecation_warning.py:9: UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24...
```

### Solution A: Add Warning Filter to Training Scripts

Add this to the top of your training script (after imports):

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')
```

### Solution B: Set Environment Variable (Recommended)

Add this to all training scripts before running Python:

```bash
# At the top of train_groot_mini_mvp.sh, train_groot_so101_mvp.sh, etc.
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"
```

### Solution C: Modify LeRobot Dataset Code (Most Clean)

Edit `/home/jrobot/project/XLeRobot/lerobot/common/datasets/lerobot_dataset.py`:

At the top of the file (after imports), add:

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io._video_deprecation_warning')
```

### Recommended Approach

**Use Solution B (environment variable) for training scripts**:

Edit each training script and add after the shebang:

```bash
#!/bin/bash

# Suppress torchvision video deprecation warnings
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"

# Rest of script...
```

**Files to Update:**
- `/home/jrobot/project/XLeRobot/scripts/train_groot_mini_mvp.sh`
- `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_mini_mvp.sh`
- `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_so101_mvp.sh`
- `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_so101_full.sh`

---

## 3. Track LeRobot Changes to GitHub

### Problem
You modified files in the cloned lerobot repository:
- `/home/jrobot/project/XLeRobot/lerobot/common/policies/pi0/modeling_pi05.py`
- Possibly other files in the `lerobot/` directory

These changes are not tracked in your GitHub because `lerobot` is cloned from HuggingFace's repository.

### Solution: Fork and Track Changes

#### Step 1: Fork the LeRobot Repository

1. Go to https://github.com/huggingface/lerobot
2. Click "Fork" button (top right)
3. This creates your fork at: `https://github.com/YOUR_USERNAME/lerobot`

#### Step 2: Add Your Fork as Remote

```bash
cd /home/jrobot/project/XLeRobot

# Check current remotes
git remote -v
# Should show:
# origin  https://github.com/huggingface/lerobot.git (fetch)
# origin  https://github.com/huggingface/lerobot.git (push)

# Add your fork as a new remote called 'myfork'
git remote add myfork https://github.com/YOUR_USERNAME/lerobot.git

# Or if you use SSH:
git remote add myfork git@github.com:YOUR_USERNAME/lerobot.git

# Verify
git remote -v
# Should now show both 'origin' and 'myfork'
```

#### Step 3: Create a Branch for Your Changes

```bash
cd /home/jrobot/project/XLeRobot

# Create and switch to a new branch
git checkout -b custom-pi05-modifications

# Check what files you've modified
git status

# Check the diff
git diff lerobot/common/policies/pi0/modeling_pi05.py
```

#### Step 4: Commit Your Changes

```bash
# Stage the modified files
git add lerobot/common/policies/pi0/modeling_pi05.py
# Add any other modified files

# Create a commit with a descriptive message
git commit -m "Add LoRA support for Pi0.5 fine-tuning

- Modified modeling_pi05.py to support LoRA adapters
- Added LoRA configuration for PaliGemma and Action Expert
- Enables efficient fine-tuning on SO-101 robot
- Tested with rank=16, alpha=32, dropout=0.1"
```

#### Step 5: Push to Your Fork

```bash
# Push your branch to your fork
git push myfork custom-pi05-modifications

# If you want to set this as the default upstream:
git push --set-upstream myfork custom-pi05-modifications
```

#### Step 6: Keep Your Fork Updated

```bash
# Fetch latest changes from HuggingFace's lerobot
git fetch origin

# If you want to merge upstream changes into your branch:
git checkout custom-pi05-modifications
git merge origin/main

# Or rebase (cleaner history):
git rebase origin/main

# Push updated branch
git push myfork custom-pi05-modifications --force-with-lease
```

### Alternative: Track Changes Separately

If you don't want to fork the entire lerobot repo, you can track just your changes:

```bash
cd /home/jrobot/project/Isaac-GR00T/custom

# Create a patches directory
mkdir -p patches/lerobot

# Generate a patch file for your changes
cd /home/jrobot/project/XLeRobot
git diff lerobot/common/policies/pi0/modeling_pi05.py > \
    /home/jrobot/project/Isaac-GR00T/custom/patches/lerobot/pi05_lora_support.patch

# Commit this patch to your Isaac-GR00T repo
cd /home/jrobot/project/Isaac-GR00T
git add custom/patches/lerobot/pi05_lora_support.patch
git commit -m "Add patch for Pi0.5 LoRA support"
git push
```

Then to apply the patch on a new setup:

```bash
cd /home/jrobot/project/XLeRobot
git apply /home/jrobot/project/Isaac-GR00T/custom/patches/lerobot/pi05_lora_support.patch
```

### Recommended Approach

**Fork the repository** (first option) because:
1. ✅ Proper version control for all changes
2. ✅ Easy to sync with upstream updates
3. ✅ Can create pull requests if you want to contribute back
4. ✅ Full git history and blame
5. ✅ Better collaboration if working with others

### Documentation of Changes

Create a file documenting what you modified:

```bash
cat > /home/jrobot/project/Isaac-GR00T/custom/jdocs/LEROBOT_MODIFICATIONS.md <<'EOF'
# LeRobot Repository Modifications

**Fork:** https://github.com/YOUR_USERNAME/lerobot
**Branch:** custom-pi05-modifications
**Base:** huggingface/lerobot main branch

## Modified Files

### lerobot/common/policies/pi0/modeling_pi05.py

**Changes:**
- Added LoRA adapter support for Pi0.5 model
- Added `use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout` parameters
- Wrapped PaliGemma and Action Expert (Gemma) with LoRA adapters
- Enables efficient fine-tuning with only 0.7-1.6% trainable parameters

**Lines Modified:** ~500-650

**Why:** Pi0.5 base implementation doesn't support LoRA, which is essential for fine-tuning on consumer hardware (RTX 5090 24GB).

**Testing:** Validated with SO-101 dataset, 100-step mini-MVP training successful.

## Setup Instructions

```bash
# Clone your fork instead of upstream
git clone https://github.com/YOUR_USERNAME/lerobot.git XLeRobot
cd XLeRobot
git checkout custom-pi05-modifications

# Continue with normal setup...
```

## Syncing with Upstream

```bash
# Add upstream remote (one-time)
git remote add upstream https://github.com/huggingface/lerobot.git

# Fetch and merge upstream changes
git fetch upstream
git checkout custom-pi05-modifications
git rebase upstream/main

# Resolve conflicts if any, then push
git push myfork custom-pi05-modifications --force-with-lease
```
EOF
```

---

## Summary of Action Items

### Immediate Updates:

1. **✅ Update `convert_lerobot_v3_to_groot.py`**
   - Add `split_parquet_files()` function
   - Call it in `main()` after metadata conversions

2. **✅ Suppress video warnings in training scripts**
   - Add `export PYTHONWARNINGS="ignore::UserWarning:torchvision.io"` to all 4 training scripts

3. **✅ Fork lerobot and track your changes**
   - Fork https://github.com/huggingface/lerobot
   - Add your fork as remote
   - Commit and push changes to your fork
   - Create `LEROBOT_MODIFICATIONS.md` documentation

### Files to Modify:

1. `/home/jrobot/project/Isaac-GR00T/custom/scripts/convert_lerobot_v3_to_groot.py`
2. `/home/jrobot/project/XLeRobot/scripts/train_groot_mini_mvp.sh`
3. `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_mini_mvp.sh`
4. `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_so101_mvp.sh`
5. `/home/jrobot/project/Isaac-GR00T/custom/scripts/train_groot_so101_full.sh`

### Git Operations:

```bash
# 1. Fork lerobot on GitHub first

# 2. Add fork as remote
cd /home/jrobot/project/XLeRobot
git remote add myfork git@github.com:YOUR_USERNAME/lerobot.git

# 3. Create branch and commit
git checkout -b custom-pi05-modifications
git add lerobot/common/policies/pi0/modeling_pi05.py
git commit -m "Add LoRA support for Pi0.5 fine-tuning"
git push myfork custom-pi05-modifications

# 4. Track in Isaac-GR00T
cd /home/jrobot/project/Isaac-GR00T
# (Create LEROBOT_MODIFICATIONS.md as shown above)
git add custom/jdocs/LEROBOT_MODIFICATIONS.md
git commit -m "Document LeRobot modifications for Pi0.5 LoRA"
git push
```

---

**All these improvements are optional but highly recommended for a clean, maintainable setup!**
