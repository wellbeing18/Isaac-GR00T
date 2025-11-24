# Git Workflow Guide for Isaac-GR00T Customization

**Last Updated:** 2025-11-24

---

## Overview

This guide explains how to manage your custom scripts and documentation alongside the upstream Isaac-GR00T repository.

### Goals

1. ✅ Keep custom scripts and docs in Isaac-GR00T directory
2. ✅ Track your changes with Git
3. ✅ Push your changes to your own GitHub repository
4. ✅ Pull upstream updates from NVIDIA's Isaac-GR00T
5. ✅ Avoid conflicts between your work and upstream changes

---

## Git Workflow Strategy

We'll use a **fork + custom directory** approach:

```
Isaac-GR00T/
├── scripts/              ← NVIDIA's original scripts (don't modify)
├── gr00t/                ← NVIDIA's code (don't modify)
├── custom/               ← YOUR custom scripts (safe from conflicts)
│   ├── scripts/
│   │   ├── train_groot_mini_mvp.sh
│   │   ├── train_groot_so101_mvp.sh
│   │   ├── train_groot_so101_full.sh
│   │   └── setup_groot_env.sh
│   └── README.md
├── jdocs/                ← YOUR documentation (safe from conflicts)
│   ├── SETUP_RECOVERY_GUIDE.md
│   ├── GIT_WORKFLOW_GUIDE.md
│   └── ...
└── .git/
```

**Why This Works:**
- `custom/` and `jdocs/` directories won't exist in upstream NVIDIA repo
- No merge conflicts when pulling from upstream
- Clear separation between NVIDIA code and your customizations
- Easy to share your setup with others

---

## Setup Instructions

### Step 1: Fork Isaac-GR00T on GitHub

1. Go to: https://github.com/NVIDIA/Isaac-GR00T
2. Click "Fork" button (top right)
3. Create fork in your GitHub account (e.g., `yourusername/Isaac-GR00T`)

### Step 2: Configure Git Remotes

```bash
cd ~/project/Isaac-GR00T

# Verify current remote (should be NVIDIA's repo)
git remote -v
# Output: origin  https://github.com/NVIDIA/Isaac-GR00T (fetch)
#         origin  https://github.com/NVIDIA/Isaac-GR00T (push)

# Rename NVIDIA's remote to 'upstream'
git remote rename origin upstream

# Add YOUR fork as 'origin'
git remote add origin https://github.com/YOUR_USERNAME/Isaac-GR00T.git

# Verify remotes
git remote -v
# Output:
#   origin    https://github.com/YOUR_USERNAME/Isaac-GR00T.git (fetch)
#   origin    https://github.com/YOUR_USERNAME/Isaac-GR00T.git (push)
#   upstream  https://github.com/NVIDIA/Isaac-GR00T (fetch)
#   upstream  https://github.com/NVIDIA/Isaac-GR00T (push)
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

### Step 3: Create Custom Directory Structure

```bash
cd ~/project/Isaac-GR00T

# Create custom directory for your scripts
mkdir -p custom/scripts

# jdocs directory already exists (we created it earlier)
```

### Step 4: Copy GR00T Scripts to Custom Directory

```bash
# Copy scripts from XLeRobot to Isaac-GR00T/custom/
cp /home/jrobot/project/XLeRobot/scripts/train_groot_mini_mvp.sh custom/scripts/
cp /home/jrobot/project/XLeRobot/scripts/train_groot_so101_mvp.sh custom/scripts/
cp /home/jrobot/project/XLeRobot/scripts/train_groot_so101_full.sh custom/scripts/
cp /home/jrobot/project/XLeRobot/scripts/setup_groot_env.sh custom/scripts/

# Make scripts executable
chmod +x custom/scripts/*.sh
```

### Step 5: Create Custom README

```bash
cat > custom/README.md << 'EOF'
# Custom Scripts for Isaac-GR00T

This directory contains custom training scripts for SO-101 left arm.

## Scripts

- `train_groot_mini_mvp.sh` - Ultra-quick validation (100 steps, 5-10 min)
- `train_groot_so101_mvp.sh` - MVP training (500 steps, 1-2 hours)
- `train_groot_so101_full.sh` - Full training (10,000 steps, 6-8 hours)
- `setup_groot_env.sh` - One-time environment setup

## Usage

```bash
cd ~/project/Isaac-GR00T

# Run mini-MVP test
bash custom/scripts/train_groot_mini_mvp.sh

# Run full MVP
bash custom/scripts/train_groot_so101_mvp.sh

# Run full training
bash custom/scripts/train_groot_so101_full.sh
```

## Documentation

See `/jdocs` directory for:
- Setup recovery guide
- Git workflow guide
- Training documentation
EOF
```

### Step 6: Update .gitignore (Preserve Custom Work)

```bash
cd ~/project/Isaac-GR00T

# Check if custom/ and jdocs/ are already ignored
cat .gitignore | grep -E "(custom|jdocs)"

# If not found, add them to a custom .gitignore section
cat >> .gitignore << 'EOF'

# Custom user scripts and documentation
# (Remove these lines if you want to commit custom/ and jdocs/)
# custom/
# jdocs/
EOF
```

**Note:** We're NOT actually ignoring them - we WANT to commit them. The commented lines are just for reference.

### Step 7: Commit Your Custom Work

```bash
cd ~/project/Isaac-GR00T

# Check what will be committed
git status

# Add your custom files
git add custom/
git add jdocs/

# Commit with descriptive message
git commit -m "Add custom SO-101 training scripts and documentation

- Add mini-MVP, MVP, and full training scripts
- Add setup recovery guide
- Add Git workflow documentation
- All scripts configured for SO-101 left arm dual-camera setup"

# Create a custom branch (optional but recommended)
git checkout -b custom-so101-setup

# Or stay on main branch if you prefer
```

### Step 8: Push to Your Fork

```bash
cd ~/project/Isaac-GR00T

# Push to YOUR fork
git push origin main
# Or if you created a branch:
# git push origin custom-so101-setup
```

---

## Daily Workflow

### Making Changes to Your Scripts

```bash
cd ~/project/Isaac-GR00T

# Edit scripts as needed
nano custom/scripts/train_groot_mini_mvp.sh

# Commit changes
git add custom/
git commit -m "Update mini-MVP batch size for better VRAM usage"

# Push to your fork
git push origin main
```

### Pulling Upstream Updates from NVIDIA

```bash
cd ~/project/Isaac-GR00T

# Fetch latest from NVIDIA
git fetch upstream

# Merge upstream changes into your branch
git merge upstream/main

# If there are conflicts (unlikely with custom/ directory approach):
# - Resolve conflicts
# - git add <resolved-files>
# - git commit

# Push merged changes to your fork
git push origin main
```

### Regular Update Routine

Run this weekly or before major training runs:

```bash
cd ~/project/Isaac-GR00T

# Pull upstream updates
git fetch upstream
git merge upstream/main

# Push to your fork
git push origin main

# Verify environment still works
conda activate groot
python -c "import gr00t; print('GR00T OK')"
```

---

## Script Path Updates

Since scripts are now in `Isaac-GR00T/custom/scripts/`, we need to update paths.

### No Changes Needed!

The scripts use `$ISAAC_GROOT_ROOT` and absolute paths, so they work from any location:

```bash
# These all work:
cd ~/project/Isaac-GR00T && bash custom/scripts/train_groot_mini_mvp.sh
cd ~/project/XLeRobot && bash ~/project/Isaac-GR00T/custom/scripts/train_groot_mini_mvp.sh
cd ~ && bash project/Isaac-GR00T/custom/scripts/train_groot_mini_mvp.sh
```

### Add Convenience Aliases (Optional)

```bash
# Add to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Isaac-GR00T aliases
alias groot_mini='cd ~/project/Isaac-GR00T && bash custom/scripts/train_groot_mini_mvp.sh'
alias groot_mvp='cd ~/project/Isaac-GR00T && bash custom/scripts/train_groot_so101_mvp.sh'
alias groot_full='cd ~/project/Isaac-GR00T && bash custom/scripts/train_groot_so101_full.sh'
alias groot_dir='cd ~/project/Isaac-GR00T'
EOF

# Apply changes
source ~/.bashrc
```

Then just type:
```bash
groot_mini   # Runs mini-MVP from anywhere
groot_mvp    # Runs MVP from anywhere
groot_full   # Runs full training from anywhere
groot_dir    # Jump to Isaac-GR00T directory
```

---

## Sharing Your Setup

### Option 1: Share Your Fork URL

Others can clone your fork directly:

```bash
git clone https://github.com/YOUR_USERNAME/Isaac-GR00T.git
cd Isaac-GR00T
conda create -n groot python=3.10
conda activate groot
pip install -e .[base]

# Run your scripts
bash custom/scripts/train_groot_mini_mvp.sh
```

### Option 2: Create Installation Script

```bash
cat > custom/scripts/quick_install.sh << 'EOF'
#!/bin/bash
# Quick installation script for custom Isaac-GR00T setup

set -e

echo "Installing custom Isaac-GR00T setup..."

# Clone your fork
cd ~/project
git clone https://github.com/YOUR_USERNAME/Isaac-GR00T.git
cd Isaac-GR00T

# Create conda environment
conda create -n groot python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate groot

# Install dependencies
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# Set environment variable
echo 'export ISAAC_GROOT_ROOT=$HOME/project/Isaac-GR00T' >> ~/.bashrc

echo "Setup complete! Run: bash custom/scripts/train_groot_mini_mvp.sh"
EOF

chmod +x custom/scripts/quick_install.sh
```

---

## Troubleshooting

### "Permission denied" when pushing

You may need to authenticate:

```bash
# Option 1: Use SSH (recommended)
git remote set-url origin git@github.com:YOUR_USERNAME/Isaac-GR00T.git

# Option 2: Use GitHub CLI
gh auth login

# Option 3: Use Personal Access Token
# Set up token at: https://github.com/settings/tokens
```

### Merge conflicts with upstream

```bash
# Check what's conflicting
git status

# If conflicts in your custom/ or jdocs/:
# This shouldn't happen since upstream doesn't have these directories
# Just keep your version

# If conflicts in NVIDIA's files (scripts/, gr00t/):
# Keep upstream version (you shouldn't modify these)
git checkout --theirs <file>
git add <file>
git commit
```

### Accidentally modified NVIDIA's files

```bash
# Reset to upstream version
git checkout upstream/main -- scripts/gr00t_finetune.py

# Or reset entire directory
git checkout upstream/main -- scripts/
```

### Want to start fresh

```bash
# Backup your custom work
cp -r custom/ ~/custom_backup/
cp -r jdocs/ ~/jdocs_backup/

# Reset repository
cd ~/project/Isaac-GR00T
git fetch upstream
git reset --hard upstream/main

# Restore your custom work
cp -r ~/custom_backup/* custom/
cp -r ~/jdocs_backup/* jdocs/

# Recommit
git add custom/ jdocs/
git commit -m "Restore custom scripts after reset"
```

---

## Best Practices

### 1. Never Modify Upstream Files Directly

❌ **Don't:**
```bash
nano scripts/gr00t_finetune.py  # NVIDIA's file - will conflict
```

✅ **Do:**
```bash
# Create wrapper or copy to custom/
cp scripts/gr00t_finetune.py custom/scripts/gr00t_finetune_so101.py
nano custom/scripts/gr00t_finetune_so101.py
```

### 2. Keep Custom Directory Organized

```
custom/
├── scripts/           ← Shell scripts for training
├── configs/           ← Custom config files
├── notebooks/         ← Jupyter notebooks for analysis
└── README.md          ← Documentation
```

### 3. Use Descriptive Commit Messages

✅ **Good:**
```bash
git commit -m "Update mini-MVP script: reduce batch size to 2 for 10-episode testing"
```

❌ **Bad:**
```bash
git commit -m "update"
```

### 4. Pull Upstream Regularly

```bash
# Every Monday or before major training runs
git fetch upstream
git merge upstream/main
```

### 5. Document Your Changes

Keep `custom/README.md` updated with:
- What each script does
- How to use them
- Any special requirements
- Recent changes

---

## Quick Reference

### Essential Commands

```bash
# Check status
git status

# See what changed
git diff

# Add and commit
git add custom/ jdocs/
git commit -m "Description"

# Push to your fork
git push origin main

# Pull from NVIDIA
git fetch upstream
git merge upstream/main

# View commit history
git log --oneline --graph --all

# Check remotes
git remote -v
```

### Directory Structure After Setup

```
~/project/Isaac-GR00T/
├── .git/                          ← Git repository
├── scripts/                       ← NVIDIA's original scripts
├── gr00t/                         ← NVIDIA's code
├── custom/                        ← YOUR scripts (new)
│   ├── scripts/
│   │   ├── train_groot_mini_mvp.sh
│   │   ├── train_groot_so101_mvp.sh
│   │   ├── train_groot_so101_full.sh
│   │   └── setup_groot_env.sh
│   └── README.md
└── jdocs/                         ← YOUR documentation (new)
    ├── SETUP_RECOVERY_GUIDE.md
    ├── GIT_WORKFLOW_GUIDE.md
    └── ...
```

---

## Summary

**Setup (One-Time):**
1. Fork Isaac-GR00T on GitHub
2. Configure remotes (upstream + origin)
3. Create `custom/` directory
4. Copy scripts to `custom/scripts/`
5. Commit and push to your fork

**Daily Use:**
```bash
# Work from Isaac-GR00T directory
cd ~/project/Isaac-GR00T

# Run training
bash custom/scripts/train_groot_mini_mvp.sh

# Commit changes
git add custom/
git commit -m "Update training config"
git push origin main

# Pull NVIDIA updates (weekly)
git fetch upstream
git merge upstream/main
```

**Benefits:**
- ✅ Your scripts tracked in Git
- ✅ Easy to pull NVIDIA updates
- ✅ No merge conflicts
- ✅ Easy to share with others
- ✅ Clear separation of concerns

Ready to set up? Follow Step 1 above! 🚀
