#!/bin/bash
################################################################################
# Isaac-GR00T Environment Setup Script
# Creates a separate 'groot' conda environment for GR00T training
################################################################################

set -e

echo "========================================================================"
echo "Isaac-GR00T Environment Setup"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Clone Isaac-GR00T repository"
echo "  2. Create 'groot' conda environment (Python 3.10)"
echo "  3. Install Isaac-GR00T with dependencies"
echo "  4. Install flash-attn 2.7.1"
echo ""
echo "Time: ~5-10 minutes"
echo "Disk: ~5GB for environment + ~500MB for repo"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo ""
echo "========================================================================"
echo "Step 1/4: Clone Isaac-GR00T"
echo "========================================================================"

# Create ~/project directory if it doesn't exist
mkdir -p ~/project

cd ~/project
if [ -d "Isaac-GR00T" ]; then
    echo "✓ Isaac-GR00T already cloned"
    cd Isaac-GR00T
    git pull
else
    echo "Cloning Isaac-GR00T to ~/project/Isaac-GR00T..."
    git clone https://github.com/NVIDIA/Isaac-GR00T
    cd Isaac-GR00T
fi

GROOT_PATH=$(pwd)
echo "Isaac-GR00T location: $GROOT_PATH"

echo ""
echo "========================================================================"
echo "Step 2/4: Create 'groot' Conda Environment"
echo "========================================================================"

# Check if groot env already exists
if conda env list | grep -q "^groot "; then
    echo "⚠️  'groot' environment already exists"
    echo "Options:"
    echo "  1. Remove and recreate (recommended)"
    echo "  2. Skip creation and use existing"
    echo ""
    read -p "Enter choice (1 or 2): " choice

    if [ "$choice" = "1" ]; then
        echo "Removing existing 'groot' environment..."
        conda env remove -n groot -y
        echo "Creating fresh 'groot' environment..."
        conda create -n groot python=3.10 -y
    else
        echo "Using existing 'groot' environment..."
    fi
else
    echo "Creating 'groot' environment with Python 3.10..."
    conda create -n groot python=3.10 -y
fi

echo ""
echo "========================================================================"
echo "Step 3/4: Install Isaac-GR00T Dependencies"
echo "========================================================================"

echo "Activating 'groot' environment..."
source /home/jrobot/anaconda3/bin/activate groot

echo "Installing Isaac-GR00T with [base] dependencies..."
pip install -e .[base]

echo ""
echo "Installed packages:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"

echo ""
echo "========================================================================"
echo "Step 4/4: Install flash-attn"
echo "========================================================================"

echo "⚠️  WARNING: flash-attn compilation takes 5-10 minutes"
echo "Installing flash-attn 2.7.1.post4..."
pip install --no-build-isolation flash-attn==2.7.1.post4

echo ""
echo "Verifying installation..."
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Isaac-GR00T cloned to: $GROOT_PATH"
echo "  ✓ 'groot' conda environment created"
echo "  ✓ All dependencies installed"
echo ""
echo "To use GR00T:"
echo "  conda activate groot"
echo "  export ISAAC_GROOT_ROOT=$GROOT_PATH"
echo ""
echo "Add to ~/.bashrc for convenience:"
echo "  echo 'export ISAAC_GROOT_ROOT=$GROOT_PATH' >> ~/.bashrc"
echo ""
echo "Next steps:"
echo "  1. Run mini-MVP test:"
echo "     cd /home/jrobot/project/XLeRobot"
echo "     bash scripts/train_groot_mini_mvp.sh"
echo ""
