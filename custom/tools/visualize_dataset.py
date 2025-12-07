#!/usr/bin/env python3
"""
Robot Dataset Visualizer - CLI Entry Point

A Gradio-based web application for visualizing GR00T/LeRobot robot arm datasets
with synchronized camera views and time-series plots.

Usage:
    python visualize_dataset.py [OPTIONS]

Options:
    --dataset PATH      Path to GR00T dataset (default: /home/jrobot/project/XLeRobot/datasets_groot)
    --port INT          Gradio server port (default: 7860)
    --share             Create a public link

Examples:
    # Basic usage with default dataset
    python visualize_dataset.py

    # Specify custom dataset path
    python visualize_dataset.py --dataset /path/to/my/dataset

    # Create a public shareable link
    python visualize_dataset.py --share

    # Use a different port
    python visualize_dataset.py --port 8080
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from dataset_visualizer.app import main

if __name__ == "__main__":
    main()
