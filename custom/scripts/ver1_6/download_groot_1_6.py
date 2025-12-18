#!/usr/bin/env python3
"""
Download GR00T N1.6-3B model from HuggingFace.

This script downloads the base GR00T model to local storage for faster loading
during training and inference. The model is ~6GB and includes:
- Eagle backbone (Qwen3-1.7B + SigLIP2)
- 32-layer DiT action head
- Pre-trained weights

Usage:
    python download_groot_1_6.py

The model will be cached in the HuggingFace cache directory:
    ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/
"""

import os
import sys
from pathlib import Path

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
MODEL_ID = "nvidia/GR00T-N1.6-3B"
LOCAL_DIR = None  # Set to a path to download to specific location, or None for HF cache
# ============================================================================


def main():
    print("=" * 70)
    print("GR00T N1.6-3B Model Download")
    print("=" * 70)
    print(f"Model ID: {MODEL_ID}")
    print(f"Local dir: {LOCAL_DIR or 'HuggingFace cache (default)'}")
    print("=" * 70)

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    # Check if already downloaded
    api = HfApi()
    try:
        model_info = api.model_info(MODEL_ID)
        print(f"\nModel found on HuggingFace:")
        print(f"  - Last modified: {model_info.lastModified}")
        print(f"  - Downloads: {model_info.downloads}")
    except Exception as e:
        print(f"WARNING: Could not fetch model info: {e}")

    print("\nStarting download...")
    print("This may take 10-30 minutes depending on network speed.")
    print("-" * 70)

    try:
        # Download the model
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False if LOCAL_DIR else "auto",
            resume_download=True,  # Resume if interrupted
        )

        print("-" * 70)
        print(f"\nDownload complete!")
        print(f"Model location: {local_path}")

        # List downloaded files
        path = Path(local_path)
        files = list(path.glob("**/*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"Total size: {total_size / 1e9:.2f} GB")
        print(f"Files: {len([f for f in files if f.is_file()])}")

        # Check for key files
        key_files = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        ]
        print("\nKey files:")
        for kf in key_files:
            found = list(path.glob(f"**/{kf}"))
            status = "✓" if found else "✗"
            print(f"  {status} {kf}")

        print("\n" + "=" * 70)
        print("Model ready for use!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check network connection")
        print("2. Ensure you have enough disk space (~10GB)")
        print("3. Try: huggingface-cli login (if model requires auth)")
        sys.exit(1)


if __name__ == "__main__":
    main()
