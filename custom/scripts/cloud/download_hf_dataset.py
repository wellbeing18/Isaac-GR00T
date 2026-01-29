#!/usr/bin/env python3
"""
Download LeRobot dataset from HuggingFace Hub for GROOT training.

This script downloads a LeRobot v3.0 dataset from HuggingFace Hub to a local
directory, ready for conversion to GROOT format.

Usage:
    # Download default dataset (jasmine314342/picknplace-bimanual-464)
    python download_hf_dataset.py

    # Download specific dataset
    python download_hf_dataset.py --repo-id your-username/your-dataset

    # Download to specific directory
    python download_hf_dataset.py --output /path/to/output

    # Download specific version
    python download_hf_dataset.py --revision v3.0

Example workflow:
    # 1. Download dataset
    python custom/scripts/cloud/download_hf_dataset.py \
        --repo-id jasmine314342/picknplace-bimanual-464 \
        --output ./datasets_lerobot/picknplace-bimanual-464

    # 2. Convert to GROOT format
    python custom/scripts/cloud/convert_bimanual_to_groot.py \
        --input ./datasets_lerobot/picknplace-bimanual-464 \
        --output ./datasets/bimanual_groot

    # 3. Train
    DATASET_PATH=./datasets/bimanual_groot bash custom/scripts/cloud/train_groot_bimanual.sh
"""

import argparse
import json
import sys
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_REPO_ID = "jasmine314342/picknplace-bimanual-464"
DEFAULT_OUTPUT_DIR = "./datasets_lerobot"
DEFAULT_REVISION = "main"  # or "v3.0" for specific version
# ============================================================================


def download_dataset(repo_id: str, output_dir: Path, revision: str = "main") -> Path:
    """Download dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # Extract dataset name from repo_id (e.g., "user/dataset" -> "dataset")
    dataset_name = repo_id.split("/")[-1]
    local_dir = output_dir / dataset_name

    print("=" * 70)
    print("Downloading LeRobot Dataset from HuggingFace Hub")
    print("=" * 70)
    print(f"  Repository:  {repo_id}")
    print(f"  Revision:    {revision}")
    print(f"  Output:      {local_dir}")
    print("=" * 70)
    print()

    # Check if already downloaded
    if (local_dir / "meta" / "info.json").exists():
        print(f"Dataset already exists at {local_dir}")
        print("Use --force to re-download.")

        # Print dataset info
        with open(local_dir / "meta" / "info.json") as f:
            info = json.load(f)
        print(f"  - Version: {info.get('codebase_version', 'N/A')}")
        print(f"  - Episodes: {info.get('total_episodes', 'N/A')}")
        print(f"  - Frames: {info.get('total_frames', 'N/A')}")
        return local_dir

    print("Downloading dataset (this may take a while)...")
    print()

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
        )
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check if repo_id is correct")
        print("  2. For private datasets, run: huggingface-cli login")
        print("  3. Check your internet connection")
        sys.exit(1)

    print()
    print("Download complete!")

    # Verify and print info
    info_path = local_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print()
        print("Dataset Info:")
        print(f"  - Version: {info.get('codebase_version', 'N/A')}")
        print(f"  - Episodes: {info.get('total_episodes', 'N/A')}")
        print(f"  - Frames: {info.get('total_frames', 'N/A')}")
        print(f"  - FPS: {info.get('fps', 'N/A')}")

        # Check features
        features = info.get("features", {})
        state_dim = features.get("observation.state", {}).get("shape", [0])[0]
        action_dim = features.get("action", {}).get("shape", [0])[0]
        print(f"  - State dim: {state_dim}")
        print(f"  - Action dim: {action_dim}")

        # Check video keys
        video_keys = [k for k in features.keys() if k.startswith("observation.images.")]
        print(f"  - Cameras: {video_keys}")
    else:
        print("WARNING: meta/info.json not found - dataset may be incomplete")

    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download LeRobot dataset from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id", "-r",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help=f"Dataset revision/version (default: {DEFAULT_REVISION})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle force re-download
    if args.force:
        dataset_name = args.repo_id.split("/")[-1]
        local_dir = output_dir / dataset_name
        if local_dir.exists():
            import shutil
            print(f"Removing existing dataset at {local_dir}")
            shutil.rmtree(local_dir)

    local_dir = download_dataset(args.repo_id, output_dir, args.revision)

    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. Convert to GROOT format:")
    print(f"   python custom/scripts/cloud/convert_bimanual_to_groot.py \\")
    print(f"       --input {local_dir} \\")
    print(f"       --output ./datasets/bimanual_groot")
    print()
    print("2. Start training:")
    print(f"   DATASET_PATH=./datasets/bimanual_groot \\")
    print(f"       bash custom/scripts/cloud/train_groot_bimanual.sh")
    print()


if __name__ == "__main__":
    main()
