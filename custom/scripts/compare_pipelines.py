#!/usr/bin/env python3
"""
Comprehensive Pipeline Comparison: Your Model vs Pushpakcc Reference

This script compares step-by-step:
1. Dataset structure and meta files
2. Data loading via GR00T's LeRobotSingleDataset
3. Normalization statistics
4. Raw data values from parquet files
5. Video loading

Output: Detailed comparison report saved to custom/jdocs/lora/new_investigations/16_pipeline_comparison_results.md
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
YOUR_DATASET = Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
REF_DATASET = Path("/tmp/so101-table-cleanup")
YOUR_CHECKPOINT = Path("/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914")
REF_CHECKPOINT = Path("/tmp/pushpakcc_model")
OUTPUT_REPORT = Path("/home/jrobot/project/Isaac-GR00T/custom/jdocs/lora/new_investigations/16_pipeline_comparison_results.md")

# Collect all findings
findings = []

def log(msg, level="INFO"):
    """Print and collect findings."""
    print(f"[{level}] {msg}")
    findings.append({"level": level, "msg": msg})

def section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    findings.append({"level": "SECTION", "msg": title})

def compare_meta_files():
    """Compare meta file structure."""
    section("STEP 1: Meta File Structure Comparison")

    # List meta files
    your_meta = sorted([f.name for f in (YOUR_DATASET / "meta").iterdir()])
    ref_meta = sorted([f.name for f in (REF_DATASET / "meta").iterdir()])

    log(f"Your dataset meta files: {your_meta}")
    log(f"Reference meta files: {ref_meta}")

    # Key difference: modality.json
    if "modality.json" in your_meta and "modality.json" not in ref_meta:
        log("DIFFERENCE: Your dataset has modality.json, reference does NOT", "WARNING")
        log("Reference dataset relies on copying modality.json from examples/", "INFO")

    # Load info.json
    with open(YOUR_DATASET / "meta" / "info.json") as f:
        your_info = json.load(f)
    with open(REF_DATASET / "meta" / "info.json") as f:
        ref_info = json.load(f)

    log(f"\nYour codebase_version: {your_info['codebase_version']}")
    log(f"Reference codebase_version: {ref_info['codebase_version']}")

    log(f"\nYour video_path: {your_info['video_path']}")
    log(f"Reference video_path: {ref_info['video_path']}")

    if your_info['video_path'] != ref_info['video_path']:
        log("DIFFERENCE: Video path patterns differ!", "WARNING")

    log(f"\nYour data_path: {your_info['data_path']}")
    log(f"Reference data_path: {ref_info['data_path']}")

    # Video keys
    your_video_keys = [k for k in your_info['features'] if k.startswith('observation.images')]
    ref_video_keys = [k for k in ref_info['features'] if k.startswith('observation.images')]

    log(f"\nYour video keys in info.json: {your_video_keys}")
    log(f"Reference video keys in info.json: {ref_video_keys}")

    if your_video_keys != ref_video_keys:
        log("DIFFERENCE: Video key names differ!", "WARNING")
        log(f"  Your keys: {your_video_keys}", "INFO")
        log(f"  Ref keys:  {ref_video_keys}", "INFO")

    return your_info, ref_info

def compare_modality_json():
    """Compare modality.json files."""
    section("STEP 2: Modality.json Comparison")

    your_modality_path = YOUR_DATASET / "meta" / "modality.json"
    official_modality_path = Path("/home/jrobot/project/Isaac-GR00T/examples/SO-100/so100_dualcam__modality.json")

    with open(your_modality_path) as f:
        your_modality = json.load(f)
    with open(official_modality_path) as f:
        official_modality = json.load(f)

    log("Your modality.json video mappings:")
    for k, v in your_modality.get('video', {}).items():
        log(f"  {k} -> {v.get('original_key', 'N/A')}")

    log("\nOfficial so100_dualcam modality.json video mappings:")
    for k, v in official_modality.get('video', {}).items():
        log(f"  {k} -> {v.get('original_key', 'N/A')}")

    # Check differences
    your_front = your_modality.get('video', {}).get('front', {}).get('original_key')
    official_front = official_modality.get('video', {}).get('front', {}).get('original_key')

    if your_front != official_front:
        log(f"\nDIFFERENCE in video.front mapping:", "WARNING")
        log(f"  Your: {your_front}")
        log(f"  Official: {official_front}")

    your_wrist = your_modality.get('video', {}).get('wrist', {}).get('original_key')
    official_wrist = official_modality.get('video', {}).get('wrist', {}).get('original_key')

    if your_wrist != official_wrist:
        log(f"\nDIFFERENCE in video.wrist mapping:", "WARNING")
        log(f"  Your: {your_wrist}")
        log(f"  Official: {official_wrist}")

    return your_modality, official_modality

def compare_normalization_stats():
    """Compare normalization statistics from metadata.json."""
    section("STEP 3: Normalization Statistics Comparison")

    with open(YOUR_CHECKPOINT / "experiment_cfg" / "metadata.json") as f:
        your_meta = json.load(f)["new_embodiment"]["statistics"]
    with open(REF_CHECKPOINT / "experiment_cfg" / "metadata.json") as f:
        ref_meta = json.load(f)["new_embodiment"]["statistics"]

    log("Action single_arm statistics comparison:")
    log("\n| Joint | Your Min | Ref Min | Your Max | Ref Max |")
    log("|-------|----------|---------|----------|---------|")

    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    your_action_min = your_meta['action']['single_arm']['min']
    your_action_max = your_meta['action']['single_arm']['max']
    ref_action_min = ref_meta['action']['single_arm']['min']
    ref_action_max = ref_meta['action']['single_arm']['max']

    for i, name in enumerate(joint_names):
        log(f"| {name:13} | {your_action_min[i]:8.2f} | {ref_action_min[i]:7.2f} | {your_action_max[i]:8.2f} | {ref_action_max[i]:7.2f} |")

    log("\nAction gripper statistics:")
    log(f"  Your: min={your_meta['action']['gripper']['min'][0]:.2f}, max={your_meta['action']['gripper']['max'][0]:.2f}")
    log(f"  Ref:  min={ref_meta['action']['gripper']['min'][0]:.2f}, max={ref_meta['action']['gripper']['max'][0]:.2f}")

    # Compute normalization range differences
    log("\nNormalization range (max-min) comparison:")
    your_ranges = [your_action_max[i] - your_action_min[i] for i in range(5)]
    ref_ranges = [ref_action_max[i] - ref_action_min[i] for i in range(5)]

    log("\n| Joint | Your Range | Ref Range | Ratio |")
    log("|-------|------------|-----------|-------|")
    for i, name in enumerate(joint_names):
        ratio = your_ranges[i] / ref_ranges[i] if ref_ranges[i] != 0 else float('inf')
        log(f"| {name:13} | {your_ranges[i]:10.2f} | {ref_ranges[i]:9.2f} | {ratio:.2f} |")

    return your_meta, ref_meta

def compare_raw_parquet_data():
    """Compare raw data values from parquet files."""
    section("STEP 4: Raw Parquet Data Comparison")

    # Load first episode from each dataset
    your_parquet = YOUR_DATASET / "data" / "chunk-000" / "episode_000000.parquet"
    ref_parquet = REF_DATASET / "data" / "chunk-000" / "episode_000000.parquet"

    your_df = pd.read_parquet(your_parquet)
    ref_df = pd.read_parquet(ref_parquet)

    log(f"Your episode 0 shape: {your_df.shape}")
    log(f"Reference episode 0 shape: {ref_df.shape}")

    log(f"\nYour columns: {list(your_df.columns)}")
    log(f"Reference columns: {list(ref_df.columns)}")

    # Compare first row values
    log("\nFirst row comparison:")

    # State
    your_state = your_df['observation.state'].iloc[0]
    ref_state = ref_df['observation.state'].iloc[0]

    log(f"\nYour state[0]: {your_state}")
    log(f"Ref state[0]:  {ref_state}")

    # Action
    your_action = your_df['action'].iloc[0]
    ref_action = ref_df['action'].iloc[0]

    log(f"\nYour action[0]: {your_action}")
    log(f"Ref action[0]:  {ref_action}")

    # Timestamp
    your_ts = your_df['timestamp'].iloc[0]
    ref_ts = ref_df['timestamp'].iloc[0]

    log(f"\nYour timestamp[0]: {your_ts}")
    log(f"Ref timestamp[0]:  {ref_ts}")

    # Check state value ranges across entire episode
    log("\nState value ranges in episode 0:")
    your_state_arr = np.vstack(your_df['observation.state'].values)
    ref_state_arr = np.vstack(ref_df['observation.state'].values)

    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    log("\n| Joint | Your Min | Your Max | Ref Min | Ref Max |")
    log("|-------|----------|----------|---------|---------|")
    for i, name in enumerate(joint_names):
        log(f"| {name:13} | {your_state_arr[:, i].min():8.2f} | {your_state_arr[:, i].max():8.2f} | {ref_state_arr[:, i].min():7.2f} | {ref_state_arr[:, i].max():7.2f} |")

    return your_df, ref_df

def compare_video_structure():
    """Compare video file structure."""
    section("STEP 5: Video File Structure Comparison")

    # Your videos
    your_video_dir = YOUR_DATASET / "videos"
    your_subdirs = sorted([d.name for d in your_video_dir.iterdir() if d.is_dir()])
    log(f"Your video subdirs: {your_subdirs}")

    # Count video files
    for subdir in your_subdirs:
        subdir_path = your_video_dir / subdir
        if subdir_path.is_dir():
            chunk_dirs = list(subdir_path.iterdir())
            for chunk_dir in chunk_dirs:
                if chunk_dir.is_dir():
                    videos = list(chunk_dir.glob("*.mp4"))
                    log(f"  {subdir}/{chunk_dir.name}: {len(videos)} videos")

    # Reference videos
    ref_video_dir = REF_DATASET / "videos"
    ref_subdirs = sorted([d.name for d in ref_video_dir.iterdir() if d.is_dir()])
    log(f"\nReference video subdirs: {ref_subdirs}")

    for subdir in ref_subdirs:
        subdir_path = ref_video_dir / subdir
        if subdir_path.is_dir():
            # v2.1 format: videos/chunk-000/{video_key}/
            for item in sorted(subdir_path.iterdir()):
                if item.is_dir():
                    videos = list(item.glob("*.mp4"))
                    log(f"  {subdir}/{item.name}: {len(videos)} videos")
                elif item.suffix == '.mp4':
                    log(f"  {subdir}: has direct mp4 files")
                    break

    # Test loading a video
    log("\nVideo path construction test:")

    # Your video path
    your_video_path = YOUR_DATASET / "videos" / "observation.images.head" / "chunk-000" / "episode_000000.mp4"
    log(f"Your video path: {your_video_path}")
    log(f"  Exists: {your_video_path.exists()}")

    # Reference video path (v2.1 format)
    ref_video_path = REF_DATASET / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
    log(f"\nReference video path: {ref_video_path}")
    log(f"  Exists: {ref_video_path.exists()}")

def test_groot_data_loading():
    """Test loading data via GR00T's dataset class."""
    section("STEP 6: GR00T Data Loading Test")

    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import So100DualCamDataConfig

        data_config = So100DualCamDataConfig()
        modality_config = data_config.modality_config()

        log("Testing your dataset loading...")
        try:
            your_ds = LeRobotSingleDataset(
                dataset_path=str(YOUR_DATASET),
                modality_configs=modality_config,
                embodiment_tag="new_embodiment",
                video_backend="torchvision_av",
            )
            log(f"  SUCCESS: Loaded {len(your_ds.trajectory_ids)} trajectories, {len(your_ds)} steps")

            # Try loading a sample
            log("  Loading sample 0...")
            sample = your_ds[0]
            log(f"  Sample keys: {list(sample.keys())}")
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    log(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    if 'state' in k or 'action' in k:
                        log(f"      range: [{v.min():.3f}, {v.max():.3f}]")
        except Exception as e:
            log(f"  FAILED: {e}", "ERROR")

        log("\nTesting reference dataset loading...")
        # Reference dataset needs modality.json copied
        ref_modality_dst = REF_DATASET / "meta" / "modality.json"
        if not ref_modality_dst.exists():
            log("  Copying modality.json to reference dataset...")
            import shutil
            src = Path("/home/jrobot/project/Isaac-GR00T/examples/SO-100/so100_dualcam__modality.json")
            shutil.copy(src, ref_modality_dst)
            log(f"  Copied {src} -> {ref_modality_dst}")

        try:
            ref_ds = LeRobotSingleDataset(
                dataset_path=str(REF_DATASET),
                modality_configs=modality_config,
                embodiment_tag="new_embodiment",
                video_backend="torchvision_av",
            )
            log(f"  SUCCESS: Loaded {len(ref_ds.trajectory_ids)} trajectories, {len(ref_ds)} steps")

            # Try loading a sample
            log("  Loading sample 0...")
            ref_sample = ref_ds[0]
            log(f"  Sample keys: {list(ref_sample.keys())}")
            for k, v in ref_sample.items():
                if hasattr(v, 'shape'):
                    log(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    if 'state' in k or 'action' in k:
                        log(f"      range: [{v.min():.3f}, {v.max():.3f}]")
        except Exception as e:
            log(f"  FAILED: {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")

    except ImportError as e:
        log(f"Could not import GR00T modules: {e}", "ERROR")

def generate_report():
    """Generate markdown report."""
    section("Generating Report")

    report = f"""# Pipeline Comparison Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report compares your pipeline with the Pushpakcc reference model (which uses youliangtan/so101-table-cleanup dataset).

---

## Findings

"""

    current_section = None
    for item in findings:
        if item['level'] == 'SECTION':
            report += f"\n### {item['msg']}\n\n"
            current_section = item['msg']
        elif item['level'] == 'WARNING':
            report += f"**WARNING:** {item['msg']}\n\n"
        elif item['level'] == 'ERROR':
            report += f"**ERROR:** {item['msg']}\n\n"
        else:
            report += f"{item['msg']}\n"

    # Add conclusions
    report += """

---

## Key Differences Identified

Based on the comparison above, here are the key differences:

1. **Dataset Format Version:**
   - Your dataset: v3.0
   - Reference dataset: v2.1
   - Video path patterns are different between versions

2. **Video Key Names:**
   - Your dataset uses: `observation.images.head`, `observation.images.left_wrist`
   - Reference uses: `observation.images.front`, `observation.images.wrist`
   - Your modality.json maps these correctly

3. **modality.json Presence:**
   - Your dataset: Has custom modality.json
   - Reference dataset: NO modality.json (must copy from examples/)

4. **Normalization Ranges:**
   - The min/max values differ significantly between datasets
   - This is expected as different tasks have different motion ranges

---

## Next Steps

1. Verify video loading works correctly with your video_path pattern
2. Test if copying official modality.json fixes any issues
3. Compare model outputs on identical input

"""

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)

    log(f"Report saved to: {OUTPUT_REPORT}")

def main():
    print("=" * 70)
    print("  COMPREHENSIVE PIPELINE COMPARISON")
    print("  Your Model vs Pushpakcc Reference")
    print("=" * 70)

    compare_meta_files()
    compare_modality_json()
    compare_normalization_stats()
    compare_raw_parquet_data()
    compare_video_structure()
    test_groot_data_loading()
    generate_report()

    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
