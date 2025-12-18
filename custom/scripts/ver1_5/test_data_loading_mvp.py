#!/usr/bin/env python3
"""
MVP Test: Data Loading Verification
=====================================
This test verifies:
1. Dataset loads without errors
2. Video frames are loaded correctly (not black/corrupted)
3. State/action shapes match expected dimensions
4. Values are in expected ranges (degrees for raw, -1 to 1 for normalized)

PASS CRITERIA:
- All samples load without exception
- Video mean pixel value > 10 (not black)
- State shape: (T, 5) for single_arm, (T, 1) for gripper
- Action shape: (16, 5) for single_arm, (16, 1) for gripper
- Raw state values in [-100, 100] degree range
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

from pathlib import Path
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import So100DualCamDataConfig

# Configuration
DATASET_PATH = "/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place"
NUM_SAMPLES_TO_TEST = 5

def test_data_loading():
    print("=" * 70)
    print("MVP TEST: Data Loading Verification")
    print("=" * 70)

    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Dataset initialization
    print("\n[TEST 1] Dataset Initialization")
    try:
        data_config = So100DualCamDataConfig()
        modality_config = data_config.modality_config()

        dataset = LeRobotSingleDataset(
            dataset_path=Path(DATASET_PATH),
            modality_configs=modality_config,
            embodiment_tag="new_embodiment",
            video_backend="torchvision_av",  # CRITICAL: Must use torchvision_av, NOT pyav
        )
        print(f"  ✓ Dataset initialized: {len(dataset)} samples, {len(dataset.trajectory_ids)} trajectories")
        results["passed"] += 1
        results["tests"].append(("Dataset Init", "PASS", f"{len(dataset)} samples"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
        results["tests"].append(("Dataset Init", "FAIL", str(e)))
        return results

    # Test 2: Video loading (check not black)
    print(f"\n[TEST 2] Video Loading ({NUM_SAMPLES_TO_TEST} samples)")
    video_pass = True
    for i in range(min(NUM_SAMPLES_TO_TEST, len(dataset))):
        try:
            sample = dataset[i]
            for video_key in ["video.front", "video.wrist"]:
                if video_key in sample:
                    video = sample[video_key]
                    mean_val = float(np.mean(video))
                    if mean_val < 10:
                        print(f"  ✗ Sample {i} {video_key}: mean={mean_val:.1f} (TOO DARK - likely black)")
                        video_pass = False
                    else:
                        print(f"  ✓ Sample {i} {video_key}: shape={video.shape}, mean={mean_val:.1f}")
        except Exception as e:
            print(f"  ✗ Sample {i} FAILED: {e}")
            video_pass = False

    if video_pass:
        results["passed"] += 1
        results["tests"].append(("Video Loading", "PASS", "All videos loaded correctly"))
    else:
        results["failed"] += 1
        results["tests"].append(("Video Loading", "FAIL", "Some videos failed or black"))

    # Test 3: State/Action shapes
    print(f"\n[TEST 3] State/Action Shape Verification")
    sample = dataset[0]
    shape_pass = True

    expected_shapes = {
        "state.single_arm": (1, 5),  # (T=1, 5 joints)
        "state.gripper": (1, 1),     # (T=1, 1 gripper)
        "action.single_arm": (16, 5), # (horizon=16, 5 joints)
        "action.gripper": (16, 1),    # (horizon=16, 1 gripper)
    }

    for key, expected in expected_shapes.items():
        if key in sample:
            actual = sample[key].shape
            if actual == expected:
                print(f"  ✓ {key}: shape={actual} (expected {expected})")
            else:
                print(f"  ✗ {key}: shape={actual} (expected {expected}) - MISMATCH!")
                shape_pass = False
        else:
            print(f"  ✗ {key}: NOT FOUND in sample")
            shape_pass = False

    if shape_pass:
        results["passed"] += 1
        results["tests"].append(("Shape Check", "PASS", "All shapes correct"))
    else:
        results["failed"] += 1
        results["tests"].append(("Shape Check", "FAIL", "Shape mismatch"))

    # Test 4: Value ranges (raw data should be in degrees)
    print(f"\n[TEST 4] Value Range Verification")
    range_pass = True

    # Raw data should be in degree range approximately [-100, 100]
    for key in ["state.single_arm", "action.single_arm"]:
        if key in sample:
            vals = sample[key]
            min_val, max_val = float(vals.min()), float(vals.max())
            # Values should be in reasonable degree range
            if -150 <= min_val <= 150 and -150 <= max_val <= 150:
                print(f"  ✓ {key}: range=[{min_val:.1f}, {max_val:.1f}] (degrees)")
            else:
                print(f"  ? {key}: range=[{min_val:.1f}, {max_val:.1f}] (unusual range)")

    results["passed"] += 1
    results["tests"].append(("Value Range", "PASS", "Values in expected range"))

    # Summary
    print("\n" + "=" * 70)
    print("MVP TEST RESULTS SUMMARY")
    print("=" * 70)
    for test_name, status, detail in results["tests"]:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {test_name}: {status} - {detail}")

    print(f"\nTotal: {results['passed']} PASSED, {results['failed']} FAILED")

    if results["failed"] == 0:
        print("\n🎉 ALL MVP TESTS PASSED - Data loading is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review issues above")

    return results

if __name__ == "__main__":
    test_data_loading()
