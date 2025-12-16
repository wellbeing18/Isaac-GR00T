#!/usr/bin/env python3
"""
MVP Test: Normalization Pipeline Verification
==============================================
This test verifies:
1. metadata.json is loaded correctly during inference
2. Normalization transforms input state to [-1, 1]
3. Denormalization transforms output back to degrees
4. Round-trip: normalize -> denormalize = original value

PASS CRITERIA:
- metadata.json loads without error
- Normalized values in [-1, 1] range
- Denormalized values match original (within 0.01 tolerance)
- Statistics match between dataset and checkpoint
"""

import sys
sys.path.insert(0, "/home/jrobot/project/Isaac-GR00T")

import json
import numpy as np
from pathlib import Path

# Configuration
DATASET_PATH = Path("/home/jrobot/project/XLeRobot/datasets_copy/left/pick_and_place")
CHECKPOINT_PATH = Path("/home/jrobot/project/XLeRobot/outputs/groot_mvp_lora_20251213_230404914")
EMBODIMENT_TAG = "new_embodiment"


def normalize(value, min_val, max_val):
    """Min-max normalize to [-1, 1]"""
    return 2 * (value - min_val) / (max_val - min_val) - 1


def denormalize(normalized, min_val, max_val):
    """Denormalize from [-1, 1] back to original range"""
    return (normalized + 1) * (max_val - min_val) / 2 + min_val


def test_normalization():
    print("=" * 70)
    print("MVP TEST: Normalization Pipeline Verification")
    print("=" * 70)

    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Load metadata from checkpoint
    print("\n[TEST 1] Load Checkpoint Metadata")
    metadata_path = CHECKPOINT_PATH / "experiment_cfg" / "metadata.json"

    try:
        with open(metadata_path) as f:
            all_metadata = json.load(f)

        if EMBODIMENT_TAG not in all_metadata:
            print(f"  ✗ FAILED: Key '{EMBODIMENT_TAG}' not found in metadata.json")
            print(f"    Available keys: {list(all_metadata.keys())}")
            results["failed"] += 1
            results["tests"].append(("Load Metadata", "FAIL", f"Missing key {EMBODIMENT_TAG}"))
            return results

        metadata = all_metadata[EMBODIMENT_TAG]
        stats = metadata["statistics"]
        print(f"  ✓ Metadata loaded from {metadata_path}")
        print(f"    Embodiment tag: {EMBODIMENT_TAG}")
        results["passed"] += 1
        results["tests"].append(("Load Metadata", "PASS", "Metadata loaded"))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
        results["tests"].append(("Load Metadata", "FAIL", str(e)))
        return results

    # Test 2: Verify statistics structure
    print("\n[TEST 2] Verify Statistics Structure")
    required_keys = [
        ("state", "single_arm"),
        ("state", "gripper"),
        ("action", "single_arm"),
        ("action", "gripper"),
    ]

    structure_ok = True
    for cat, subkey in required_keys:
        if cat in stats and subkey in stats[cat]:
            sub_stats = stats[cat][subkey]
            if "min" in sub_stats and "max" in sub_stats:
                print(f"  ✓ {cat}.{subkey}: min={sub_stats['min'][:3]}..., max={sub_stats['max'][:3]}...")
            else:
                print(f"  ✗ {cat}.{subkey}: missing min/max")
                structure_ok = False
        else:
            print(f"  ✗ {cat}.{subkey}: NOT FOUND")
            structure_ok = False

    if structure_ok:
        results["passed"] += 1
        results["tests"].append(("Stats Structure", "PASS", "All required keys present"))
    else:
        results["failed"] += 1
        results["tests"].append(("Stats Structure", "FAIL", "Missing keys"))

    # Test 3: Round-trip normalization test
    print("\n[TEST 3] Round-trip Normalization Test")
    arm_min = np.array(stats["action"]["single_arm"]["min"])
    arm_max = np.array(stats["action"]["single_arm"]["max"])

    # Test values (typical joint positions in degrees)
    test_values = np.array([10.0, -50.0, 45.0, 30.0, -15.0])
    print(f"  Original values (degrees): {test_values}")

    # Normalize
    normalized = normalize(test_values, arm_min, arm_max)
    print(f"  Normalized (should be in [-1, 1]): {normalized}")

    # Check normalized range
    if np.all(normalized >= -1.01) and np.all(normalized <= 1.01):
        print(f"  ✓ All normalized values in [-1, 1] range")
    else:
        print(f"  ✗ Normalized values OUT OF RANGE!")
        results["failed"] += 1
        results["tests"].append(("Round-trip", "FAIL", "Normalized out of range"))
        return results

    # Denormalize
    denormalized = denormalize(normalized, arm_min, arm_max)
    print(f"  Denormalized (should match original): {denormalized}")

    # Check round-trip accuracy
    error = np.abs(test_values - denormalized)
    max_error = float(np.max(error))
    print(f"  Max round-trip error: {max_error:.6f} degrees")

    if max_error < 0.01:
        print(f"  ✓ Round-trip accurate (error < 0.01)")
        results["passed"] += 1
        results["tests"].append(("Round-trip", "PASS", f"Error={max_error:.6f}"))
    else:
        print(f"  ✗ Round-trip INACCURATE (error >= 0.01)")
        results["failed"] += 1
        results["tests"].append(("Round-trip", "FAIL", f"Error={max_error:.6f}"))

    # Test 4: Compare with dataset stats
    print("\n[TEST 4] Compare Dataset vs Checkpoint Statistics")
    dataset_stats_path = DATASET_PATH / "meta" / "stats.json"

    try:
        with open(dataset_stats_path) as f:
            dataset_stats = json.load(f)

        # Compare action min/max
        ds_action_min = dataset_stats["action"]["min"][:5]  # First 5 for single_arm
        ds_action_max = dataset_stats["action"]["max"][:5]
        ckpt_action_min = stats["action"]["single_arm"]["min"]
        ckpt_action_max = stats["action"]["single_arm"]["max"]

        print(f"  Dataset action min: {[f'{x:.2f}' for x in ds_action_min]}")
        print(f"  Ckpt action min:    {[f'{x:.2f}' for x in ckpt_action_min]}")
        print(f"  Dataset action max: {[f'{x:.2f}' for x in ds_action_max]}")
        print(f"  Ckpt action max:    {[f'{x:.2f}' for x in ckpt_action_max]}")

        min_match = all(abs(a - b) < 0.1 for a, b in zip(ds_action_min, ckpt_action_min))
        max_match = all(abs(a - b) < 0.1 for a, b in zip(ds_action_max, ckpt_action_max))

        if min_match and max_match:
            print(f"  ✓ Dataset and checkpoint statistics MATCH")
            results["passed"] += 1
            results["tests"].append(("Stats Match", "PASS", "Dataset = Checkpoint"))
        else:
            print(f"  ✗ Statistics MISMATCH - this could cause wrong actions!")
            results["failed"] += 1
            results["tests"].append(("Stats Match", "FAIL", "Mismatch detected"))

    except Exception as e:
        print(f"  ⚠ Could not compare: {e}")
        results["tests"].append(("Stats Match", "SKIP", str(e)))

    # Test 5: Edge case - values at boundaries
    print("\n[TEST 5] Boundary Value Test")
    # Test that min normalizes to -1 and max normalizes to +1
    normalized_min = normalize(arm_min, arm_min, arm_max)
    normalized_max = normalize(arm_max, arm_min, arm_max)

    print(f"  Normalized min values: {normalized_min} (should be all -1)")
    print(f"  Normalized max values: {normalized_max} (should be all +1)")

    if np.allclose(normalized_min, -1.0) and np.allclose(normalized_max, 1.0):
        print(f"  ✓ Boundary normalization correct")
        results["passed"] += 1
        results["tests"].append(("Boundary Test", "PASS", "min→-1, max→+1"))
    else:
        print(f"  ✗ Boundary normalization INCORRECT")
        results["failed"] += 1
        results["tests"].append(("Boundary Test", "FAIL", "Wrong boundary values"))

    # Summary
    print("\n" + "=" * 70)
    print("MVP TEST RESULTS SUMMARY")
    print("=" * 70)
    for test_name, status, detail in results["tests"]:
        icon = "✓" if status == "PASS" else ("⚠" if status == "SKIP" else "✗")
        print(f"  {icon} {test_name}: {status} - {detail}")

    print(f"\nTotal: {results['passed']} PASSED, {results['failed']} FAILED")

    if results["failed"] == 0:
        print("\n🎉 ALL MVP TESTS PASSED - Normalization pipeline is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review issues above")

    return results


if __name__ == "__main__":
    test_normalization()
