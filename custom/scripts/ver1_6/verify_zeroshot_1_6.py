#!/usr/bin/env python3
"""
Verify GR00T N1.6 Base Model Zero-Shot Inference.

This script tests the GR00T N1.6 base model before finetuning to verify:
1. Model loads correctly
2. Modality config registration works (NEW_EMBODIMENT tag)
3. Dataset loading works with the converted dataset
4. Inference produces reasonable outputs (correct shape, no NaN/Inf)
5. Inference timing is acceptable (should be ~40-50ms on RTX 5090)

This helps identify configuration issues before investing time in training.

Usage:
    python verify_zeroshot_1_6.py [--dataset PATH] [--num-episodes NUM]

Reference: scripts/deployment/standalone_inference_script.py, getting_started/policy.md
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Base model (downloaded from HuggingFace)
BASE_MODEL = "nvidia/GR00T-N1.6-3B"

# Dataset path (converted dataset)
DEFAULT_DATASET = "/home/jrobot/project/Isaac-GR00T/datasets/so101_pick_place_groot"

# Modality config path
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"

# Device to run on
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Test parameters
NUM_TEST_EPISODES = 3
NUM_STEPS_PER_EPISODE = 5  # Run a few steps per episode
ACTION_HORIZON = 16       # Must match modality config

# Expected output dimensions for SO-101
EXPECTED_ARM_DIM = 5
EXPECTED_GRIPPER_DIM = 1
EXPECTED_ACTION_DIM = EXPECTED_ARM_DIM + EXPECTED_GRIPPER_DIM
# ============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def import_modality_config(config_path: str):
    """Import the modality config module to register NEW_EMBODIMENT."""
    logger.info(f"Loading modality config from: {config_path}")

    # Convert relative path to absolute
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Modality config not found: {full_path}")

    # Import the module to register the config
    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    logger.info("  Modality config loaded and registered")


def test_model_loading(model_path: str = BASE_MODEL):
    """Test that the model loads correctly with custom modality config."""
    logger.info("\n[1/5] Testing model loading...")

    from pathlib import Path
    import torch
    from transformers import AutoModel, AutoProcessor
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    start_time = time.time()

    try:
        # Import model registration
        import gr00t.model  # noqa: F401

        model_dir = Path(model_path)

        # Load model
        model = AutoModel.from_pretrained(model_dir)
        model.eval()
        model.to(device=DEVICE, dtype=torch.bfloat16)

        # Load processor with modality config override for NEW_EMBODIMENT
        # This allows us to use custom modality config with base model
        modality_configs_override = {
            EmbodimentTag.NEW_EMBODIMENT.value: MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
        }
        processor = AutoProcessor.from_pretrained(
            model_dir,
            modality_configs=modality_configs_override
        )
        processor.eval()

        load_time = time.time() - start_time

        logger.info(f"  Model loaded successfully in {load_time:.2f}s")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  Model dtype: bfloat16")
        logger.info(f"  Modality config overridden for NEW_EMBODIMENT")

        # Create a simple policy wrapper
        class SimplePolicyWrapper:
            def __init__(self, model, processor, embodiment_tag):
                self.model = model
                self.processor = processor
                self.embodiment_tag = embodiment_tag
                self.modality_configs = processor.get_modality_configs()[embodiment_tag.value]
                self.collate_fn = processor.collator

                # Extract language key
                language_keys = self.modality_configs["language"].modality_keys
                assert len(language_keys) == 1, "Only one language key is supported"
                self.language_key = language_keys[0]

            def get_modality_config(self):
                return self.modality_configs

            def get_action(self, observation):
                """Run inference and return action."""
                # Unbatch observation
                unbatched_obs = []
                batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
                for i in range(batch_size):
                    unbatched_value = {
                        "video": {k: v[i] for k, v in observation["video"].items()},
                        "state": {k: v[i] for k, v in observation["state"].items()},
                        "language": {k: v[i] for k, v in observation["language"].items()},
                    }
                    unbatched_obs.append(unbatched_value)

                # Convert to VLAStepData
                from gr00t.data.types import VLAStepData
                step_data_list = []
                for obs in unbatched_obs:
                    step_data = VLAStepData(
                        images=obs["video"],
                        states=obs["state"],
                        actions={},
                        text=obs["language"][self.language_key][0],
                        embodiment=self.embodiment_tag,
                    )
                    step_data_list.append(step_data)

                # Process and run model
                batch = self.collate_fn(step_data_list)
                batch = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in batch.items()}

                with torch.no_grad():
                    output = self.model(**batch)

                # Decode actions
                actions = self.processor.decode_actions(
                    output.action_pred,
                    embodiment_tag=self.embodiment_tag.value,
                )
                return actions, {}

        policy = SimplePolicyWrapper(model, processor, EmbodimentTag.NEW_EMBODIMENT)
        return policy

    except Exception as e:
        logger.error(f"  Failed to load model: {e}")
        raise


def test_dataset_loading(dataset_path: str, policy):
    """Test that the dataset loads correctly with the modality config."""
    logger.info("\n[2/5] Testing dataset loading...")

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    try:
        modality_config = policy.get_modality_config()
        logger.info(f"  Modality config retrieved for NEW_EMBODIMENT")

        loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            video_backend="torchcodec",
        )

        logger.info(f"  Dataset loaded: {len(loader)} episodes")
        logger.info(f"  Episode lengths: {loader.episode_lengths[:5]}... (showing first 5)")

        # Get statistics
        stats = loader.get_dataset_statistics()
        logger.info(f"  Statistics keys: {list(stats.keys())}")

        return loader

    except Exception as e:
        logger.error(f"  Failed to load dataset: {e}")
        raise


def test_single_inference(policy, loader, episode_idx: int = 0, step_idx: int = 0):
    """Test inference on a single observation."""
    logger.info(f"\n[3/5] Testing single inference (episode {episode_idx}, step {step_idx})...")

    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from copy import deepcopy

    try:
        # Load episode
        episode = loader[episode_idx]
        logger.info(f"  Episode {episode_idx} loaded: {len(episode)} frames")

        # Extract step data
        modality_configs = deepcopy(loader.modality_configs)
        modality_configs.pop("action")  # Remove action for observation-only extraction

        data_point = extract_step_data(
            episode, step_idx, modality_configs, EmbodimentTag.NEW_EMBODIMENT
        )

        # Build observation dict
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for lang_key in loader.modality_configs["language"].modality_keys:
            obs[lang_key] = data_point.text

        logger.info(f"  Observation built with keys: {list(obs.keys())}")

        # Parse to expected format
        parsed_obs = parse_observation(obs, loader.modality_configs)

        # Log observation shapes
        logger.info("  Observation shapes:")
        for mod_key in parsed_obs:
            if isinstance(parsed_obs[mod_key], dict):
                for k, v in parsed_obs[mod_key].items():
                    if isinstance(v, np.ndarray):
                        logger.info(f"    {mod_key}.{k}: {v.shape} ({v.dtype})")
                    else:
                        logger.info(f"    {mod_key}.{k}: {type(v)}")

        # Run inference
        start_time = time.time()
        action, info = policy.get_action(parsed_obs)
        inference_time = time.time() - start_time

        logger.info(f"  Inference completed in {inference_time*1000:.1f}ms")

        # Check action output
        logger.info("  Action output shapes:")
        for k, v in action.items():
            logger.info(f"    {k}: {v.shape} ({v.dtype})")

        return action, inference_time

    except Exception as e:
        logger.error(f"  Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def parse_observation(obs: dict, modality_configs: dict) -> dict:
    """Parse raw observation to expected policy input format."""
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"

            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]

    return new_obs


def test_action_validity(action: dict):
    """Test that the action output is valid (no NaN, correct dimensions)."""
    logger.info("\n[4/5] Testing action validity...")

    all_ok = True

    for key, value in action.items():
        # Check for NaN
        if np.isnan(value).any():
            logger.error(f"  {key}: Contains NaN values!")
            all_ok = False
        else:
            logger.info(f"  {key}: No NaN values")

        # Check for Inf
        if np.isinf(value).any():
            logger.error(f"  {key}: Contains Inf values!")
            all_ok = False
        else:
            logger.info(f"  {key}: No Inf values")

        # Check shape (should be [horizon, dim])
        expected_horizon = ACTION_HORIZON
        if value.ndim != 2:
            logger.error(f"  {key}: Expected 2D array, got {value.ndim}D")
            all_ok = False
        elif value.shape[0] != expected_horizon:
            logger.warning(f"  {key}: Horizon is {value.shape[0]}, expected {expected_horizon}")

        # Check dimensions based on key
        if "single_arm" in key:
            if value.shape[1] != EXPECTED_ARM_DIM:
                logger.error(f"  {key}: Dim is {value.shape[1]}, expected {EXPECTED_ARM_DIM}")
                all_ok = False
        elif "gripper" in key:
            if value.shape[1] != EXPECTED_GRIPPER_DIM:
                logger.error(f"  {key}: Dim is {value.shape[1]}, expected {EXPECTED_GRIPPER_DIM}")
                all_ok = False

        # Show value range
        logger.info(f"  {key}: range=[{value.min():.4f}, {value.max():.4f}], mean={value.mean():.4f}")

    return all_ok


def test_inference_timing(policy, loader, num_runs: int = 5):
    """Test inference timing across multiple runs."""
    logger.info(f"\n[5/5] Testing inference timing ({num_runs} runs)...")

    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from copy import deepcopy

    times = []

    # Warmup run
    logger.info("  Warming up (first run may be slow due to compilation)...")
    episode = loader[0]
    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    data_point = extract_step_data(episode, 0, modality_configs, EmbodimentTag.NEW_EMBODIMENT)

    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for lang_key in loader.modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text

    parsed_obs = parse_observation(obs, loader.modality_configs)

    # Warmup
    _ = policy.get_action(parsed_obs)

    # Timed runs
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        _ = policy.get_action(parsed_obs)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    logger.info(f"  Inference timing:")
    logger.info(f"    Average: {avg_time*1000:.1f}ms")
    logger.info(f"    Min:     {min_time*1000:.1f}ms")
    logger.info(f"    Max:     {max_time*1000:.1f}ms")

    # Target is ~40-50ms on RTX 5090
    if avg_time > 0.1:
        logger.warning(f"  Inference is slow (>100ms). Consider checking GPU utilization.")
    elif avg_time > 0.05:
        logger.info(f"  Inference timing is acceptable (50-100ms range)")
    else:
        logger.info(f"  Inference timing is excellent (<50ms)")

    return times


def main():
    parser = argparse.ArgumentParser(
        description="Verify GR00T N1.6 base model zero-shot inference"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to dataset (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=NUM_TEST_EPISODES,
        help=f"Number of episodes to test (default: {NUM_TEST_EPISODES})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BASE_MODEL,
        help=f"Model path or HF model ID (default: {BASE_MODEL})"
    )
    args = parser.parse_args()

    # Update base model if specified - use local variable instead of global
    model_to_use = args.model

    print("=" * 70)
    print("GR00T N1.6 Zero-Shot Verification")
    print("=" * 70)
    print(f"Model:   {model_to_use}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {DEVICE}")
    print("=" * 70)

    # Check dataset exists
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        logger.error("Run convert_lerobot_v3_to_groot_1_6.py first to create the dataset")
        sys.exit(1)

    try:
        # Step 0: Import modality config
        import_modality_config(MODALITY_CONFIG_PATH)

        # Step 1: Load model
        policy = test_model_loading(model_to_use)

        # Step 2: Load dataset
        loader = test_dataset_loading(args.dataset, policy)

        # Step 3: Test single inference
        action, _ = test_single_inference(policy, loader)

        # Step 4: Validate action output
        valid = test_action_validity(action)

        # Step 5: Test timing
        test_inference_timing(policy, loader)

        # Summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        if valid:
            print("  RESULT: All checks passed!")
            print("  Zero-shot inference is working correctly.")
            print("\n  The model may not produce meaningful actions without finetuning,")
            print("  but the pipeline is verified to be working.")
            print("\n  Next: Run train_groot_so101_1_6.sh to finetune the model")
        else:
            print("  RESULT: Some checks failed!")
            print("  Please review the errors above before proceeding.")
            sys.exit(1)

        print("=" * 70)

    except Exception as e:
        logger.error(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
