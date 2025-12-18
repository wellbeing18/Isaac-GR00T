#!/usr/bin/env python3
"""
Dataset Integrity Verification Tool

Verifies that a GR00T dataset is correctly formatted and will load properly
during training. Catches video/action misalignment issues BEFORE training.

Checks:
1. Video format: per-episode videos vs consolidated
2. Frame uniqueness: different episodes should have different frames
3. Timestamp alignment: timestamps must match video seek positions
4. Baseline MAE: ensures data is discriminative (action != state)
5. Dataloader simulation: loads data exactly as training does

Usage:
    python verify_dataset_integrity.py --dataset /path/to/dataset
    python verify_dataset_integrity.py --dataset /path/to/dataset --visual --save-report
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def load_info_json(dataset_path: Path) -> Dict:
    """Load dataset info.json."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    with open(info_path) as f:
        return json.load(f)


def load_episodes_jsonl(dataset_path: Path) -> List[Dict]:
    """Load episodes.jsonl."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def check_video_format(dataset_path: Path, info: Dict) -> Tuple[bool, str, Dict]:
    """
    Check if dataset uses per-episode or consolidated video format.

    Returns:
        (is_valid, message, details)
    """
    video_path_pattern = info.get("video_path", "")

    details = {
        "video_path_pattern": video_path_pattern,
        "format": "unknown",
        "per_episode_videos": 0,
        "consolidated_videos": 0,
    }

    # Check pattern
    uses_episode_format = "episode_{episode_index" in video_path_pattern
    uses_file_format = "file-" in video_path_pattern

    # Count actual video files
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        per_episode = list(videos_dir.glob("**/episode_*.mp4"))
        consolidated = list(videos_dir.glob("**/file-*.mp4"))
        details["per_episode_videos"] = len(per_episode)
        details["consolidated_videos"] = len(consolidated)

    if uses_episode_format and details["per_episode_videos"] > 0:
        details["format"] = "per_episode"
        return True, "Using per-episode video format (correct)", details
    elif uses_file_format or details["consolidated_videos"] > 0:
        details["format"] = "consolidated"
        if details["per_episode_videos"] == 0:
            return False, "Using consolidated video format WITHOUT per-episode videos - DATA CORRUPTION LIKELY", details
        else:
            return False, "Mixed format: has both consolidated and per-episode videos", details
    else:
        return False, f"Unknown video format: {video_path_pattern}", details


def check_timestamp_alignment(dataset_path: Path, episodes: List[Dict], info: Dict, sample_episodes: List[int] = None) -> Tuple[bool, str, Dict]:
    """
    Check if timestamps are properly aligned for the video format.

    For consolidated videos: timestamps should be cumulative (FAIL if reset to 0)
    For per-episode videos: timestamps SHOULD start near 0 (this is correct)
    """
    video_path_pattern = info.get("video_path", "")
    uses_per_episode = "episode_{episode_index" in video_path_pattern

    if sample_episodes is None:
        # Sample first, middle, last episodes
        all_eps = [ep["episode_index"] for ep in episodes]
        if len(all_eps) >= 3:
            sample_episodes = [all_eps[0], all_eps[len(all_eps)//2], all_eps[-1]]
        else:
            sample_episodes = all_eps

    details = {
        "episodes_checked": [],
        "timestamp_ranges": {},
        "resets_to_zero": [],
        "video_format": "per_episode" if uses_per_episode else "consolidated",
    }

    cumulative_time = 0.0
    issues = []

    for ep_idx in sample_episodes:
        chunk_idx = ep_idx // 1000
        parquet_path = dataset_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"

        if not parquet_path.exists():
            issues.append(f"Episode {ep_idx} parquet not found")
            continue

        try:
            df = pd.read_parquet(parquet_path)
            ts_min = df["timestamp"].min()
            ts_max = df["timestamp"].max()
            duration = ts_max - ts_min

            details["episodes_checked"].append(ep_idx)
            details["timestamp_ranges"][str(ep_idx)] = {"min": float(ts_min), "max": float(ts_max), "duration": float(duration)}

            # Check if timestamp resets to 0
            if ep_idx > 0 and ts_min < 1.0:
                details["resets_to_zero"].append(ep_idx)

            cumulative_time += duration + 0.1  # Small gap between episodes
        except Exception as e:
            issues.append(f"Episode {ep_idx}: {str(e)[:50]}")

    # For per-episode videos, timestamps SHOULD reset to 0 - this is correct
    if uses_per_episode:
        if details["resets_to_zero"]:
            return True, f"Timestamps start near 0 for each episode (correct for per-episode videos)", details
        else:
            return True, "Timestamps properly aligned for per-episode format", details

    # For consolidated videos, timestamps resetting to 0 is a BUG
    if details["resets_to_zero"]:
        return False, f"Timestamps reset to 0 for episodes {details['resets_to_zero']} - will cause frame misalignment with consolidated video", details

    if issues:
        return False, "; ".join(issues), details

    return True, "Timestamps properly aligned for consolidated video", details


def extract_frame_pyav(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame using PyAV."""
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Seek to approximate position
        target_pts = int(frame_idx * stream.average_rate.denominator / stream.average_rate.numerator * stream.time_base.denominator / stream.time_base.numerator)

        current_idx = 0
        for frame in container.decode(video=0):
            if current_idx == frame_idx:
                arr = frame.to_ndarray(format="rgb24")
                container.close()
                return arr
            current_idx += 1
            if current_idx > frame_idx + 10:  # Safety limit
                break
        container.close()
    except Exception as e:
        pass
    return None


def check_frame_uniqueness(dataset_path: Path, episodes: List[Dict], video_key: str = "observation.images.head", num_samples: int = 5) -> Tuple[bool, str, Dict]:
    """
    Check that different episodes have different first frames using the ACTUAL dataloader.
    If all episodes have the same first frame, data is corrupted.

    This uses LeRobotSingleDataset.get_step_data() - the EXACT same method used in training.
    This ensures we catch any video/timestamp misalignment issues.
    """
    details = {
        "episodes_sampled": [],
        "frame_hashes": {},
        "duplicate_frames": [],
        "method": "LeRobotSingleDataset (same as training)",
    }

    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config

        data_cfg = load_data_config("so100_dualcam")
        modality_config = data_cfg.modality_config()

        # Load dataset WITHOUT transforms to get raw video frames
        dataset = LeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=modality_config,
            video_backend="torchvision_av",
            transforms=None,  # No transforms - get raw video
            embodiment_tag="new_embodiment",
        )

        # Sample episodes evenly across the dataset
        num_episodes = len(dataset.trajectory_lengths)
        if num_episodes <= num_samples:
            sample_eps = list(range(num_episodes))
        else:
            step = num_episodes // num_samples
            sample_eps = [i * step for i in range(num_samples)]

        details["episodes_sampled"] = sample_eps
        frame_hashes = {}

        for ep_idx in sample_eps:
            try:
                # Load step 0 of each episode - this uses the exact same code path as training
                sample = dataset.get_step_data(ep_idx, 0)

                # Video data is in sample with keys like "video.front", "video.wrist"
                video_data = None
                for key in ["video.front", "video.wrist", "video.webcam"]:
                    if key in sample:
                        video_data = sample[key]
                        break

                if video_data is None:
                    details["frame_hashes"][ep_idx] = "NO_VIDEO_DATA"
                    continue

                # video_data shape: [T, H, W, C] where T=1 for single observation
                if hasattr(video_data, "numpy"):
                    video_data = video_data.numpy()
                frame = video_data[0]  # First frame, shape [H, W, C]

                # Hash the frame
                h = hashlib.md5(frame.tobytes()).hexdigest()[:16]
                details["frame_hashes"][ep_idx] = h

                if h in frame_hashes:
                    details["duplicate_frames"].append((ep_idx, frame_hashes[h]))
                else:
                    frame_hashes[h] = ep_idx

            except Exception as e:
                details["frame_hashes"][ep_idx] = f"ERROR: {str(e)[:50]}"

        # Count valid hashes
        valid_hashes = [h for h in details["frame_hashes"].values()
                        if not str(h).startswith("ERROR") and h != "NO_VIDEO_DATA"]
        unique_hashes = len(set(valid_hashes))

        if not valid_hashes:
            return False, "Could not extract any frames via dataloader", details

        if details["duplicate_frames"]:
            dups = [f"ep{a}==ep{b}" for a, b in details["duplicate_frames"]]
            return False, f"DUPLICATE FRAMES DETECTED: {', '.join(dups)} - data is corrupted!", details

        return True, f"All {unique_hashes} sampled episodes have unique first frames (via dataloader)", details

    except Exception as e:
        details["error"] = str(e)
        return False, f"Frame uniqueness check failed: {str(e)[:100]}", details


def compute_baseline_mae(dataset_path: Path, episodes: List[Dict], max_frames: int = 10000) -> Tuple[float, Dict]:
    """
    Compute baseline MAE for action=state prediction.
    This gives a lower bound - any useful model must beat this.
    """
    all_actions = []
    all_states = []
    frames_loaded = 0

    for ep in episodes:
        if frames_loaded >= max_frames:
            break

        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // 1000
        parquet_path = dataset_path / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"

        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        actions = np.array(df["action"].tolist())
        states = np.array(df["observation.state"].tolist())

        all_actions.append(actions)
        all_states.append(states)
        frames_loaded += len(actions)

    if not all_actions:
        return -1, {"error": "No data loaded"}

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    # MAE for action=state baseline
    mae_per_joint = np.mean(np.abs(all_actions - all_states), axis=0)
    mae_overall = np.mean(mae_per_joint)

    details = {
        "frames_analyzed": len(all_actions),
        "mae_per_joint": mae_per_joint.tolist(),
        "mae_overall": float(mae_overall),
        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    }

    return mae_overall, details


def simulate_dataloader(dataset_path: Path, num_samples: int = 5) -> Tuple[bool, str, Dict]:
    """
    Simulate how the training dataloader loads data.
    This catches issues that only appear during actual training.
    """
    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config
    except ImportError:
        return False, "Could not import GR00T modules", {}

    details = {
        "samples_loaded": 0,
        "errors": [],
        "sample_shapes": {},
    }

    try:
        data_cfg = load_data_config("so100_dualcam")
        modality_config = data_cfg.modality_config()

        # Try different video backends (torchvision_av first since decord has issues with some videos)
        video_backends = ["torchvision_av", "decord"]
        dataset = None

        for backend in video_backends:
            try:
                dataset = LeRobotSingleDataset(
                    dataset_path=str(dataset_path),
                    modality_configs=modality_config,
                    video_backend=backend,
                    transforms=None,
                    embodiment_tag="new_embodiment",
                )
                break
            except Exception as e:
                details["errors"].append(f"Backend {backend}: {str(e)[:100]}")

        if dataset is None:
            return False, "Could not initialize dataset with any video backend", details

        details["errors"] = []  # Clear backend errors if we succeeded

        # Sample from different episodes
        total_eps = len(dataset.trajectory_lengths)
        if total_eps <= num_samples:
            sample_eps = list(range(total_eps))
        else:
            step = total_eps // num_samples
            sample_eps = [i * step for i in range(num_samples)]

        for ep_idx in sample_eps:
            try:
                # get_step_data takes (trajectory_id, base_index)
                sample = dataset.get_step_data(ep_idx, 0)
                details["samples_loaded"] += 1

                # Record shapes (convert to native Python types for JSON)
                for key, val in sample.items():
                    if hasattr(val, "shape"):
                        details["sample_shapes"][key] = [int(x) for x in val.shape]
            except Exception as e:
                details["errors"].append(f"Episode {ep_idx}: {str(e)[:100]}")

        if details["errors"]:
            # Show first error for more context
            first_error = details["errors"][0] if details["errors"] else "Unknown"
            return False, f"Dataloader errors: {len(details['errors'])} failures. First: {first_error}", details

        return True, f"Successfully loaded {details['samples_loaded']} samples via LeRobotSingleDataset", details

    except Exception as e:
        details["errors"].append(str(e)[:200])
        return False, f"Dataloader simulation failed: {str(e)[:100]}", details


def simulate_training_batch(dataset_path: Path, batch_size: int = 4, num_batches: int = 3) -> Tuple[bool, str, Dict]:
    """
    Simulate actual training: DataLoader with shuffling, batching, transforms, collation.
    This is the most thorough check - it tests what training actually does.

    Training pipeline:
    1. LeRobotSingleDataset loads video frames using get_frames_by_timestamps()
    2. Transforms process video: VideoToTensor -> VideoCrop -> VideoResize -> VideoColorJitter -> VideoToNumpy
    3. GR00TTransform converts video to eagle_content with PIL Images
    4. DefaultDataCollator processes eagle_content through eagle_processor
    5. Result: batch has 'eagle_pixel_values' (video data) and other keys
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config
        from gr00t.experiment.trainer import BaseSampler
        from gr00t.model.transforms import DefaultDataCollator
    except ImportError as e:
        return False, f"Could not import required modules: {e}", {}

    details = {
        "batches_loaded": 0,
        "total_samples": 0,
        "errors": [],
        "batch_shapes": {},
        "shuffle_test": "not_run",
        "video_variance_test": "not_run",
        "video_key_found": None,
    }

    try:
        data_cfg = load_data_config("so100_dualcam")
        modality_config = data_cfg.modality_config()
        transforms = data_cfg.transform()

        # Create dataset with transforms (as training does)
        dataset = LeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=modality_config,
            video_backend="torchvision_av",
            transforms=transforms,
            embodiment_tag="new_embodiment",
        )

        # Create sampler with shuffling (as training does)
        sampler = BaseSampler(dataset, shuffle=True, seed=42)

        # Create collator (as training does)
        data_collator = DefaultDataCollator()

        # Create DataLoader (as training does)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Single-threaded for debugging
            drop_last=True,
            collate_fn=data_collator,  # Use the same collator as training
        )

        # Load batches
        video_data_list = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            details["batches_loaded"] += 1
            details["total_samples"] += batch_size

            # Record shapes from first batch
            if batch_idx == 0:
                for key, val in batch.items():
                    if hasattr(val, "shape"):
                        details["batch_shapes"][key] = [int(x) for x in val.shape]

            # After GR00TTransform + DefaultDataCollator:
            # Video is processed by eagle_processor and stored as 'eagle_pixel_values'
            # This is the actual key used in training
            video_key = None
            for k in ["eagle_pixel_values", "pixel_values"]:
                if k in batch and batch[k] is not None:
                    video_key = k
                    break

            if video_key:
                details["video_key_found"] = video_key
                video_data_list.append(batch[video_key].cpu().numpy())

        # Test 1: Verify shuffling works (different epochs should give different order)
        sampler.set_epoch(0)
        indices_epoch0 = list(iter(sampler))[:20]
        sampler.set_epoch(1)
        indices_epoch1 = list(iter(sampler))[:20]
        if indices_epoch0 == indices_epoch1:
            details["shuffle_test"] = "FAIL: Same order in different epochs"
        else:
            details["shuffle_test"] = "PASS: Different order in different epochs"

        # Test 2: Verify video frames have variance (not all the same)
        if video_data_list:
            all_frames = np.concatenate(video_data_list, axis=0)
            # Check variance across batch dimension
            frame_stds = np.std(all_frames, axis=0).mean()
            if frame_stds < 0.001:
                details["video_variance_test"] = f"FAIL: Video frames have near-zero variance ({frame_stds:.6f}) - likely all same frame"
            else:
                details["video_variance_test"] = f"PASS: Video frames have variance ({frame_stds:.4f})"
        else:
            details["video_variance_test"] = "SKIP: No video data in batch (keys: " + ", ".join(details["batch_shapes"].keys()) + ")"

        if details["errors"]:
            return False, f"Training simulation errors: {details['errors']}", details

        return True, f"Successfully simulated {details['batches_loaded']} training batches ({details['total_samples']} samples)", details

    except Exception as e:
        import traceback
        details["errors"].append(f"{str(e)[:200]}\n{traceback.format_exc()[:300]}")
        return False, f"Training simulation failed: {str(e)[:100]}", details


def generate_visual_report(dataset_path: Path, episodes: List[Dict], output_dir: Path, video_key: str = "observation.images.head"):
    """
    Generate visual verification showing EXACTLY what gets fed to the model during training.

    For each sample, shows:
    - Both camera views (video.front + video.wrist)
    - State values (6 joints)
    - Action values (16-step horizon)
    - Task description

    This proves the training pipeline loads correct, aligned data.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import torch
    except ImportError:
        print("  Skipping visual report (PIL/torch not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Generating training input visualization...")

    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config

        data_cfg = load_data_config("so100_dualcam")
        modality_config = data_cfg.modality_config()

        # Load dataset WITHOUT transforms to see raw data
        dataset = LeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=modality_config,
            video_backend="torchvision_av",
            transforms=None,
            embodiment_tag="new_embodiment",
        )

        # Sample 4 episodes evenly
        num_episodes = len(dataset.trajectory_lengths)
        num_samples = min(4, num_episodes)
        step = max(1, num_episodes // num_samples)
        sample_eps = [i * step for i in range(num_samples)]

        # Try to get font
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font_large = font_medium = font_small = ImageFont.load_default()

        # Create detailed visualization for each sample
        sample_images = []

        for ep_idx in sample_eps:
            try:
                # Load a sample from middle of episode (not just first frame)
                ep_length = dataset.trajectory_lengths[ep_idx]
                mid_step = min(ep_length // 2, ep_length - 17)  # Need 16 future actions
                mid_step = max(0, mid_step)

                sample = dataset.get_step_data(ep_idx, mid_step)

                # Extract data
                front_cam = sample.get("video.front")
                wrist_cam = sample.get("video.wrist")
                state_arm = sample.get("state.single_arm")
                state_grip = sample.get("state.gripper")
                action_arm = sample.get("action.single_arm")
                action_grip = sample.get("action.gripper")
                task_desc = sample.get("annotation.human.task_description", "N/A")

                # Convert to numpy
                def to_numpy(x):
                    if x is None:
                        return None
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    return np.array(x)

                front_cam = to_numpy(front_cam)
                wrist_cam = to_numpy(wrist_cam)
                state_arm = to_numpy(state_arm)
                state_grip = to_numpy(state_grip)
                action_arm = to_numpy(action_arm)
                action_grip = to_numpy(action_grip)

                # Create sample visualization (width=800)
                img_width = 900
                img_height = 400
                img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)

                # Title
                title = f"Episode {ep_idx}, Step {mid_step}/{ep_length}"
                draw.text((10, 5), title, fill=(0, 0, 0), font=font_large)

                # Task description
                if isinstance(task_desc, list):
                    task_desc = task_desc[0] if task_desc else "N/A"
                task_text = f"Task: {task_desc[:80]}..." if len(str(task_desc)) > 80 else f"Task: {task_desc}"
                draw.text((10, 30), task_text, fill=(0, 100, 0), font=font_medium)

                # Camera images (side by side)
                y_offset = 55
                cam_height = 160

                if front_cam is not None:
                    frame = front_cam[0] if front_cam.ndim == 4 else front_cam
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    pil_frame = Image.fromarray(frame)
                    # Resize to fit
                    aspect = pil_frame.width / pil_frame.height
                    new_w = int(cam_height * aspect)
                    pil_frame = pil_frame.resize((new_w, cam_height))
                    img.paste(pil_frame, (10, y_offset))
                    draw.text((10, y_offset + cam_height + 2), "video.front (head cam)", fill=(100, 100, 100), font=font_small)

                if wrist_cam is not None:
                    frame = wrist_cam[0] if wrist_cam.ndim == 4 else wrist_cam
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    pil_frame = Image.fromarray(frame)
                    aspect = pil_frame.width / pil_frame.height
                    new_w = int(cam_height * aspect)
                    pil_frame = pil_frame.resize((new_w, cam_height))
                    img.paste(pil_frame, (250, y_offset))
                    draw.text((250, y_offset + cam_height + 2), "video.wrist (wrist cam)", fill=(100, 100, 100), font=font_small)

                # State values (right side)
                x_state = 500
                draw.text((x_state, y_offset), "STATE (observation):", fill=(0, 0, 150), font=font_medium)
                joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
                if state_arm is not None:
                    state_vals = state_arm[0] if state_arm.ndim == 2 else state_arm
                    for i, (name, val) in enumerate(zip(joint_names, state_vals)):
                        draw.text((x_state, y_offset + 20 + i * 16), f"  {name}: {val:.2f}°", fill=(0, 0, 0), font=font_small)
                if state_grip is not None:
                    grip_val = state_grip[0][0] if state_grip.ndim == 2 else state_grip[0]
                    draw.text((x_state, y_offset + 20 + 5 * 16), f"  gripper: {grip_val:.2f}°", fill=(0, 0, 0), font=font_small)

                # Action values (16-step horizon)
                x_action = 700
                draw.text((x_action, y_offset), "ACTION (16 steps):", fill=(150, 0, 0), font=font_medium)
                if action_arm is not None:
                    # Show first, middle, last action
                    for idx, label in [(0, "t+0"), (7, "t+7"), (15, "t+15")]:
                        if idx < len(action_arm):
                            act = action_arm[idx]
                            act_str = f"{label}: [{', '.join([f'{v:.1f}' for v in act[:3]])}...]"
                            row = {"t+0": 0, "t+7": 1, "t+15": 2}[label]
                            draw.text((x_action, y_offset + 20 + row * 16), act_str, fill=(0, 0, 0), font=font_small)

                # Frame hash for uniqueness check
                if front_cam is not None:
                    frame_hash = hashlib.md5(front_cam.tobytes()).hexdigest()[:8]
                    draw.text((10, img_height - 40), f"Frame hash: {frame_hash}", fill=(150, 150, 150), font=font_small)

                # Alignment verification: show that this is from the correct parquet row
                draw.text((10, img_height - 22), f"Source: episode_{ep_idx:06d}.parquet row {mid_step} → episode_{ep_idx:06d}.mp4", fill=(100, 100, 100), font=font_small)

                # Separator line
                draw.line([(0, img_height - 1), (img_width, img_height - 1)], fill=(200, 200, 200), width=2)

                sample_images.append(img)

            except Exception as e:
                print(f"    Warning: Could not process episode {ep_idx}: {e}")
                import traceback
                traceback.print_exc()

        if not sample_images:
            print("  No samples could be processed")
            return

        # Stack all sample images vertically
        total_height = sum(img.height for img in sample_images)
        combined = Image.new("RGB", (sample_images[0].width, total_height), color=(255, 255, 255))
        y = 0
        for img in sample_images:
            combined.paste(img, (0, y))
            y += img.height

        # Add header
        header_height = 40
        final_img = Image.new("RGB", (combined.width, combined.height + header_height), color=(240, 240, 255))
        draw = ImageDraw.Draw(final_img)
        draw.text((10, 10), "TRAINING INPUT VERIFICATION - Exact data fed to model", fill=(0, 0, 100), font=font_large)
        final_img.paste(combined, (0, header_height))

        output_path = output_dir / "training_inputs_visualization.png"
        final_img.save(output_path)
        print(f"  Saved training input visualization to: {output_path}")
        print(f"  This shows EXACTLY what the model receives during training:")
        print(f"    - Both camera views (video.front, video.wrist)")
        print(f"    - State values (6 joints)")
        print(f"    - Action horizon (16 steps)")
        print(f"    - Task description")
        print(f"  If frame hashes are identical across episodes, data is corrupted!")

        # Also create simple frame grid for quick duplicate check
        print("\n  Creating frame uniqueness grid...")
        _generate_frame_grid(dataset, sample_eps, output_dir, font_small)

    except Exception as e:
        import traceback
        print(f"  Error generating visual report: {e}")
        traceback.print_exc()


def _generate_frame_grid(dataset, sample_eps: List[int], output_dir: Path, font):
    """Generate a simple grid showing first frame from each episode for quick uniqueness check."""
    from PIL import Image, ImageDraw

    frames = []
    labels = []
    hashes = []

    # Expand to more episodes for better coverage
    num_episodes = len(dataset.trajectory_lengths)
    expanded_eps = list(range(0, num_episodes, max(1, num_episodes // 9)))[:9]

    for ep_idx in expanded_eps:
        try:
            sample = dataset.get_step_data(ep_idx, 0)
            front_cam = sample.get("video.front")
            if front_cam is not None:
                if hasattr(front_cam, "numpy"):
                    front_cam = front_cam.numpy()
                frame = front_cam[0] if front_cam.ndim == 4 else front_cam
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                frames.append(frame)
                labels.append(f"Ep {ep_idx}")
                hashes.append(hashlib.md5(frame.tobytes()).hexdigest()[:8])
        except:
            pass

    if not frames:
        return

    # Create grid
    h, w = frames[0].shape[:2]
    # Scale down for grid
    scale = 0.4
    new_h, new_w = int(h * scale), int(w * scale)

    cols = 3
    rows = (len(frames) + cols - 1) // cols
    label_height = 35

    grid_img = Image.new("RGB", (cols * new_w, rows * (new_h + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    unique_hashes = set(hashes)
    hash_colors = {}
    colors = [(0, 150, 0), (200, 0, 0), (0, 0, 200), (200, 100, 0), (100, 0, 200)]
    for i, h in enumerate(unique_hashes):
        hash_colors[h] = colors[i % len(colors)]

    for i, (frame, label, h) in enumerate(zip(frames, labels, hashes)):
        r, c = i // cols, i % cols
        x = c * new_w
        y = r * (new_h + label_height)

        # Resize and paste frame
        pil_frame = Image.fromarray(frame).resize((new_w, new_h))
        grid_img.paste(pil_frame, (x, y + label_height))

        # Label with hash (color-coded for duplicates)
        color = hash_colors[h]
        if len([x for x in hashes if x == h]) > 1:
            color = (255, 0, 0)  # Red for duplicates
        draw.text((x + 5, y + 5), f"{label}", fill=(0, 0, 0), font=font)
        draw.text((x + 5, y + 18), f"hash: {h}", fill=color, font=font)

    # Summary
    num_unique = len(unique_hashes)
    num_total = len(frames)
    status = "PASS" if num_unique == num_total else "FAIL - DUPLICATES!"
    status_color = (0, 150, 0) if num_unique == num_total else (255, 0, 0)

    output_path = output_dir / "frame_uniqueness_grid.png"
    grid_img.save(output_path)
    print(f"  Saved frame uniqueness grid to: {output_path}")
    print(f"  Unique frames: {num_unique}/{num_total} - {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify GR00T dataset integrity before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", type=Path, required=True, help="Path to dataset")
    parser.add_argument("--visual", "-v", action="store_true", help="Generate visual verification images")
    parser.add_argument("--save-report", "-s", action="store_true", help="Save JSON report")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory for reports")

    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1

    output_dir = args.output_dir or dataset_path / "verification_report"

    print("=" * 70)
    print("GR00T Dataset Integrity Verification")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print()

    report = {
        "dataset_path": str(dataset_path),
        "checks": {},
        "passed": True,
    }

    # Load metadata
    try:
        info = load_info_json(dataset_path)
        episodes = load_episodes_jsonl(dataset_path)
        print(f"Loaded metadata: {len(episodes)} episodes, {info.get('total_frames', '?')} frames")
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return 1

    # Check 1: Video format
    print("\n[1/5] Checking video format...")
    ok, msg, details = check_video_format(dataset_path, info)
    report["checks"]["video_format"] = {"passed": ok, "message": msg, "details": details}
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        report["passed"] = False

    # Check 2: Timestamp alignment
    print("\n[2/5] Checking timestamp alignment...")
    ok, msg, details = check_timestamp_alignment(dataset_path, episodes, info)
    report["checks"]["timestamp_alignment"] = {"passed": ok, "message": msg, "details": details}
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        report["passed"] = False

    # Check 3: Frame uniqueness
    print("\n[3/5] Checking frame uniqueness across episodes...")
    ok, msg, details = check_frame_uniqueness(dataset_path, episodes)
    report["checks"]["frame_uniqueness"] = {"passed": ok, "message": msg, "details": details}
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        report["passed"] = False

    # Check 4: Baseline MAE
    print("\n[4/5] Computing baseline MAE (action=state)...")
    baseline_mae, details = compute_baseline_mae(dataset_path, episodes)
    report["checks"]["baseline_mae"] = {"value": baseline_mae, "details": details}
    print(f"  Baseline MAE: {baseline_mae:.2f}° (model must significantly beat this)")
    if "mae_per_joint" in details:
        joint_names = details.get("joint_names", [f"joint_{i}" for i in range(6)])
        for name, mae in zip(joint_names, details["mae_per_joint"]):
            print(f"    {name}: {mae:.2f}°")

    # Check 5: Dataloader simulation
    print("\n[5/6] Simulating training dataloader...")
    ok, msg, details = simulate_dataloader(dataset_path)
    report["checks"]["dataloader_simulation"] = {"passed": ok, "message": msg, "details": details}
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        report["passed"] = False

    # Check 6: Full training simulation (shuffling, batching, transforms)
    print("\n[6/6] Simulating full training pipeline (shuffling, batching, transforms)...")
    ok, msg, details = simulate_training_batch(dataset_path, batch_size=4, num_batches=3)
    report["checks"]["training_simulation"] = {"passed": ok, "message": msg, "details": details}
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if "shuffle_test" in details:
        print(f"    Shuffle test: {details['shuffle_test']}")
    if "video_variance_test" in details:
        print(f"    Video variance test: {details['video_variance_test']}")
    if not ok:
        report["passed"] = False

    # Visual report
    if args.visual:
        print("\n[VISUAL] Generating frame comparison image...")
        generate_visual_report(dataset_path, episodes, output_dir)

    # Summary
    print("\n" + "=" * 70)
    if report["passed"]:
        print("VERIFICATION PASSED - Dataset appears valid for training")
    else:
        print("VERIFICATION FAILED - Dataset has issues that will cause training problems")
        print("\nFailed checks:")
        for check_name, check_data in report["checks"].items():
            if isinstance(check_data, dict) and check_data.get("passed") == False:
                print(f"  - {check_name}: {check_data.get('message', 'Unknown error')}")
    print("=" * 70)

    # Save report
    if args.save_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(convert_to_json_serializable(report), f, indent=2)
        print(f"\nReport saved to: {report_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
