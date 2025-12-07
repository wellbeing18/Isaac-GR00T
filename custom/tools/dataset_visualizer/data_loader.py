"""
Data loading utilities for the dataset visualizer.

Handles loading metadata, episode data, video frames, and predictions
from GR00T/LeRobot format datasets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


@dataclass
class DatasetMetadata:
    """Container for dataset-level metadata."""
    dataset_path: str
    fps: int
    total_episodes: int
    robot_type: str
    joint_names: List[str]
    video_keys: List[str]  # e.g., ["observation.images.head", "observation.images.left_wrist"]
    episodes: List[Dict]   # From episodes.jsonl: [{"episode_index": 0, "length": 180, "task_index": 0}, ...]
    tasks: List[str]       # From tasks.jsonl: ["grasp the red cube", ...]
    modality: Dict         # From modality.json


@dataclass
class EpisodeData:
    """Container for a single episode's data."""
    episode_index: int
    task: str
    length: int
    timestamps: np.ndarray      # Shape: (length,)
    actions: np.ndarray         # Shape: (length, 6)
    states: np.ndarray          # Shape: (length, 6)


@dataclass
class PredictionData:
    """Container for model prediction data."""
    frame_indices: np.ndarray       # Shape: (N,)
    predicted_actions: np.ndarray   # Shape: (N, 6)
    source_file: str


def load_dataset_metadata(dataset_path: str) -> DatasetMetadata:
    """
    Load dataset metadata from the meta/ directory.

    Args:
        dataset_path: Path to the dataset root directory

    Returns:
        DatasetMetadata object containing all metadata
    """
    path = Path(dataset_path)
    meta_path = path / "meta"

    # Load info.json
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)

    # Load modality.json
    with open(meta_path / "modality.json", "r") as f:
        modality = json.load(f)

    # Load tasks.jsonl
    tasks = []
    with open(meta_path / "tasks.jsonl", "r") as f:
        for line in f:
            task_data = json.loads(line.strip())
            tasks.append(task_data.get("task", ""))

    # Load episodes.jsonl
    episodes = []
    with open(meta_path / "episodes.jsonl", "r") as f:
        for line in f:
            ep_data = json.loads(line.strip())
            episodes.append(ep_data)

    # Extract joint names from info.json features
    joint_names = []
    if "features" in info and "action" in info["features"]:
        joint_names = info["features"]["action"].get("names", [])
    if not joint_names:
        # Default joint names for SO-101
        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ]

    # Extract video keys from modality.json
    video_keys = []
    if "video" in modality:
        for key, val in modality["video"].items():
            original_key = val.get("original_key", f"observation.images.{key}")
            video_keys.append(original_key)

    if not video_keys:
        # Try to find from info.json features
        for key in info.get("features", {}):
            if key.startswith("observation.images."):
                video_keys.append(key)

    return DatasetMetadata(
        dataset_path=dataset_path,
        fps=info.get("fps", 30),
        total_episodes=info.get("total_episodes", len(episodes)),
        robot_type=info.get("robot_type", "unknown"),
        joint_names=joint_names,
        video_keys=video_keys,
        episodes=episodes,
        tasks=tasks,
        modality=modality,
    )


def get_episode_info(metadata: DatasetMetadata, episode_index: int) -> Tuple[int, int, str]:
    """
    Get episode information.

    Returns:
        Tuple of (length, task_index, task_description)
    """
    # Find episode in metadata
    for ep in metadata.episodes:
        if ep["episode_index"] == episode_index:
            length = ep["length"]
            task_index = ep.get("task_index", 0)
            task = metadata.tasks[task_index] if task_index < len(metadata.tasks) else ""
            return length, task_index, task

    raise ValueError(f"Episode {episode_index} not found in metadata")


def calculate_frame_offset(episodes: List[Dict], episode_index: int) -> int:
    """
    Calculate the frame offset for an episode within its chunk.

    Videos are chunked by 1000 episodes. Each chunk contains all frames
    from episodes [chunk_id * 1000, (chunk_id + 1) * 1000) concatenated.

    Args:
        episodes: List of episode metadata dicts
        episode_index: The episode to find offset for

    Returns:
        Frame offset within the chunk video file
    """
    chunk_id = episode_index // 1000
    chunk_start = chunk_id * 1000

    offset = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        # Only count episodes in the same chunk
        if ep_idx // 1000 == chunk_id:
            if ep_idx < episode_index:
                offset += ep["length"]
            elif ep_idx == episode_index:
                break

    return offset


def load_episode_data(dataset_path: str, episode_index: int, metadata: DatasetMetadata) -> EpisodeData:
    """
    Load action and state data for a specific episode from parquet.

    Args:
        dataset_path: Path to the dataset root directory
        episode_index: Index of the episode to load
        metadata: Dataset metadata

    Returns:
        EpisodeData containing actions, states, and timestamps
    """
    path = Path(dataset_path)

    # Calculate chunk from episode index
    chunk_id = episode_index // 1000
    parquet_path = path / f"data/chunk-{chunk_id:03d}/episode_{episode_index:06d}.parquet"

    # Read parquet file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Filter by episode_index in case file contains multiple episodes
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode_index].reset_index(drop=True)

    # Extract data
    actions = np.array(df["action"].tolist())  # Convert list of arrays to 2D array
    states = np.array(df["observation.state"].tolist())

    # Get timestamps - handle both scalar and array formats
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].values
        if isinstance(timestamps[0], (list, np.ndarray)):
            timestamps = np.array([t[0] if len(t) > 0 else 0.0 for t in timestamps])
    else:
        # Generate timestamps from frame index and fps
        timestamps = df["frame_index"].values / metadata.fps

    # Get episode info
    length, task_index, task = get_episode_info(metadata, episode_index)

    return EpisodeData(
        episode_index=episode_index,
        task=task,
        length=len(actions),
        timestamps=timestamps.astype(np.float32),
        actions=actions.astype(np.float32),
        states=states.astype(np.float32),
    )


def extract_episode_frames(
    dataset_path: str,
    episode_index: int,
    metadata: DatasetMetadata,
    camera_key: str,
    video_backend: str = "pyav"
) -> np.ndarray:
    """
    Extract all video frames for an episode.

    Args:
        dataset_path: Path to the dataset root directory
        episode_index: Index of the episode
        metadata: Dataset metadata
        camera_key: Camera key (e.g., "observation.images.head")
        video_backend: Video backend to use ("pyav", "opencv", "decord")

    Returns:
        numpy array of frames, shape (length, H, W, 3)
    """
    path = Path(dataset_path)

    # Calculate chunk and offset
    chunk_id = episode_index // 1000
    offset = calculate_frame_offset(metadata.episodes, episode_index)
    length, _, _ = get_episode_info(metadata, episode_index)

    # Build video path
    video_path = path / f"videos/{camera_key}/chunk-{chunk_id:03d}/file-000.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract frames using the appropriate backend
    indices = list(range(offset, offset + length))

    if video_backend == "pyav":
        frames = _extract_frames_pyav(str(video_path), indices)
    elif video_backend == "opencv":
        frames = _extract_frames_opencv(str(video_path), indices)
    else:
        # Try to import gr00t video utilities
        try:
            from gr00t.utils.video import get_frames_by_indices
            frames = get_frames_by_indices(str(video_path), indices, video_backend=video_backend)
        except ImportError:
            frames = _extract_frames_pyav(str(video_path), indices)

    return frames


def _extract_frames_pyav(video_path: str, indices: List[int]) -> np.ndarray:
    """Extract frames using PyAV."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Build set of target indices for fast lookup
    target_indices = set(indices)
    max_index = max(indices)

    frames = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in target_indices:
            frames[i] = frame.to_ndarray(format="rgb24")
        if i >= max_index:
            break

    container.close()

    # Return frames in order
    return np.array([frames[i] for i in indices])


def _extract_frames_opencv(video_path: str, indices: List[int]) -> np.ndarray:
    """Extract frames using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            raise ValueError(f"Failed to read frame at index {idx}")

    cap.release()
    return np.array(frames)


def load_predictions(file_path: str) -> Optional[PredictionData]:
    """
    Load prediction data from a JSON or NPZ file.

    Expected JSON format:
    {
        "predictions": {
            "frame_indices": [0, 1, 2, ...],
            "predicted_actions": [[...], [...], ...]
        }
    }
    or simplified:
    {
        "frame_indices": [0, 1, 2, ...],
        "predicted_actions": [[...], [...], ...]
    }

    Expected NPZ format:
    - frame_indices: array of shape (N,)
    - predicted_actions: array of shape (N, 6)

    Args:
        file_path: Path to prediction file

    Returns:
        PredictionData or None if loading fails
    """
    if file_path is None:
        return None

    file_path = str(file_path)

    try:
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)

            # Handle nested format
            if "predictions" in data:
                data = data["predictions"]

            return PredictionData(
                frame_indices=np.array(data["frame_indices"]),
                predicted_actions=np.array(data["predicted_actions"]),
                source_file=file_path,
            )

        elif file_path.endswith(".npz"):
            data = np.load(file_path)
            return PredictionData(
                frame_indices=data["frame_indices"],
                predicted_actions=data["predicted_actions"],
                source_file=file_path,
            )

        else:
            print(f"Unsupported prediction file format: {file_path}")
            return None

    except Exception as e:
        print(f"Error loading predictions from {file_path}: {e}")
        return None


def get_episode_list(metadata: DatasetMetadata) -> List[Tuple[int, str]]:
    """
    Get a list of (episode_index, display_label) tuples for UI dropdown.

    Args:
        metadata: Dataset metadata

    Returns:
        List of tuples: [(0, "Episode 0 - grasp the red cube"), ...]
    """
    result = []
    for ep in metadata.episodes:
        ep_idx = ep["episode_index"]
        task_idx = ep.get("task_index", 0)
        task = metadata.tasks[task_idx] if task_idx < len(metadata.tasks) else ""
        label = f"Episode {ep_idx} - {task}" if task else f"Episode {ep_idx}"
        result.append((ep_idx, label))

    return result
