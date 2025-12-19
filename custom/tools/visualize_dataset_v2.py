#!/usr/bin/env python3
"""
Robot Dataset Visualizer - LeRobot v2/v3 Format Support

A Gradio-based web application for visualizing robot arm datasets
with synchronized camera views and time-series plots.

Features:
- Support for LeRobot v2 and v3 dataset formats
- Sidebar layout: dataset browser on left, display on right
- Compact view: cameras + charts visible without scrolling
- Synchronized video playback with joint data plots

Usage:
    python visualize_dataset_v2.py [OPTIONS]

Options:
    --dataset PATH      Path to dataset (default: current directory)
    --port INT          Gradio server port (default: 7860)
    --share             Create a public link
    --format v2|v3      Dataset format version (default: v2)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetMetadata:
    """Container for dataset-level metadata."""
    dataset_path: str
    format_version: str  # "v2" or "v3"
    fps: int
    total_episodes: int
    robot_type: str
    joint_names: List[str]
    video_keys: List[str]
    video_path_template: str
    data_path_template: str
    episodes: List[Dict]
    tasks: List[str]


@dataclass
class EpisodeData:
    """Container for a single episode's data."""
    episode_index: int
    task: str
    length: int
    timestamps: np.ndarray
    actions: np.ndarray
    states: np.ndarray


# =============================================================================
# Dataset Loading - v2 and v3 format support
# =============================================================================

def detect_format_version(dataset_path: str) -> str:
    """Auto-detect dataset format version."""
    path = Path(dataset_path)
    info_path = path / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    version = info.get("codebase_version", "v2.0")
    if version.startswith("v2"):
        return "v2"
    elif version.startswith("v3"):
        return "v3"
    else:
        # Default to v2 for older formats
        return "v2"


def load_dataset_metadata(dataset_path: str, format_version: str = "auto") -> DatasetMetadata:
    """Load dataset metadata supporting both v2 and v3 formats."""
    path = Path(dataset_path)
    meta_path = path / "meta"

    # Auto-detect format if needed
    if format_version == "auto":
        format_version = detect_format_version(dataset_path)

    # Load info.json
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)

    # Load tasks.jsonl
    tasks = []
    tasks_file = meta_path / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line.strip())
                    tasks.append(task_data.get("task", ""))

    # Load episodes.jsonl
    episodes = []
    episodes_file = meta_path / "episodes.jsonl"
    if episodes_file.exists():
        with open(episodes_file, "r") as f:
            for line in f:
                if line.strip():
                    ep_data = json.loads(line.strip())
                    episodes.append(ep_data)

    # Extract joint names
    joint_names = []
    if "features" in info and "action" in info["features"]:
        joint_names = info["features"]["action"].get("names", [])
    if not joint_names:
        joint_names = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

    # Extract video keys from features
    video_keys = []
    for key in info.get("features", {}):
        if key.startswith("observation.images."):
            video_keys.append(key)

    # Get path templates
    if format_version == "v2":
        # v2 format: videos/chunk-{chunk}/{video_key}/episode_{index}.mp4
        video_path_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        data_path_template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
    else:
        # v3 format: videos/{video_key}/chunk-{chunk}/file-000.mp4 (concatenated)
        video_path_template = info.get(
            "video_path",
            "videos/{video_key}/chunk-{episode_chunk:03d}/file-000.mp4"
        )
        data_path_template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )

    return DatasetMetadata(
        dataset_path=dataset_path,
        format_version=format_version,
        fps=info.get("fps", 30),
        total_episodes=info.get("total_episodes", len(episodes)),
        robot_type=info.get("robot_type", "unknown"),
        joint_names=joint_names,
        video_keys=video_keys,
        video_path_template=video_path_template,
        data_path_template=data_path_template,
        episodes=episodes,
        tasks=tasks,
    )


def get_video_path(metadata: DatasetMetadata, episode_index: int, video_key: str) -> Path:
    """Get the video file path for an episode and camera."""
    path = Path(metadata.dataset_path)
    chunk_id = episode_index // 1000

    # Format the template
    video_path = metadata.video_path_template.format(
        episode_chunk=chunk_id,
        video_key=video_key,
        episode_index=episode_index,
    )

    return path / video_path


def load_episode_data(metadata: DatasetMetadata, episode_index: int) -> EpisodeData:
    """Load action and state data for a specific episode."""
    path = Path(metadata.dataset_path)
    chunk_id = episode_index // 1000

    # Format parquet path
    parquet_path = path / metadata.data_path_template.format(
        episode_chunk=chunk_id,
        episode_index=episode_index,
    )

    if pq is None:
        raise ImportError("pyarrow is required for loading parquet files")

    # Read parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Filter by episode if needed
    if "episode_index" in df.columns:
        ep_col = df["episode_index"]
        if hasattr(ep_col.iloc[0], '__len__'):
            df = df[[e[0] == episode_index if len(e) > 0 else False for e in ep_col]]
        else:
            df = df[ep_col == episode_index]
        df = df.reset_index(drop=True)

    # Extract data
    actions = np.array(df["action"].tolist())
    states = np.array(df["observation.state"].tolist())

    # Get timestamps
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].values
        if isinstance(timestamps[0], (list, np.ndarray)):
            timestamps = np.array([t[0] if len(t) > 0 else 0.0 for t in timestamps])
    else:
        timestamps = np.arange(len(actions)) / metadata.fps

    # Get task
    task = ""
    for ep in metadata.episodes:
        if ep.get("episode_index") == episode_index:
            task_idx = ep.get("task_index", 0)
            if task_idx < len(metadata.tasks):
                task = metadata.tasks[task_idx]
            break

    return EpisodeData(
        episode_index=episode_index,
        task=task,
        length=len(actions),
        timestamps=timestamps.astype(np.float32),
        actions=actions.astype(np.float32),
        states=states.astype(np.float32),
    )


def extract_video_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    """Extract a single frame from video."""
    if not video_path.exists():
        return None

    # Try av library first (handles AV1 codec)
    if HAS_AV:
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_index:
                    result = frame.to_ndarray(format="rgb24")
                    container.close()
                    return result
                if i > frame_index:
                    break

            container.close()
        except Exception as e:
            print(f"av failed for {video_path}: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"cv2 failed for {video_path}: {e}")

    return None


def extract_all_frames(video_path: Path) -> Optional[np.ndarray]:
    """Extract all frames from a video file."""
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None

    # Try av library first (handles AV1 codec)
    if HAS_AV:
        try:
            container = av.open(str(video_path))
            frames = []

            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

            container.close()

            if frames:
                return np.array(frames)
        except Exception as e:
            print(f"av failed for {video_path}: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

            if frames:
                return np.array(frames)
        except Exception as e:
            print(f"cv2 failed for {video_path}: {e}")

    return None


# =============================================================================
# Plotting
# =============================================================================

JOINT_DISPLAY_NAMES = [
    "Shoulder Pan", "Shoulder Lift", "Elbow Flex",
    "Wrist Flex", "Wrist Roll", "Gripper"
]

COLORS = {
    "state": "#3b82f6",      # Blue
    "action": "#22c55e",     # Green
    "marker": "#f97316",     # Orange
}


def generate_compact_plots(
    episode: EpisodeData,
    current_frame: int = 0,
    height: int = 400,
) -> go.Figure:
    """Generate compact 2x3 joint plots for single-page view."""

    num_joints = min(len(JOINT_DISPLAY_NAMES), episode.states.shape[1])

    # Create 2 rows x 3 cols subplot
    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=JOINT_DISPLAY_NAMES[:num_joints],
    )

    x_axis = np.arange(episode.length)

    for i in range(num_joints):
        row = i // 3 + 1
        col = i % 3 + 1

        # State line (solid blue)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.states[:, i],
                mode="lines",
                name="State",
                line=dict(color=COLORS["state"], width=1.5),
                legendgroup="state",
                showlegend=(i == 0),
            ),
            row=row, col=col,
        )

        # Action line (dashed green)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.actions[:, i],
                mode="lines",
                name="Action",
                line=dict(color=COLORS["action"], width=1.5, dash="dash"),
                legendgroup="action",
                showlegend=(i == 0),
            ),
            row=row, col=col,
        )

        # Current frame marker
        fig.add_vline(
            x=current_frame,
            line_dash="solid",
            line_color=COLORS["marker"],
            line_width=2,
            row=row, col=col,
        )

    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_dark",
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
    )

    # Smaller subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = 10

    return fig


# =============================================================================
# Gradio Application
# =============================================================================

class VisualizerState:
    """Application state."""
    def __init__(self):
        self.metadata: Optional[DatasetMetadata] = None
        self.episode_data: Optional[EpisodeData] = None
        self.video_frames: Dict[str, np.ndarray] = {}
        self.current_frame: int = 0


def create_app(default_dataset: str = ".", default_format: str = "v2") -> gr.Blocks:
    """Create the Gradio application with sidebar layout."""

    state = VisualizerState()

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def load_dataset(dataset_path: str, format_version: str) -> Tuple:
        """Load dataset and return episode list."""
        try:
            fmt = format_version if format_version != "auto" else "auto"
            state.metadata = load_dataset_metadata(dataset_path, fmt)

            # Build episode choices
            choices = []
            for ep in state.metadata.episodes:
                idx = ep.get("episode_index", len(choices))
                task_idx = ep.get("task_index", 0)
                task = state.metadata.tasks[task_idx] if task_idx < len(state.metadata.tasks) else ""
                label = f"Ep {idx}: {task[:30]}..." if len(task) > 30 else f"Ep {idx}: {task}" if task else f"Episode {idx}"
                choices.append((label, idx))

            status = f"✓ Loaded {state.metadata.total_episodes} episodes ({state.metadata.format_version})"
            info = f"Robot: {state.metadata.robot_type} | FPS: {state.metadata.fps} | Cameras: {len(state.metadata.video_keys)}"

            return (
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                status,
                info,
            )
        except Exception as e:
            return (
                gr.Dropdown(choices=[]),
                f"✗ Error: {str(e)[:100]}",
                "",
            )

    def load_episode(episode_index: int) -> Tuple:
        """Load episode data and first frame."""
        if state.metadata is None or episode_index is None:
            return (None, None, 0, gr.Slider(maximum=1), None, "No episode", "")

        try:
            # Load data
            state.episode_data = load_episode_data(state.metadata, episode_index)
            state.current_frame = 0

            # Load video frames for each camera
            state.video_frames = {}
            for video_key in state.metadata.video_keys:
                video_path = get_video_path(state.metadata, episode_index, video_key)
                frames = extract_all_frames(video_path)
                if frames is not None:
                    state.video_frames[video_key] = frames

            # Get first frames
            cam1_frame = None
            cam2_frame = None
            video_keys = list(state.video_frames.keys())
            if len(video_keys) >= 1:
                cam1_frame = state.video_frames[video_keys[0]][0]
            if len(video_keys) >= 2:
                cam2_frame = state.video_frames[video_keys[1]][0]

            # Generate plot
            plot = generate_compact_plots(state.episode_data, current_frame=0)

            frame_info = f"Frame 0/{state.episode_data.length-1}"
            task_info = state.episode_data.task

            return (
                cam1_frame,
                cam2_frame,
                0,
                gr.Slider(maximum=state.episode_data.length - 1),
                plot,
                frame_info,
                task_info,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (None, None, 0, gr.Slider(maximum=1), None, f"Error: {e}", "")

    def update_frame(frame_idx: int) -> Tuple:
        """Update display for current frame."""
        if state.episode_data is None:
            return (None, None, None, "No data")

        frame_idx = int(frame_idx)
        state.current_frame = frame_idx

        # Get frames
        cam1_frame = None
        cam2_frame = None
        video_keys = list(state.video_frames.keys())

        if len(video_keys) >= 1:
            frames = state.video_frames[video_keys[0]]
            if frame_idx < len(frames):
                cam1_frame = frames[frame_idx]

        if len(video_keys) >= 2:
            frames = state.video_frames[video_keys[1]]
            if frame_idx < len(frames):
                cam2_frame = frames[frame_idx]

        # Generate plot
        plot = generate_compact_plots(state.episode_data, current_frame=frame_idx)

        frame_info = f"Frame {frame_idx}/{state.episode_data.length-1}"

        return (cam1_frame, cam2_frame, plot, frame_info)

    def step_frame(current: int, delta: int) -> int:
        """Step frame by delta."""
        if state.episode_data is None:
            return 0
        new_frame = max(0, min(int(current) + delta, state.episode_data.length - 1))
        return new_frame

    # -------------------------------------------------------------------------
    # Build UI
    # -------------------------------------------------------------------------

    with gr.Blocks(
        title="LeRobot Dataset Visualizer",
        fill_height=True,
    ) as app:

        gr.Markdown("## 🤖 LeRobot Dataset Visualizer")

        with gr.Row():
            # -----------------------------------------------------------------
            # Left Sidebar: Dataset Browser
            # -----------------------------------------------------------------
            with gr.Column(scale=1, min_width=280, elem_classes=["sidebar"]):
                gr.Markdown("### 📁 Dataset")

                dataset_path = gr.Textbox(
                    label="Path",
                    value=default_dataset,
                    placeholder="/path/to/dataset",
                )

                format_dropdown = gr.Dropdown(
                    label="Format",
                    choices=[("LeRobot v2", "v2"), ("LeRobot v3", "v3"), ("Auto-detect", "auto")],
                    value=default_format,
                )

                load_btn = gr.Button("📂 Load Dataset", variant="primary")

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

                info_text = gr.Textbox(
                    label="Info",
                    interactive=False,
                    lines=1,
                )

                gr.Markdown("### 📋 Episodes")

                episode_dropdown = gr.Dropdown(
                    label="Select Episode",
                    choices=[],
                    interactive=True,
                )

                task_display = gr.Textbox(
                    label="Task",
                    interactive=False,
                    lines=2,
                )

            # -----------------------------------------------------------------
            # Right Main Display: Cameras + Timeline + Charts
            # -----------------------------------------------------------------
            with gr.Column(scale=3, elem_classes=["main-display"]):

                # Camera views side by side
                with gr.Row():
                    with gr.Column(scale=1):
                        cam1_image = gr.Image(
                            label="Camera 1 (Head)",
                            type="numpy",
                            height=240,
                        )
                    with gr.Column(scale=1):
                        cam2_image = gr.Image(
                            label="Camera 2 (Wrist)",
                            type="numpy",
                            height=240,
                        )

                # Timeline controls
                with gr.Row():
                    prev_10_btn = gr.Button("⏪ -10", scale=1)
                    prev_btn = gr.Button("◀ -1", scale=1)
                    frame_info = gr.Textbox(
                        value="Frame 0/0",
                        interactive=False,
                        scale=2,
                        show_label=False,
                    )
                    next_btn = gr.Button("+1 ▶", scale=1)
                    next_10_btn = gr.Button("+10 ⏩", scale=1)

                timeline_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Timeline",
                    interactive=True,
                )

                # Joint plots - compact 2x3 grid
                joint_plots = gr.Plot(
                    label="Joint States & Actions",
                    elem_classes=["compact-plot"],
                )

        # ---------------------------------------------------------------------
        # Event Bindings
        # ---------------------------------------------------------------------

        load_btn.click(
            fn=load_dataset,
            inputs=[dataset_path, format_dropdown],
            outputs=[episode_dropdown, status_text, info_text],
        )

        episode_dropdown.change(
            fn=load_episode,
            inputs=[episode_dropdown],
            outputs=[
                cam1_image, cam2_image,
                timeline_slider, timeline_slider,
                joint_plots, frame_info, task_display,
            ],
        )

        timeline_slider.change(
            fn=update_frame,
            inputs=[timeline_slider],
            outputs=[cam1_image, cam2_image, joint_plots, frame_info],
        )

        # Navigation buttons
        prev_10_btn.click(
            fn=lambda x: step_frame(x, -10),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        prev_btn.click(
            fn=lambda x: step_frame(x, -1),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        next_btn.click(
            fn=lambda x: step_frame(x, 1),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        next_10_btn.click(
            fn=lambda x: step_frame(x, 10),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LeRobot Dataset Visualizer")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=".",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["v2", "v3", "auto"],
        default="v2",
        help="Dataset format version (default: v2)",
    )
    args = parser.parse_args()

    app = create_app(default_dataset=args.dataset, default_format=args.format)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
