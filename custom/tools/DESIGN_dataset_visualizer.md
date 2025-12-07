# Robot Dataset Visualization Tool - Design Document

## Overview

A Gradio-based web application for visualizing GR00T/LeRobot robot arm datasets, inspired by the [LeRobot HuggingFace Space visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

**Reference:** `/home/jrobot/Pictures/visualize_tool.png`

## Features

1. **Multi-camera video playback** - Synchronized head and wrist camera views
2. **Time-series joint plots** - Action and observation.state data for all 6 joints
3. **Prediction comparison** - Overlay predicted actions from model inference
4. **Episode browser** - Navigate between episodes with task descriptions
5. **Interactive timeline** - Frame-by-frame scrubbing with playback controls

## Dataset Format

### Expected Structure
```
dataset/
├── meta/
│   ├── info.json         # Dataset metadata (fps, features, paths)
│   ├── modality.json     # GR00T modality mapping
│   ├── tasks.jsonl       # Task descriptions
│   └── episodes.jsonl    # Episode lengths and task indices
├── videos/
│   ├── observation.images.head/chunk-*/file-000.mp4
│   └── observation.images.left_wrist/chunk-*/file-000.mp4
└── data/
    └── chunk-*/episode_*.parquet
```

### Parquet Schema
| Column | Type | Description |
|--------|------|-------------|
| `action` | float32[6] | Joint commands [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] |
| `observation.state` | float32[6] | Joint states (same order as action) |
| `timestamp` | float32 | Time in seconds |
| `frame_index` | int64 | Frame number within episode |
| `episode_index` | int64 | Episode identifier |
| `task_index` | int64 | Index into tasks.jsonl |

### Video Chunking
Videos are stored in chunks of up to 1000 episodes concatenated per chunk file. To extract frames for episode N:
1. Determine chunk: `chunk_id = episode_index // 1000`
2. Calculate frame offset by summing lengths of previous episodes in chunk
3. Extract frames at `[offset, offset + length)`

---

## Architecture

### File Structure
```
custom/tools/
├── DESIGN_dataset_visualizer.md     # This document
├── visualize_dataset.py              # CLI entry point
└── dataset_visualizer/
    ├── __init__.py
    ├── app.py                        # Gradio application
    ├── data_loader.py                # Dataset/video loading
    ├── plot_utils.py                 # Plotly graph generation
    └── requirements.txt              # Dependencies
```

### Module Responsibilities

#### data_loader.py
```python
# Core data structures
@dataclass
class DatasetMetadata:
    fps: int
    total_episodes: int
    robot_type: str
    joint_names: List[str]
    video_keys: List[str]
    episodes: List[Dict]  # From episodes.jsonl
    tasks: List[str]      # From tasks.jsonl

@dataclass
class EpisodeData:
    episode_index: int
    task: str
    length: int
    timestamps: np.ndarray      # (length,)
    actions: np.ndarray         # (length, 6)
    states: np.ndarray          # (length, 6)

@dataclass
class PredictionData:
    frame_indices: np.ndarray
    predicted_actions: np.ndarray  # (N, 6)
    source_file: str

# Functions
def load_dataset_metadata(dataset_path: str) -> DatasetMetadata
def load_episode_data(dataset_path: str, episode_index: int, metadata: DatasetMetadata) -> EpisodeData
def extract_episode_frames(dataset_path: str, episode_index: int, metadata: DatasetMetadata, camera: str) -> np.ndarray
def load_predictions(file_path: str) -> Optional[PredictionData]
```

#### plot_utils.py
```python
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def generate_joint_plots(
    episode: EpisodeData,
    predictions: Optional[PredictionData],
    current_frame: int,
    show_recorded: bool = True,
    show_predicted: bool = False
) -> go.Figure:
    """
    Generate Plotly figure with 6 subplots (one per joint).

    Lines per subplot:
    - observation.state: solid blue line
    - action: dashed green line
    - predicted: dotted red line (if predictions provided and show_predicted=True)
    - Vertical marker: red line at current_frame position
    """
```

#### app.py
```python
def create_app(default_dataset: str = None) -> gr.Blocks:
    """Create the Gradio application."""
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HEADER                                                                  │
│ Dataset: [/home/jrobot/project/XLeRobot/datasets_groot____] [Load]      │
│ Episode: [▼ Episode 0 - grasp the red cube           ]  (120 episodes)  │
│ Task: "grasp the red cube"                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ VIDEO SECTION                                                           │
│ ┌─────────────────────────┐    ┌─────────────────────────┐              │
│ │                         │    │                         │              │
│ │   observation.images    │    │   observation.images    │              │
│ │        .head            │    │      .left_wrist        │              │
│ │      640x480            │    │        640x480          │              │
│ │                         │    │                         │              │
│ └─────────────────────────┘    └─────────────────────────┘              │
├─────────────────────────────────────────────────────────────────────────┤
│ PLAYBACK CONTROLS                                                       │
│ [⏮][◀][▶ Play][▶][⏭]    Frame: 42 / 180    Time: 1.40s                │
│ [═══════════════════●═══════════════════════════════════════]           │
│ Timeline (0 ────────────────────────────────────────── 180)             │
├─────────────────────────────────────────────────────────────────────────┤
│ DATA OVERLAY OPTIONS                                                    │
│ [✓] Show Recorded (action + state)   [ ] Show Predicted Actions         │
│ Predictions file: [Browse...] (JSON or NPZ format)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ JOINT PLOTS (Plotly interactive)                                        │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ shoulder_pan: 12.5°                                               │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ │ shoulder_lift: -45.2°                                             │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ │ elbow_flex: 30.1°                                                 │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ │ wrist_flex: 55.0°                                                 │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ │ wrist_roll: 2.3°                                                  │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ │ gripper: 0.5                                                      │   │
│ │ ──────────────────────|────────────────────────────               │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│ Legend: ── state (blue)  -- action (green)  ·· predicted (red)          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Video Frame Extraction

Using existing `gr00t/utils/video.py`:

```python
from gr00t.utils.video import get_frames_by_indices

def extract_episode_frames(dataset_path: str, episode_index: int, metadata: DatasetMetadata, camera: str) -> np.ndarray:
    """Extract all frames for an episode from the chunked video."""
    # Calculate chunk and offset
    chunk_id = episode_index // 1000
    offset = calculate_frame_offset(metadata.episodes, episode_index)
    length = metadata.episodes[episode_index]['length']

    # Build video path
    video_key = f"observation.images.{camera}"
    video_path = f"{dataset_path}/videos/{video_key}/chunk-{chunk_id:03d}/file-000.mp4"

    # Extract frames
    indices = list(range(offset, offset + length))
    frames = get_frames_by_indices(video_path, indices, video_backend="pyav")
    return frames  # Shape: (length, H, W, 3)
```

### Synchronization Strategy

Since Gradio's Video component doesn't expose frame-level events, we use:
1. **Slider-based navigation**: Timeline slider controls `current_frame` state
2. **Image component for video**: Display single frame, update on slider change
3. **Cached frames**: Load all episode frames on episode selection
4. **Coordinated updates**: Slider change triggers image + plot + value updates

```python
timeline_slider.change(
    fn=update_display,
    inputs=[state, timeline_slider, show_recorded, show_predicted],
    outputs=[head_image, wrist_image, joint_plots, frame_label, time_label, ...current_values...]
)
```

### Prediction File Format

Support two formats:

**JSON:**
```json
{
    "metadata": {
        "model_checkpoint": "/path/to/checkpoint",
        "episode_index": 0,
        "timestamp": "2025-12-06T10:30:00"
    },
    "predictions": {
        "frame_indices": [0, 1, 2, 3, ...],
        "predicted_actions": [
            [12.5, -45.2, 30.1, 55.0, 2.3, 0.5],
            [12.6, -45.1, 30.2, 55.1, 2.4, 0.6],
            ...
        ]
    }
}
```

**NPZ:**
```python
np.savez('predictions.npz',
    frame_indices=np.array([0, 1, 2, ...]),
    predicted_actions=np.array([[12.5, -45.2, ...], ...])
)
```

---

## Dependencies

```txt
gradio>=4.44.0
plotly>=5.24.0
pyarrow>=19.0.0
av>=12.0.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## Usage

### Basic Usage
```bash
python custom/tools/visualize_dataset.py
# Opens browser at http://localhost:7860
```

### With Arguments
```bash
python custom/tools/visualize_dataset.py \
    --dataset /home/jrobot/project/XLeRobot/datasets_groot \
    --port 7860 \
    --share  # Create public Gradio link
```

### Within Python
```python
from dataset_visualizer.app import create_app

app = create_app(default_dataset="/path/to/dataset")
app.launch()
```

---

## Reference Implementation

### Key Files to Study
- `gr00t/utils/video.py:40-72` - `get_frames_by_indices()`
- `scripts/load_dataset.py:88-169` - `plot_state_action_space()`
- `scripts/load_dataset.py:69-85` - `get_modality_keys()`

### LeRobot Visualizer Source
- Web version: https://github.com/huggingface/lerobot-dataset-visualizer (Next.js)
- Uses Recharts for time-series, HTML5 video for playback

---

## Implementation Checklist

- [ ] Create `data_loader.py` with metadata and episode loading
- [ ] Create `plot_utils.py` with Plotly graph generation
- [ ] Create `app.py` with Gradio layout
- [ ] Create `visualize_dataset.py` CLI launcher
- [ ] Test with dataset at `/home/jrobot/project/XLeRobot/datasets_groot`
- [ ] Add prediction file upload and overlay
- [ ] Polish UI (dark theme, responsive layout)
