"""
Gradio application for the dataset visualizer.

This module creates the web interface for visualizing robot arm datasets
with synchronized camera views and time-series plots.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from .data_loader import (
    DatasetMetadata,
    EpisodeData,
    PredictionData,
    load_dataset_metadata,
    load_episode_data,
    extract_episode_frames,
    load_predictions,
    get_episode_list,
)
from .plot_utils import (
    generate_joint_plots,
    get_current_values,
    JOINT_NAMES,
)


# Default dataset path
DEFAULT_DATASET = "/home/jrobot/project/XLeRobot/datasets_groot"


class VisualizerState:
    """Application state container."""

    def __init__(self):
        self.metadata: Optional[DatasetMetadata] = None
        self.episode_data: Optional[EpisodeData] = None
        self.predictions: Optional[PredictionData] = None
        self.frames_cache: Dict[str, np.ndarray] = {}  # {camera_key: frames}
        self.current_frame: int = 0


def create_app(default_dataset: str = DEFAULT_DATASET) -> gr.Blocks:
    """
    Create the Gradio application.

    Args:
        default_dataset: Default path to load

    Returns:
        Gradio Blocks application
    """
    # Application state
    state = VisualizerState()

    def load_dataset(dataset_path: str) -> Tuple[gr.Dropdown, str, str]:
        """Load dataset and return updated UI components."""
        try:
            state.metadata = load_dataset_metadata(dataset_path)
            episodes = get_episode_list(state.metadata)
            choices = [(label, idx) for idx, label in episodes]

            return (
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                f"Loaded {state.metadata.total_episodes} episodes",
                "",
            )
        except Exception as e:
            return (
                gr.Dropdown(choices=[]),
                f"Error loading dataset: {e}",
                "",
            )

    def load_episode(episode_index: int) -> Tuple:
        """Load episode data and frames."""
        if state.metadata is None or episode_index is None:
            return (None, None, 0, gr.Slider(maximum=1), None, "No episode loaded", "")

        try:
            # Load episode data
            state.episode_data = load_episode_data(
                state.metadata.dataset_path,
                episode_index,
                state.metadata,
            )

            # Load video frames
            state.frames_cache = {}
            for video_key in state.metadata.video_keys:
                try:
                    frames = extract_episode_frames(
                        state.metadata.dataset_path,
                        episode_index,
                        state.metadata,
                        video_key,
                    )
                    state.frames_cache[video_key] = frames
                except Exception as e:
                    print(f"Error loading frames for {video_key}: {e}")

            state.current_frame = 0

            # Get first frames for each camera
            head_frame = None
            wrist_frame = None
            for key, frames in state.frames_cache.items():
                if "head" in key:
                    head_frame = frames[0]
                elif "wrist" in key:
                    wrist_frame = frames[0]

            # Generate initial plot
            plot = generate_joint_plots(
                state.episode_data,
                state.predictions,
                current_frame=0,
                show_recorded=True,
                show_predicted=False,
            )

            return (
                head_frame,
                wrist_frame,
                0,  # slider value
                gr.Slider(maximum=state.episode_data.length - 1),
                plot,
                f"Frame: 0 / {state.episode_data.length}",
                state.episode_data.task,
            )
        except Exception as e:
            print(f"Error loading episode: {e}")
            return (None, None, 0, gr.Slider(maximum=1), None, f"Error: {e}", "")

    def update_frame(
        frame_idx: int,
        show_recorded: bool,
        show_predicted: bool,
    ) -> Tuple:
        """Update display based on current frame."""
        if state.episode_data is None:
            return (None, None, None, "", "")

        frame_idx = int(frame_idx)
        state.current_frame = frame_idx

        # Get frames
        head_frame = None
        wrist_frame = None
        for key, frames in state.frames_cache.items():
            if frame_idx < len(frames):
                if "head" in key:
                    head_frame = frames[frame_idx]
                elif "wrist" in key:
                    wrist_frame = frames[frame_idx]

        # Generate plot
        plot = generate_joint_plots(
            state.episode_data,
            state.predictions if show_predicted else None,
            current_frame=frame_idx,
            show_recorded=show_recorded,
            show_predicted=show_predicted,
        )

        # Frame info
        time_val = state.episode_data.timestamps[frame_idx]
        frame_info = f"Frame: {frame_idx} / {state.episode_data.length}  |  Time: {time_val:.2f}s"

        # Current values
        values = get_current_values(state.episode_data, frame_idx, state.predictions)
        values_text = format_values(values, show_predicted)

        return (head_frame, wrist_frame, plot, frame_info, values_text)

    def format_values(values: dict, show_predicted: bool) -> str:
        """Format current values for display."""
        lines = []
        for joint in JOINT_NAMES:
            state_val = values["states"][joint]
            action_val = values["actions"][joint]
            line = f"{joint}: {state_val:.1f} / {action_val:.1f}"
            if show_predicted and joint in values.get("predicted", {}):
                pred_val = values["predicted"][joint]
                line += f" / {pred_val:.1f}"
            lines.append(line)
        return "\n".join(lines)

    def step_frame(current: int, delta: int) -> int:
        """Step frame by delta."""
        if state.episode_data is None:
            return 0
        new_frame = max(0, min(current + delta, state.episode_data.length - 1))
        return new_frame

    def load_prediction_file(file) -> str:
        """Load prediction file."""
        if file is None:
            state.predictions = None
            return "No predictions loaded"

        state.predictions = load_predictions(file.name)
        if state.predictions is not None:
            return f"Loaded {len(state.predictions.frame_indices)} predictions"
        return "Failed to load predictions"

    # Build UI
    with gr.Blocks(
        title="Robot Dataset Visualizer",
    ) as app:
        gr.Markdown("# Robot Dataset Visualizer")

        # Header Section
        with gr.Row():
            dataset_path = gr.Textbox(
                label="Dataset Path",
                value=default_dataset,
                scale=4,
            )
            load_btn = gr.Button("Load Dataset", variant="primary", scale=1)

        with gr.Row():
            episode_dropdown = gr.Dropdown(
                label="Episode",
                choices=[],
                interactive=True,
                scale=2,
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                scale=2,
            )

        with gr.Row():
            task_display = gr.Textbox(
                label="Task",
                interactive=False,
                scale=4,
            )

        # Video Section
        with gr.Row():
            with gr.Column(scale=1):
                head_image = gr.Image(
                    label="Head Camera",
                    type="numpy",
                    height=360,
                )
            with gr.Column(scale=1):
                wrist_image = gr.Image(
                    label="Wrist Camera",
                    type="numpy",
                    height=360,
                )

        # Playback Controls
        with gr.Row():
            prev_10_btn = gr.Button("<<", scale=1)
            prev_btn = gr.Button("<", scale=1)
            play_btn = gr.Button("Play", variant="primary", scale=2)
            next_btn = gr.Button(">", scale=1)
            next_10_btn = gr.Button(">>", scale=1)
            frame_info = gr.Textbox(
                value="Frame: 0 / 0",
                interactive=False,
                scale=3,
                elem_classes=["frame-info"],
            )

        with gr.Row():
            timeline_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                label="Timeline",
                interactive=True,
            )

        # Overlay Options
        with gr.Row():
            show_recorded = gr.Checkbox(
                label="Show Recorded (state + action)",
                value=True,
            )
            show_predicted = gr.Checkbox(
                label="Show Predicted Actions",
                value=False,
            )
            prediction_file = gr.File(
                label="Predictions File (JSON/NPZ)",
                file_types=[".json", ".npz"],
            )
            prediction_status = gr.Textbox(
                label="Prediction Status",
                value="No predictions loaded",
                interactive=False,
                scale=1,
            )

        # Joint Plots
        with gr.Row():
            with gr.Column(scale=3):
                joint_plots = gr.Plot(label="Joint Data")
            with gr.Column(scale=1):
                values_display = gr.Textbox(
                    label="Current Values (state / action / pred)",
                    interactive=False,
                    lines=8,
                    elem_classes=["joint-values"],
                )

        # Event handlers
        load_btn.click(
            fn=load_dataset,
            inputs=[dataset_path],
            outputs=[episode_dropdown, status_text, task_display],
        )

        episode_dropdown.change(
            fn=load_episode,
            inputs=[episode_dropdown],
            outputs=[
                head_image,
                wrist_image,
                timeline_slider,
                timeline_slider,
                joint_plots,
                frame_info,
                task_display,
            ],
        )

        timeline_slider.change(
            fn=update_frame,
            inputs=[timeline_slider, show_recorded, show_predicted],
            outputs=[head_image, wrist_image, joint_plots, frame_info, values_display],
        )

        show_recorded.change(
            fn=update_frame,
            inputs=[timeline_slider, show_recorded, show_predicted],
            outputs=[head_image, wrist_image, joint_plots, frame_info, values_display],
        )

        show_predicted.change(
            fn=update_frame,
            inputs=[timeline_slider, show_recorded, show_predicted],
            outputs=[head_image, wrist_image, joint_plots, frame_info, values_display],
        )

        prediction_file.change(
            fn=load_prediction_file,
            inputs=[prediction_file],
            outputs=[prediction_status],
        ).then(
            fn=update_frame,
            inputs=[timeline_slider, show_recorded, show_predicted],
            outputs=[head_image, wrist_image, joint_plots, frame_info, values_display],
        )

        # Playback buttons
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

        # Auto-play functionality with synchronized video and plots
        play_state = gr.State({"playing": False, "speed": 1.0})

        def toggle_play(play_info, current_frame, show_rec, show_pred):
            """Toggle play/pause state."""
            is_playing = not play_info["playing"]
            play_info["playing"] = is_playing
            btn_text = "⏸ Pause" if is_playing else "▶ Play"
            return play_info, btn_text

        def advance_frame(play_info, current_frame, show_rec, show_pred):
            """Advance frame during playback. Returns updated frame and UI."""
            if not play_info["playing"] or state.episode_data is None:
                # Not playing, return current state without re-rendering
                # Use gr.update() to skip unnecessary updates
                return (
                    gr.update(),  # timeline_slider - no change
                    gr.update(),  # head_image - no change
                    gr.update(),  # wrist_image - no change
                    gr.update(),  # joint_plots - no change
                    gr.update(),  # frame_info - no change
                    gr.update(),  # values_display - no change
                    play_info,
                )

            # Advance frame
            new_frame = int(current_frame) + 1

            # Check if we've reached the end
            if new_frame >= state.episode_data.length:
                # Stop playback and reset to beginning
                play_info["playing"] = False
                new_frame = 0

            # Get updated visuals
            head_frame, wrist_frame, plot, frame_info_text, values_text = update_frame(
                new_frame, show_rec, show_pred
            )

            return (
                new_frame,
                head_frame,
                wrist_frame,
                plot,
                frame_info_text,
                values_text,
                play_info,
            )

        # Play button toggles state
        play_btn.click(
            fn=toggle_play,
            inputs=[play_state, timeline_slider, show_recorded, show_predicted],
            outputs=[play_state, play_btn],
        )

        # Timer-based playback using Gradio's every() for continuous updates
        # This creates a periodic callback that advances frames when playing
        timer = gr.Timer(value=0.033, active=True)  # ~30 FPS

        timer.tick(
            fn=advance_frame,
            inputs=[play_state, timeline_slider, show_recorded, show_predicted],
            outputs=[
                timeline_slider,
                head_image,
                wrist_image,
                joint_plots,
                frame_info,
                values_display,
                play_state,
            ],
        )

        # Update play button text when playback stops (e.g., end of episode)
        def update_play_btn_text(play_info):
            return "⏸ Pause" if play_info["playing"] else "▶ Play"

        play_state.change(
            fn=update_play_btn_text,
            inputs=[play_state],
            outputs=[play_btn],
        )

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Dataset Visualizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to GR00T dataset",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link",
    )
    args = parser.parse_args()

    app = create_app(default_dataset=args.dataset)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
