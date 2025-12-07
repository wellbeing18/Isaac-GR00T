"""
Plotting utilities for the dataset visualizer.

Uses Plotly to generate interactive time-series plots of robot joint data.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import EpisodeData, PredictionData

# Joint names for display (without .pos suffix)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

# Colors matching the reference screenshot
COLORS = {
    "state": "#3b82f6",      # Blue
    "action": "#22c55e",     # Green
    "predicted": "#ef4444",  # Red
    "marker": "#f97316",     # Orange for timeline marker
}


def generate_joint_plots(
    episode: EpisodeData,
    predictions: Optional[PredictionData] = None,
    current_frame: int = 0,
    show_recorded: bool = True,
    show_predicted: bool = False,
    height_per_joint: int = 120,
) -> go.Figure:
    """
    Generate a Plotly figure with 6 subplots (one per joint).

    Each subplot shows:
    - observation.state: solid blue line
    - action: dashed green line
    - predicted: dotted red line (if provided and show_predicted=True)
    - Vertical marker: orange line at current frame position

    Args:
        episode: Episode data containing actions, states, and timestamps
        predictions: Optional prediction data for overlay
        current_frame: Current frame index for the timeline marker
        show_recorded: Whether to show recorded actions and states
        show_predicted: Whether to show predicted actions
        height_per_joint: Height in pixels for each joint subplot

    Returns:
        Plotly Figure object
    """
    num_joints = len(JOINT_NAMES)

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=num_joints,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"{name}" for name in JOINT_NAMES],
    )

    # Use frame index as x-axis (more intuitive for scrubbing)
    x_axis = np.arange(episode.length)
    current_x = current_frame

    # Add traces for each joint
    for i, joint_name in enumerate(JOINT_NAMES):
        row = i + 1

        if show_recorded:
            # State (solid line)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=episode.states[:, i],
                    mode="lines",
                    name="state",
                    line=dict(color=COLORS["state"], width=2),
                    legendgroup="state",
                    showlegend=(i == 0),
                    hovertemplate=f"{joint_name} state: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

            # Action (dashed line)
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=episode.actions[:, i],
                    mode="lines",
                    name="action",
                    line=dict(color=COLORS["action"], width=2, dash="dash"),
                    legendgroup="action",
                    showlegend=(i == 0),
                    hovertemplate=f"{joint_name} action: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

        # Predicted actions (dotted line)
        if show_predicted and predictions is not None:
            pred_x = predictions.frame_indices
            pred_y = predictions.predicted_actions[:, i]

            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=pred_y,
                    mode="lines+markers",
                    name="predicted",
                    line=dict(color=COLORS["predicted"], width=2, dash="dot"),
                    marker=dict(size=4),
                    legendgroup="predicted",
                    showlegend=(i == 0),
                    hovertemplate=f"{joint_name} predicted: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

        # Vertical marker for current frame
        fig.add_vline(
            x=current_x,
            line_dash="solid",
            line_color=COLORS["marker"],
            line_width=2,
            row=row,
            col=1,
        )

        # Y-axis label
        fig.update_yaxes(title_text="deg", row=row, col=1, title_font_size=10)

    # Update layout
    fig.update_layout(
        height=height_per_joint * num_joints,
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
        template="plotly_dark",
    )

    # Update x-axis on the bottom subplot
    fig.update_xaxes(title_text="Frame", row=num_joints, col=1)

    # Make subplot titles smaller
    for annotation in fig.layout.annotations:
        annotation.font.size = 11

    return fig


def generate_single_joint_plot(
    episode: EpisodeData,
    joint_index: int,
    predictions: Optional[PredictionData] = None,
    current_frame: int = 0,
    show_recorded: bool = True,
    show_predicted: bool = False,
) -> go.Figure:
    """
    Generate a single joint plot (useful for individual display).

    Args:
        episode: Episode data
        joint_index: Index of the joint (0-5)
        predictions: Optional prediction data
        current_frame: Current frame for marker
        show_recorded: Whether to show recorded data
        show_predicted: Whether to show predictions

    Returns:
        Plotly Figure for a single joint
    """
    joint_name = JOINT_NAMES[joint_index]
    x_axis = np.arange(episode.length)

    fig = go.Figure()

    if show_recorded:
        # State
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.states[:, joint_index],
                mode="lines",
                name="state",
                line=dict(color=COLORS["state"], width=2),
            )
        )

        # Action
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.actions[:, joint_index],
                mode="lines",
                name="action",
                line=dict(color=COLORS["action"], width=2, dash="dash"),
            )
        )

    if show_predicted and predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions.frame_indices,
                y=predictions.predicted_actions[:, joint_index],
                mode="lines+markers",
                name="predicted",
                line=dict(color=COLORS["predicted"], width=2, dash="dot"),
                marker=dict(size=4),
            )
        )

    # Current frame marker
    fig.add_vline(
        x=current_frame,
        line_dash="solid",
        line_color=COLORS["marker"],
        line_width=2,
    )

    fig.update_layout(
        title=joint_name,
        xaxis_title="Frame",
        yaxis_title="deg",
        height=200,
        margin=dict(l=50, r=20, t=40, b=40),
        template="plotly_dark",
    )

    return fig


def get_current_values(
    episode: EpisodeData,
    frame_index: int,
    predictions: Optional[PredictionData] = None,
) -> dict:
    """
    Get current values at a specific frame.

    Args:
        episode: Episode data
        frame_index: Current frame index
        predictions: Optional predictions

    Returns:
        Dictionary with current values for each joint
    """
    frame_index = max(0, min(frame_index, episode.length - 1))

    result = {
        "frame": frame_index,
        "time": episode.timestamps[frame_index],
        "states": {},
        "actions": {},
        "predicted": {},
    }

    for i, joint_name in enumerate(JOINT_NAMES):
        result["states"][joint_name] = float(episode.states[frame_index, i])
        result["actions"][joint_name] = float(episode.actions[frame_index, i])

    if predictions is not None:
        # Find closest prediction frame
        pred_frame_idx = np.argmin(np.abs(predictions.frame_indices - frame_index))
        for i, joint_name in enumerate(JOINT_NAMES):
            result["predicted"][joint_name] = float(
                predictions.predicted_actions[pred_frame_idx, i]
            )

    return result


def format_value_display(values: dict, show_predicted: bool = False) -> str:
    """
    Format current values for display.

    Args:
        values: Dictionary from get_current_values
        show_predicted: Whether to include predictions

    Returns:
        Formatted string for display
    """
    lines = [f"Frame: {values['frame']}  Time: {values['time']:.2f}s"]

    for joint in JOINT_NAMES:
        state = values["states"][joint]
        action = values["actions"][joint]
        line = f"{joint}: state={state:.2f}, action={action:.2f}"

        if show_predicted and joint in values.get("predicted", {}):
            pred = values["predicted"][joint]
            line += f", pred={pred:.2f}"

        lines.append(line)

    return "\n".join(lines)
