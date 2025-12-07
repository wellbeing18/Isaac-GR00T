  Features

  - Two camera views - Head and wrist cameras synced with timeline
  - 6 joint time-series plots - shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper
  - Action comparison - Toggle recorded actions, states, and predicted actions
  - Playback controls - Frame-by-frame navigation with slider
  - Episode browser - Dropdown with task descriptions

  Usage

  # Basic usage
  python custom/tools/visualize_dataset.py

  # With options
  python custom/tools/visualize_dataset.py --dataset /path/to/dataset --port 7861
  --share

  Then open http://localhost:7861 in your browser.

  Testing Verified

  - Data loading: Tested with 120 episodes from
  /home/jrobot/project/XLeRobot/datasets_groot
  - Video extraction: Extracted 182 frames (480x640x3) for episode 0
  - Plot generation: Generates 12 traces (6 joints x 2 lines)
  - Server: Responds with Gradio HTML on port 7861