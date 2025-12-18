#!/usr/bin/env python3
"""
Hardware Detection Script for SO101 Robot Setup

This script detects all connected cameras and serial ports, displays their
USB topology, and can generate/update the hardware configuration YAML file.

Usage:
    python detect_hardware.py              # Show detected devices
    python detect_hardware.py --preview    # Preview cameras to identify them
    python detect_hardware.py --generate   # Generate new config file
    python detect_hardware.py --update     # Update existing config file

Author: Claude Code Assistant
Date: 2025-12-06
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


# Config file path
CONFIG_PATH = Path(__file__).parent.parent / "cfgs" / "so101_hardware.yaml"


def run_command(cmd: str) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def detect_video_devices() -> list:
    """Detect all video devices and their USB paths."""
    devices = []

    # Get v4l2 device list
    v4l2_output = run_command("v4l2-ctl --list-devices 2>/dev/null")

    # Parse v4l2 output
    current_name = None
    current_usb_path = None

    for line in v4l2_output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Camera name line (e.g., "USB2.0_CAM1: USB2.0_CAM1 (usb-0000:80:14.0-1):")
        if '(' in line and 'usb-' in line:
            match = re.search(r'(.*?)\s*\(usb-([^)]+)\)', line)
            if match:
                current_name = match.group(1).strip().rstrip(':')
                current_usb_path = match.group(2)
        # Device path line (e.g., "/dev/video4")
        elif line.startswith('/dev/video'):
            device_path = line
            device_num = int(re.search(r'video(\d+)', device_path).group(1))

            # Only add main video devices (even numbers typically)
            if device_num % 2 == 0 or 'video0' in device_path:
                devices.append({
                    'path': device_path,
                    'index': device_num,
                    'name': current_name or 'Unknown',
                    'usb_path': current_usb_path or 'Unknown',
                    'connection': 'hub' if '.' in (current_usb_path or '') and current_usb_path.count('.') > 1 else 'direct'
                })

    return devices


def detect_serial_ports() -> list:
    """Detect all serial ports and their USB paths."""
    ports = []

    # Find ttyACM devices
    for tty_path in Path('/sys/class/tty').glob('ttyACM*'):
        device_link = tty_path / 'device'
        if device_link.exists():
            # Read the symlink to get USB path
            real_path = os.readlink(device_link)
            # Extract USB path (e.g., "3-4.1.1" from "../../../3-4.1.1:1.0")
            match = re.search(r'(\d+-[\d.]+)', real_path)
            usb_path = match.group(1) if match else 'Unknown'

            port_name = tty_path.name
            ports.append({
                'path': f'/dev/{port_name}',
                'name': port_name,
                'usb_path': usb_path,
                'connection': 'hub' if '.' in usb_path else 'direct'
            })

    # Sort by port name
    ports.sort(key=lambda x: x['path'])

    return ports


def get_usb_topology() -> str:
    """Get USB topology from lsusb -t."""
    return run_command("lsusb -t")


def print_detected_devices(videos: list, serials: list):
    """Print detected devices in a nice format."""
    print("\n" + "=" * 70)
    print("DETECTED HARDWARE DEVICES")
    print("=" * 70)

    print("\n--- VIDEO DEVICES (Cameras) ---")
    print(f"{'Device':<15} {'Index':<8} {'USB Path':<20} {'Connection':<10} {'Name'}")
    print("-" * 70)
    for v in videos:
        conn_icon = "DIRECT" if v['connection'] == 'direct' else "Hub"
        print(f"{v['path']:<15} {v['index']:<8} {v['usb_path']:<20} {conn_icon:<10} {v['name']}")

    print("\n--- SERIAL PORTS (Robot Arms) ---")
    print(f"{'Device':<15} {'USB Path':<20} {'Connection':<10}")
    print("-" * 50)
    for s in serials:
        conn_icon = "DIRECT" if s['connection'] == 'direct' else "Hub"
        print(f"{s['path']:<15} {s['usb_path']:<20} {conn_icon:<10}")

    print("\n--- USB TOPOLOGY ---")
    print(get_usb_topology())


def preview_cameras(videos: list):
    """Preview cameras to help identify them."""
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python")
        return

    print("\n" + "=" * 70)
    print("CAMERA PREVIEW MODE")
    print("=" * 70)
    print("Press 'q' to close a window, 'n' for next camera")
    print("Note which camera shows which view (head/wrist/etc.)")
    print("=" * 70)

    for v in videos:
        idx = v['index']
        name = v['name']
        usb_path = v['usb_path']

        print(f"\nOpening /dev/video{idx} ({name}, USB: {usb_path})...")

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"  Failed to open video{idx}")
            continue

        window_name = f"video{idx} - {name} - USB:{usb_path} (q=quit, n=next)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"  Failed to read from video{idx}")
                break

            # Add text overlay
            cv2.putText(frame, f"video{idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"USB: {usb_path}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('n'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if key == ord('q'):
            break

    print("\nCamera preview complete.")


def generate_config(videos: list, serials: list, output_path: Path = None):
    """Generate hardware config YAML file."""
    if output_path is None:
        output_path = CONFIG_PATH

    # Try to identify cameras by index
    head_cam = None
    wrist_cam = None
    right_arm_cam = None

    # Sort by index
    videos_sorted = sorted(videos, key=lambda x: x['index'])

    # Heuristic: prefer direct-connected cameras for head/wrist
    direct_cams = [v for v in videos_sorted if v['connection'] == 'direct' and 'HP' not in v['name']]
    hub_cams = [v for v in videos_sorted if v['connection'] == 'hub']

    if len(direct_cams) >= 2:
        head_cam = direct_cams[0]
        wrist_cam = direct_cams[1]
    elif len(direct_cams) == 1:
        head_cam = direct_cams[0]
        if hub_cams:
            wrist_cam = hub_cams[0]

    if hub_cams:
        right_arm_cam = hub_cams[-1] if len(hub_cams) > 0 else None

    # Serial ports
    left_arm_port = serials[0] if len(serials) > 0 else {'path': '/dev/ttyACM0', 'usb_path': 'unknown'}
    right_arm_port = serials[1] if len(serials) > 1 else {'path': '/dev/ttyACM1', 'usb_path': 'unknown'}

    config = {
        '_comment': f'Auto-generated by detect_hardware.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',

        'robot_arms': {
            'left': {
                'port': left_arm_port['path'],
                'usb_path': left_arm_port['usb_path'],
                'robot_id': 'xlerobot_left_arm',
                'use_degrees': True,
            },
            'right': {
                'port': right_arm_port['path'],
                'usb_path': right_arm_port['usb_path'],
                'robot_id': 'xlerobot_right_arm',
                'use_degrees': True,
            },
        },

        'cameras': {
            'head': {
                'device_index': head_cam['index'] if head_cam else 8,
                'usb_path': head_cam['usb_path'] if head_cam else 'unknown',
                'connection': head_cam['connection'] if head_cam else 'direct',
                'resolution': {'width': 640, 'height': 480},
                'fps': 30,
            },
            'wrist': {
                'device_index': wrist_cam['index'] if wrist_cam else 4,
                'usb_path': wrist_cam['usb_path'] if wrist_cam else 'unknown',
                'connection': wrist_cam['connection'] if wrist_cam else 'direct',
                'resolution': {'width': 640, 'height': 480},
                'fps': 30,
            },
            'right_arm': {
                'device_index': right_arm_cam['index'] if right_arm_cam else 6,
                'usb_path': right_arm_cam['usb_path'] if right_arm_cam else 'unknown',
                'connection': right_arm_cam['connection'] if right_arm_cam else 'hub',
                'resolution': {'width': 640, 'height': 480},
                'fps': 30,
            },
        },

        'inference': {
            'data_config': 'so100_dualcam',
            'embodiment_tag': 'new_embodiment',
            'denoising_steps': 4,
            'action_horizon': 16,
            'action_interval': 0.033,
            'default_task': 'pick the red cube from the table',
        },

        'home_positions': {
            'training_aligned': {
                'shoulder_pan': 0.0,
                'shoulder_lift': -20.0,
                'elbow_flex': 20.0,
                'wrist_flex': 60.0,
                'wrist_roll': 0.0,
                'gripper': 5.0,
            },
            'default': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 50.0,
            },
        },

        'logging': {
            'log_dir': '/home/jrobot/project/Isaac-GR00T/custom/logs',
            'eval_images_dir': '/home/jrobot/project/Isaac-GR00T/eval_images',
        },
    }

    return config


def save_config(config: dict, output_path: Path):
    """Save config to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# SO101 Robot Hardware Configuration\n")
        f.write("# ===================================\n")
        f.write(f"# Auto-generated by detect_hardware.py\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# IMPORTANT: Review and adjust camera assignments after generation!\n")
        f.write("# Use: python detect_hardware.py --preview to identify cameras\n")
        f.write("#\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig saved to: {output_path}")


def interactive_assign(videos: list, serials: list):
    """Interactive mode to assign devices."""
    print("\n" + "=" * 70)
    print("INTERACTIVE DEVICE ASSIGNMENT")
    print("=" * 70)

    # Show available video devices
    print("\nAvailable cameras:")
    for i, v in enumerate(videos):
        conn = "DIRECT" if v['connection'] == 'direct' else "Hub"
        print(f"  [{i}] video{v['index']} - {v['name']} ({conn}, USB: {v['usb_path']})")

    print("\nAssign cameras (enter number or press Enter to skip):")

    head_idx = input("  Head/central camera [0]: ").strip()
    head_cam = videos[int(head_idx)] if head_idx.isdigit() and int(head_idx) < len(videos) else videos[0] if videos else None

    wrist_idx = input("  Wrist/left-arm camera [1]: ").strip()
    wrist_cam = videos[int(wrist_idx)] if wrist_idx.isdigit() and int(wrist_idx) < len(videos) else videos[1] if len(videos) > 1 else None

    # Show available serial ports
    print("\nAvailable serial ports:")
    for i, s in enumerate(serials):
        print(f"  [{i}] {s['path']} (USB: {s['usb_path']})")

    print("\nAssign robot arms (enter number or press Enter to skip):")

    left_idx = input("  Left arm serial port [0]: ").strip()
    left_port = serials[int(left_idx)] if left_idx.isdigit() and int(left_idx) < len(serials) else serials[0] if serials else None

    return {
        'head_cam': head_cam,
        'wrist_cam': wrist_cam,
        'left_port': left_port,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect SO101 robot hardware and generate config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_hardware.py              # Show detected devices
  python detect_hardware.py --preview    # Preview cameras
  python detect_hardware.py --generate   # Generate new config
  python detect_hardware.py --interactive # Interactive assignment
        """
    )

    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview cameras to identify them"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate new config file (overwrites existing)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode to assign devices"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Output config file path (default: {CONFIG_PATH})"
    )

    args = parser.parse_args()

    # Detect devices
    print("Detecting hardware devices...")
    videos = detect_video_devices()
    serials = detect_serial_ports()

    # Print detected devices
    print_detected_devices(videos, serials)

    # Preview mode
    if args.preview:
        preview_cameras(videos)

    # Interactive mode
    if args.interactive:
        assignments = interactive_assign(videos, serials)
        # Generate with assignments
        config = generate_config(videos, serials)
        if assignments['head_cam']:
            config['cameras']['head']['device_index'] = assignments['head_cam']['index']
            config['cameras']['head']['usb_path'] = assignments['head_cam']['usb_path']
        if assignments['wrist_cam']:
            config['cameras']['wrist']['device_index'] = assignments['wrist_cam']['index']
            config['cameras']['wrist']['usb_path'] = assignments['wrist_cam']['usb_path']
        if assignments['left_port']:
            config['robot_arms']['left']['port'] = assignments['left_port']['path']
            config['robot_arms']['left']['usb_path'] = assignments['left_port']['usb_path']

        save_config(config, Path(args.output))

    # Generate mode
    elif args.generate:
        config = generate_config(videos, serials)
        save_config(config, Path(args.output))

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIG VALUES")
    print("=" * 70)

    # Find direct-connected cameras (excluding laptop webcam)
    external_cams = [v for v in videos if 'HP' not in v['name'] and 'True Vision' not in v['name']]
    direct_cams = [v for v in external_cams if v['connection'] == 'direct']

    print("\nFor so101_hardware.yaml:")
    if len(direct_cams) >= 2:
        print(f"  cameras.head.device_index: {direct_cams[0]['index']}  # {direct_cams[0]['usb_path']} (DIRECT)")
        print(f"  cameras.wrist.device_index: {direct_cams[1]['index']}  # {direct_cams[1]['usb_path']} (DIRECT)")
    elif len(direct_cams) == 1:
        print(f"  cameras.head.device_index: {direct_cams[0]['index']}  # {direct_cams[0]['usb_path']} (DIRECT)")
        print(f"  cameras.wrist.device_index: ???  # No second direct camera found!")

    if serials:
        print(f"  robot_arms.left.port: {serials[0]['path']}  # USB: {serials[0]['usb_path']}")

    print("\nTo update config, run:")
    print(f"  python {__file__} --generate")
    print(f"  # Or edit: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
