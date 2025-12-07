# GR00T SO101 Inference Diagnosis Results - Phase 2 (Async Test - Part 2)

**Date**: 2025-12-06
**Status**: Analysis of Async Inference with Camera Corruption
**Source**: `custom/logs/infer_groot_async_20251206_161308.log` and Image Analysis

---

## 1. New Findings (Camera Corruption)

The user ran the async script with the port fix. The "Port is in use" errors are GONE, confirming the thread lock worked.
However, the task still failed with random/erratic motion.

**Analysis of `eval_images` confirms severe corruption:**
*   **Head Camera**: **~40% of every frame is BLACK.**
    *   Example `head_00000.jpg`: Rows 294-479 (bottom 40%) are completely black (0).
    *   This pattern is consistent across all frames (start, middle, end).
*   **Wrist Camera**: Less severe (~4% corruption), mostly top/bottom rows or scattered lines.

### Impact on Model
**This IS the root cause of the random behavior.**
*   The model is receiving images where the bottom half (where the table/objects usually are!) is a black void.
*   The model's Vision Encoder (ViT) sees this as a massive out-of-distribution artifact.
*   Result: The model predicts random or "panic" actions because its primary sensory input is destroyed.

---

## 2. Root Cause: USB Bandwidth Saturation

The fact that corruption is *consistent* (~40% of the frame missing) and happens on *both* cameras (but worse on Head) strongly indicates **USB Bandwidth Saturation**.

*   **Setup**: 2x Cameras (640x480 RGB @ 30fps) + 1x Robot Arm (High-speed Serial)
*   **Traffic**:
    *   Cameras: 640 * 480 * 3 * 30 * 2 ≈ **55 MB/s** (Uncompressed RGB)
    *   Robot: Negligible bandwidth but high frequency interrupt requirements.
*   **Bottleneck**: If both cameras are on the same USB 2.0 root hub (theoretical 60MB/s, practical ~30-40MB/s), the bus physically cannot transmit the data fast enough. The driver drops the remaining packets for each frame, resulting in the bottom rows being empty (black).

---

## 3. Immediate Solutions

We need to reduce the USB bandwidth load or separate the devices.

### Option A: Separate USB Controllers (Hardware)
*   Plug the **Head Camera** into a USB port on the front of the PC/Jetson.
*   Plug the **Wrist Camera** into a USB port on the *back*.
*   Ensure they are on different Root Hubs (use `lsusb -t` to check).

### Option B: Reduce Camera Resolution/FPS (Software)
If hardware separation isn't possible immediately, we must reduce bandwidth.
1.  **Reduce FPS**: Drop from 30fps to **15fps**.
    *   Change `fps=30` to `fps=15` in `infer_groot_async.py`.
    *   Update `action_interval` to `0.066` (66ms).
    *   *Risk*: Control loop becomes slower (15Hz), but valid images > fast corrupted images.

2.  **Use MJPEG Compression** (If cameras support it):
    *   OpenCV defaults to uncompressed YUYV/RGB. Switching to MJPEG drastically reduces bandwidth (~5MB/s).
    *   Requires modifying `lerobot` or the `OpenCVCameraConfig` initialization.

### Option C: Reduce Image Size
*   The model expects 224x224. We are capturing 640x480.
*   We can try capturing at 320x240 if the camera supports it.

---

## 4. Recommendation

**Prioritize Option A (Hardware Fix)** if possible.
If not, **Option B (Reduce FPS)** is the quickest software mitigation.

I will update the investigation document with these findings.

