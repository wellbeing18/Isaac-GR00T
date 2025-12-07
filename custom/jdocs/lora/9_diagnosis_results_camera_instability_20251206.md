# GR00T SO101 Inference Diagnosis Results - Phase 3 (Camera Instability)

**Date**: 2025-12-06
**Status**: Analysis of Camera Connection Failure
**Source**: User-provided traceback

---

## 1. Analysis of Error

After physically moving the cameras to separate USB ports (a correct step to fix bandwidth), the script now fails during initialization with:

```
[ WARN:0@24.304] global cap_v4l.cpp:1119 tryIoctl VIDEOIO(V4L2:/dev/video4): select() timeout.
RuntimeError: OpenCVCamera(4) read failed (status=False).
```

### Key Observations
1.  **Pre-warm Success**: The script's *manual* pre-warm check passed:
    ```
    [CAMERA] Pre-warming cameras...
    Opening /dev/video8 (head)... OK
    Opening /dev/video4 (wrist)... OK
    ```
    This confirms the devices are present and accessible at `/dev/video4` and `/dev/video8`.

2.  **Runtime Failure**: The error happens later, inside `self.robot.connect()`.
    *   The pre-warm code opens, reads, and *closes* the cameras.
    *   Then `robot.connect()` tries to open them again immediately.
    *   **Hypothesis**: The camera firmware or USB driver is in a "zombie state" or "busy" state because it wasn't released cleanly or quickly enough after the pre-warm check.

3.  **"select() timeout"**: This specific V4L2 error often means the device is powered but not delivering frames, often because another process (or the previous handle) still has a lock on the stream.

---

## 2. Root Cause: Race Condition in Camera Initialization

The `infer_groot_async.py` script (likely modified recently) seems to have a "Pre-warm" block that conflicts with LeRobot's internal connection logic.

**Sequence of Events:**
1.  Script starts.
2.  `[CAMERA] Pre-warm`: Opens `/dev/video4`. Reads a frame. Closes it.
3.  `robot.activate()` calls `robot.connect()`.
4.  `robot.connect()` calls `cam.connect()` (OpenCV).
5.  **CRASH**: The camera is still "busy" closing the previous stream from step 2. The OS hasn't released the resource yet.

---

## 3. Solution

We need to modify `custom/scripts/infer_groot_async.py` to **remove or robustify the pre-warm check**. Since LeRobot handles connection internally, an external pre-warm check that opens/closes the device immediately before the main connection is dangerous.

### Recommended Fix
1.  **Disable/Remove the "Pre-warm" block** in the script. Let `robot.connect()` handle the first connection.
2.  **Add a retry/delay** mechanism if connection fails, to handle the USB re-enumeration latency.

I will modify the script to remove the conflicting pre-warm logic.

