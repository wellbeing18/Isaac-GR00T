# GR00T SO101 Inference Diagnosis Results - Phase 2 (Async Test)

**Date**: 2025-12-06
**Status**: Analysis of Async Inference Test
**Source**: User-provided terminal logs

---

## 1. Findings from Async Inference Log

The log confirms that the **Async Architecture works as designed** in terms of timing, but reveals **hardware communication issues** that are likely causing the task failure.

### Successes (Architecture)
*   **Smooth Execution**: The consumer loop ran at a stable **~29.9 Hz** (target 30Hz). This eliminates the 2Hz stop-and-go motion.
*   **Zero Stale Actions**: `Stale: 0.0%`. The producer (5.6Hz) was fast enough to keep the consumer fed with fresh trajectories.
*   **Low Latency**: Average latency dropped to **36ms**, significantly better than the ~150ms+ in blocking mode.

### FAILURES (Hardware/Task)
Despite smooth motion, the task failed ("arm goes back and forth"). The logs reveal why:

1.  **Severe Motor Communication Errors**:
    ```
    [CONSUMER] Error: Failed to sync write 'Goal_Position' ... [TxRxResult] Port is in use!
    [PRODUCER] Error: Failed to sync read 'Present_Position' ... [TxRxResult] Port is in use!
    ```
    *   There are **14+ errors** in 60 seconds where the motor commands failed to send.
    *   **Root Cause**: Thread contention. The Producer (reading state) and Consumer (writing actions) are fighting for the same serial port (`/dev/ttyACM2`) or the USB bus is saturated.
    *   **Impact**: The "back and forth" motion is likely the robot executing an old command, then jumping to a new one after a dropped packet, or the state estimation being wrong because a read failed.

2.  **Low Producer Rate**:
    *   Producer rate dropped to **5.4 Hz** (was ~7Hz in diagnosis).
    *   Inference time spiked to **~190ms** frequently.
    *   This indicates the system is under heavy load, exacerbating the USB contention.

---

## 2. Diagnosis & Next Steps

The "back and forth" behavior is not a model intelligence issue but a **control loop failure** due to dropped motor commands. The model is generating a trajectory, but the robot isn't receiving all the points, causing it to jitter or reset.

### Recommendation 1: Fix USB/Thread Contention (CRITICAL)
The `So101RobotInterface` is sharing the same `self.robot` instance across threads without a lock on the serial communication. LeRobot's `SO101Follower` is likely not thread-safe for simultaneous Read/Write operations.

**Action**: Update `infer_groot_async.py` to add a **Hardware Lock**.
*   Wrap all `robot.get_observation()` and `robot.send_action()` calls in a `with self.robot_lock:` block.
*   This will serialize access to the USB port, preventing "Port is in use!" errors.

### Recommendation 2: USB Bandwidth Management
The error `[TxRxResult] Port is in use!` combined with camera capture suggests the USB bus is saturated.
*   **Check**: Are cameras and robot on the same USB hub?
*   **Action**: If possible, move the robot arm USB to a different physical port group than the cameras.

### Recommendation 3: Reduce Denoising Steps
The inference time (190ms, 5.4Hz) is dangerously slow.
*   **Action**: Reduce `denoising_steps` from 4 to **2** (if supported) or optimize the model.
*   However, fixing the motor errors is the priority. A 5Hz smooth update is better than a 5Hz broken update.

---

## 3. Plan Update

1.  **Modify `infer_groot_async.py`**: Add thread locking for robot communication.
2.  **Retest**: Run the task again. If the "Port is in use" errors disappear, we expect the "back and forth" behavior to resolve.

