from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class QualityTracker:
    dropped_camera_frames: int = 0
    failed_robot_reads: int = 0
    failed_hand_reads: int = 0

    def reset(self) -> None:
        self.dropped_camera_frames = 0
        self.failed_robot_reads = 0
        self.failed_hand_reads = 0


def compute_quality_report(
    timestamps_monotonic: np.ndarray,
    arrays: Dict[str, np.ndarray],
    tracker: QualityTracker,
) -> Dict[str, object]:
    num_frames = int(len(timestamps_monotonic))
    intervals = np.diff(timestamps_monotonic.astype(np.float64)) if num_frames > 1 else np.asarray([], dtype=np.float64)
    duration = float(timestamps_monotonic[-1] - timestamps_monotonic[0]) if num_frames > 1 else 0.0
    estimated_fps = float((num_frames - 1) / duration) if duration > 0 else 0.0

    state_keys = [
        "arm_joint_positions",
        "arm_joint_velocities",
        "arm_joint_torques",
        "arm_end_effector_pose",
        "hand_joint_positions",
        "hand_joint_velocities",
        "hand_joint_currents_or_forces",
        "hand_joint_positions_raw",
        "hand_joint_velocities_raw",
        "hand_joint_currents_raw",
    ]
    has_nan_state = any(np.isnan(arrays[key]).any() for key in state_keys if key in arrays)
    has_nan_action = bool("actions" in arrays and np.isnan(arrays["actions"]).any())

    return {
        "num_frames": num_frames,
        "dropped_camera_frames": int(tracker.dropped_camera_frames),
        "failed_robot_reads": int(tracker.failed_robot_reads),
        "failed_hand_reads": int(tracker.failed_hand_reads),
        "estimated_fps": estimated_fps,
        "min_frame_interval": float(np.min(intervals)) if len(intervals) else 0.0,
        "max_frame_interval": float(np.max(intervals)) if len(intervals) else 0.0,
        "mean_frame_interval": float(np.mean(intervals)) if len(intervals) else 0.0,
        "has_nan_state": bool(has_nan_state),
        "has_nan_action": bool(has_nan_action),
    }
