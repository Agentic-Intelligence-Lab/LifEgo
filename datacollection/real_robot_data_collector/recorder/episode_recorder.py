from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from real_robot_data_collector.recorder.manifest import episode_name, update_manifest
from real_robot_data_collector.recorder.quality import QualityTracker, compute_quality_report
from real_robot_data_collector.recorder.schema import (
    ACTION_DIM,
    ARM_DOF,
    BRAINCO_REVO2_JOINT_ORDER,
    HAND_DOF,
    HAND_TOTAL_DOF,
    ArmState,
    HandState,
    build_frame_record,
    normalize_arm_action,
    normalize_hand_action,
)
from real_robot_data_collector.utils.image_utils import resize_frame, save_image
from real_robot_data_collector.utils.json_utils import append_jsonl_line, write_json
from real_robot_data_collector.utils.time_utils import FrameTimestamp


class EpisodeRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        image_format: str = "jpg",
        save_width: int | None = None,
        save_height: int | None = None,
        camera_id: int = 0,
        dataset_name: str | None = None,
        dry_run: bool = False,
        overwrite: bool = False,
        arm_adapter_name: str = "dummy",
        hand_adapter_name: str = "dummy",
        arm_state_source: str = "unknown",
        arm_action_source: str = "unknown",
        hand_state_source: str = "unknown",
        hand_action_source: str = "unknown",
        notes: str = "",
    ) -> None:
        image_format = image_format.lower().strip(".")
        if image_format not in {"jpg", "png"}:
            raise ValueError("--image-format must be jpg or png")

        self.output_dir = Path(output_dir)
        self.image_format = image_format
        self.save_width = save_width
        self.save_height = save_height
        self.camera_id = int(camera_id)
        self.dataset_name = dataset_name or self.output_dir.name or "robot_dataset"
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.arm_adapter_name = arm_adapter_name
        self.hand_adapter_name = hand_adapter_name
        self.arm_state_source = arm_state_source
        self.arm_action_source = arm_action_source
        self.hand_state_source = hand_state_source
        self.hand_action_source = hand_action_source
        self.notes = notes

        self.episode_id: int | None = None
        self.episode_name: str | None = None
        self.episode_dir: Path | None = None
        self.images_dir: Path | None = None
        self._timestamps_file = None
        self._jsonl_file = None
        self._active = False
        self._start_time_unix = 0.0
        self._end_time_unix = 0.0
        self._task_name = ""
        self._language_instruction = ""
        self._image_width: int | None = None
        self._image_height: int | None = None
        self.quality = QualityTracker()
        self._frames: List[Dict[str, object]] = []

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start_episode(self, episode_id: int, task_name: str, language_instruction: str) -> None:
        if self._active:
            raise RuntimeError("An episode is already active")

        self.episode_id = int(episode_id)
        self.episode_name = episode_name(self.episode_id)
        self._task_name = task_name
        self._language_instruction = language_instruction
        self._frames = []
        self.quality.reset()
        self._image_width = None
        self._image_height = None
        self._start_time_unix = time.time()

        if self.dry_run:
            self._active = True
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir = self.output_dir / self.episode_name
        if self.episode_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"{self.episode_dir} already exists. Use --overwrite to replace it.")
            shutil.rmtree(self.episode_dir)
        self.images_dir = self.episode_dir / "images" / "head"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self._timestamps_file = (self.episode_dir / "timestamps.txt").open("w", encoding="utf-8")
        self._jsonl_file = (self.episode_dir / "episode.jsonl").open("w", encoding="utf-8")
        self._active = True

    def record_frame(
        self,
        frame: np.ndarray,
        timestamp: FrameTimestamp,
        arm_state: ArmState,
        hand_state: HandState,
        arm_action: np.ndarray | None,
        hand_action: np.ndarray | None,
        arm_read_failed: bool = False,
        hand_read_failed: bool = False,
    ) -> int:
        if not self._active:
            raise RuntimeError("record_frame() called without an active episode")

        frame_index = len(self._frames)
        arm_action_array = normalize_arm_action(arm_action)
        hand_action_array = normalize_hand_action(hand_action)
        save_frame = resize_frame(frame, width=self.save_width, height=self.save_height)
        self._image_height, self._image_width = [int(v) for v in save_frame.shape[:2]]
        image_rel = f"images/head/{frame_index:06d}.{self.image_format}"

        if arm_read_failed:
            self.quality.failed_robot_reads += 1
        if hand_read_failed:
            self.quality.failed_hand_reads += 1

        if not self.dry_run:
            if self.images_dir is None or self._timestamps_file is None or self._jsonl_file is None:
                raise RuntimeError("Episode files are not initialized")
            save_image(self.images_dir / f"{frame_index:06d}.{self.image_format}", save_frame)
            self._timestamps_file.write(f"{frame_index}, {timestamp.timestamp_unix:.9f}\n")
            self._timestamps_file.flush()

        record = build_frame_record(
            frame_index=frame_index,
            timestamp_unix=timestamp.timestamp_unix,
            timestamp_monotonic=timestamp.timestamp_monotonic,
            image_head=image_rel,
            arm_state=arm_state,
            hand_state=hand_state,
            arm_action=arm_action_array,
            hand_action=hand_action_array,
            camera_timestamp_unix=timestamp.camera_timestamp_unix,
            camera_timestamp_msec=timestamp.camera_timestamp_msec,
        )
        self._frames.append(record)
        if not self.dry_run and self._jsonl_file is not None:
            append_jsonl_line(self._jsonl_file, record)
        return frame_index

    def note_dropped_camera_frame(self) -> None:
        if self._active:
            self.quality.dropped_camera_frames += 1

    def finish_episode(self) -> Optional[Dict[str, object]]:
        if not self._active:
            return None

        self._end_time_unix = time.time()
        if self._timestamps_file is not None:
            self._timestamps_file.close()
            self._timestamps_file = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

        arrays = self._build_arrays()
        quality_report = compute_quality_report(arrays["timestamps_monotonic"], arrays, self.quality)
        metadata = self._build_metadata(arrays, quality_report)
        summary = {
            "episode_id": self.episode_id,
            "episode_name": self.episode_name,
            "path": self.episode_name,
            "num_frames": int(metadata["num_frames"]),
            "duration_sec": float(metadata["duration_sec"]),
            "fps_estimated": float(metadata["fps_estimated"]),
        }

        if not self.dry_run:
            if self.episode_dir is None:
                raise RuntimeError("Episode directory is not initialized")
            np.savez_compressed(self.episode_dir / "arrays.npz", **arrays)
            write_json(self.episode_dir / "metadata.json", metadata)
            write_json(self.episode_dir / "quality_report.json", quality_report)
            update_manifest(self.output_dir, summary, dataset_name=self.dataset_name)

        self._active = False
        return summary

    def abort_or_finalize_on_exit(self) -> Optional[Dict[str, object]]:
        if self._active:
            return self.finish_episode()
        return None

    def _build_arrays(self) -> Dict[str, np.ndarray]:
        count = len(self._frames)
        timestamps_unix = np.zeros((count,), dtype=np.float64)
        timestamps_monotonic = np.zeros((count,), dtype=np.float64)
        image_paths = np.empty((count,), dtype=object)
        arrays = {
            "timestamps_unix": timestamps_unix,
            "timestamps_monotonic": timestamps_monotonic,
            "arm_joint_positions": np.full((count, ARM_DOF), np.nan, dtype=np.float32),
            "arm_joint_velocities": np.full((count, ARM_DOF), np.nan, dtype=np.float32),
            "arm_joint_torques": np.full((count, ARM_DOF), np.nan, dtype=np.float32),
            "arm_end_effector_pose": np.full((count, ARM_DOF), np.nan, dtype=np.float32),
            "hand_joint_positions": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "hand_joint_velocities": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "hand_joint_currents_or_forces": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "hand_joint_positions_raw": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "hand_joint_velocities_raw": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "hand_joint_currents_raw": np.full((count, HAND_DOF), np.nan, dtype=np.float32),
            "actions": np.full((count, ACTION_DIM), np.nan, dtype=np.float32),
            "image_paths": image_paths,
        }

        for i, record in enumerate(self._frames):
            obs = record["observation"]
            action = record["action"]
            timestamps_unix[i] = float(record["timestamp_unix"])
            timestamps_monotonic[i] = float(record["timestamp_monotonic"])
            image_paths[i] = str(record["image_head"])
            arrays["arm_joint_positions"][i] = np.asarray(obs["arm_joint_positions"], dtype=np.float32)
            arrays["arm_joint_velocities"][i] = np.asarray(obs["arm_joint_velocities"], dtype=np.float32)
            arrays["arm_joint_torques"][i] = np.asarray(obs["arm_joint_torques"], dtype=np.float32)
            arrays["arm_end_effector_pose"][i] = np.asarray(obs["arm_end_effector_pose"], dtype=np.float32)
            arrays["hand_joint_positions"][i] = np.asarray(obs["hand_joint_positions"], dtype=np.float32)
            arrays["hand_joint_velocities"][i] = np.asarray(obs["hand_joint_velocities"], dtype=np.float32)
            arrays["hand_joint_currents_or_forces"][i] = np.asarray(obs["hand_joint_currents_or_forces"], dtype=np.float32)
            arrays["hand_joint_positions_raw"][i] = np.asarray(obs["hand_joint_positions_raw"], dtype=np.float32)
            arrays["hand_joint_velocities_raw"][i] = np.asarray(obs["hand_joint_velocities_raw"], dtype=np.float32)
            arrays["hand_joint_currents_raw"][i] = np.asarray(obs["hand_joint_currents_raw"], dtype=np.float32)
            arrays["actions"][i] = np.asarray(action["full_action"], dtype=np.float32)
        return arrays

    def _build_metadata(self, arrays: Dict[str, np.ndarray], quality_report: Dict[str, object]) -> Dict[str, object]:
        count = int(len(arrays["timestamps_unix"]))
        duration = float(self._end_time_unix - self._start_time_unix)
        fps = float(quality_report["estimated_fps"])
        return {
            "episode_id": self.episode_id,
            "episode_name": self.episode_name,
            "task_name": self._task_name,
            "language_instruction": self._language_instruction,
            "start_time_unix": self._start_time_unix,
            "end_time_unix": self._end_time_unix,
            "duration_sec": duration,
            "num_frames": count,
            "camera_id": self.camera_id,
            "image_width": self._image_width,
            "image_height": self._image_height,
            "image_format": self.image_format,
            "fps_estimated": fps,
            "robot": {
                "arm": {
                    "name": "AgileX NERO",
                    "dof": ARM_DOF,
                    "adapter": self.arm_adapter_name,
                    "state_source": self.arm_state_source,
                    "action_source": self.arm_action_source,
                },
                "hand": {
                    "name": "BrainCo Revo 2",
                    "active_dof": HAND_DOF,
                    "total_dof": HAND_TOTAL_DOF,
                    "adapter": self.hand_adapter_name,
                    "sdk": "bc-stark-sdk",
                    "position_range_raw": [0, 1000],
                    "speed_range_raw": [-1000, 1000],
                    "current_range_raw": [-1000, 1000],
                    "position_normalization": "raw / 1000.0",
                    "speed_normalization": "raw / 1000.0",
                    "current_normalization": "raw / 1000.0",
                    "joint_order": BRAINCO_REVO2_JOINT_ORDER,
                    "joint_order_verified": False,
                    "state_source": self.hand_state_source,
                    "action_source": self.hand_action_source,
                    "notes": "API array order must be verified with official examples or real-device calibration.",
                },
                "action_dim": ACTION_DIM,
                "state_dim_minimum": ACTION_DIM,
            },
            "data_format_version": "1.0.0",
            "notes": self.notes,
            "json_nan_policy": "episode.jsonl writes NaN tokens for unavailable numeric values; arrays.npz stores real np.nan.",
        }
