from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List

import numpy as np

from real_robot_data_collector.recorder.episode_recorder import EpisodeRecorder
from real_robot_data_collector.recorder.manifest import next_episode_id
from real_robot_data_collector.recorder.schema import ArmState, HandState
from real_robot_data_collector.utils.time_utils import FrameTimestamp


class CollectorState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    SAVING = "SAVING"


class StateAction(str, Enum):
    NONE = "NONE"
    SAVE_REQUESTED = "SAVE_REQUESTED"
    QUIT = "QUIT"


class StateMachine:
    SPACE_KEY = 32
    Q_KEYS = {ord("q"), ord("Q")}

    def __init__(
        self,
        recorder: EpisodeRecorder,
        output_dir: str | Path,
        task_name: str,
        language_instruction: str,
    ) -> None:
        self.recorder = recorder
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.language_instruction = language_instruction
        self.state = CollectorState.IDLE
        self.next_episode_id = next_episode_id(self.output_dir)
        self.current_episode_id: int | None = None
        self.last_summary = None

    @property
    def is_recording(self) -> bool:
        return self.state == CollectorState.RECORDING

    @property
    def current_frame_index(self) -> int:
        return max(self.recorder.frame_count - 1, 0)

    def process_key(self, key: int) -> StateAction:
        if key == -1 or key == 255:
            return StateAction.NONE
        key = key & 0xFF
        if key in self.Q_KEYS:
            if self.state == CollectorState.RECORDING:
                self.state = CollectorState.SAVING
                return StateAction.QUIT
            return StateAction.QUIT
        if key == self.SPACE_KEY:
            if self.state == CollectorState.IDLE:
                self._start_episode()
                return StateAction.NONE
            if self.state == CollectorState.RECORDING:
                self.state = CollectorState.SAVING
                return StateAction.SAVE_REQUESTED
            return StateAction.NONE
        return StateAction.NONE

    def record_if_needed(
        self,
        frame: np.ndarray,
        timestamp: FrameTimestamp,
        arm_state: ArmState,
        hand_state: HandState,
        arm_action: np.ndarray | None,
        hand_action: np.ndarray | None,
        arm_read_failed: bool = False,
        hand_read_failed: bool = False,
    ) -> None:
        if self.state != CollectorState.RECORDING:
            return
        self.recorder.record_frame(
            frame=frame,
            timestamp=timestamp,
            arm_state=arm_state,
            hand_state=hand_state,
            arm_action=arm_action,
            hand_action=hand_action,
            arm_read_failed=arm_read_failed,
            hand_read_failed=hand_read_failed,
        )

    def note_dropped_camera_frame(self) -> None:
        if self.state == CollectorState.RECORDING:
            self.recorder.note_dropped_camera_frame()

    def finish_saving(self) -> None:
        if self.state != CollectorState.SAVING:
            return
        self.last_summary = self.recorder.finish_episode()
        self.next_episode_id = next_episode_id(self.output_dir)
        self.current_episode_id = None
        self.state = CollectorState.IDLE

    def finalize_for_exit(self) -> None:
        if self.state == CollectorState.RECORDING:
            self.state = CollectorState.SAVING
        if self.state == CollectorState.SAVING:
            self.finish_saving()
        else:
            self.recorder.abort_or_finalize_on_exit()

    def display_lines(self, fps: float) -> List[str]:
        next_name = f"episode_{self.next_episode_id:06d}"
        if self.state == CollectorState.IDLE:
            first = f"IDLE: Press SPACE to start {next_name}"
        elif self.state == CollectorState.RECORDING:
            ep = self.current_episode_id if self.current_episode_id is not None else self.next_episode_id
            first = f"RECORDING: episode {ep:06d}, frame {self.recorder.frame_count}, press SPACE to stop and save"
        else:
            ep = self.current_episode_id if self.current_episode_id is not None else self.next_episode_id
            first = f"SAVING: saving episode {ep:06d}"
        return [
            first,
            f"FPS: {fps:.1f}",
            f"Output: {self.output_dir}",
            "SPACE: start/stop episode",
            "Press q to quit",
        ]

    def _start_episode(self) -> None:
        episode_id = self.next_episode_id
        self.recorder.start_episode(
            episode_id=episode_id,
            task_name=self.task_name,
            language_instruction=self.language_instruction,
        )
        self.current_episode_id = episode_id
        self.state = CollectorState.RECORDING
