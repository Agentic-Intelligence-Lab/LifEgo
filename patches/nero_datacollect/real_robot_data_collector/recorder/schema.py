from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List

import numpy as np


DATA_FORMAT_VERSION = "1.0.0"
ARM_DOF = 7
HAND_DOF = 6
HAND_TOTAL_DOF = 11
ACTION_DIM = ARM_DOF + HAND_DOF
BRAINCO_REVO2_JOINT_ORDER = [
    "thumb_flex",
    "thumb_aux",
    "index_flex",
    "middle_flex",
    "ring_flex",
    "little_flex",
]


def _as_float_vector(values: Iterable[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {array.shape}")
    return array


def _nan_vector(length: int) -> np.ndarray:
    return np.full((length,), np.nan, dtype=np.float32)


def to_float_list(array: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(array, dtype=np.float32).reshape(-1)]


def to_raw_number_list(array: np.ndarray) -> List[float | int]:
    values: List[float | int] = []
    for value in np.asarray(array, dtype=np.float32).reshape(-1):
        as_float = float(value)
        if np.isfinite(as_float) and as_float.is_integer():
            values.append(int(as_float))
        else:
            values.append(as_float)
    return values


def _jsonable_sdk_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable_sdk_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_sdk_value(v) for v in value]
    return repr(value)


@dataclass(frozen=True)
class ArmState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    end_effector_pose: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_positions",
            _as_float_vector(self.joint_positions, ARM_DOF, "arm_joint_positions"),
        )
        object.__setattr__(
            self,
            "joint_velocities",
            _as_float_vector(self.joint_velocities, ARM_DOF, "arm_joint_velocities"),
        )
        object.__setattr__(
            self,
            "joint_torques",
            _as_float_vector(self.joint_torques, ARM_DOF, "arm_joint_torques"),
        )
        object.__setattr__(
            self,
            "end_effector_pose",
            _as_float_vector(self.end_effector_pose, ARM_DOF, "arm_end_effector_pose"),
        )

    @classmethod
    def nan(cls) -> "ArmState":
        return cls(_nan_vector(ARM_DOF), _nan_vector(ARM_DOF), _nan_vector(ARM_DOF), _nan_vector(ARM_DOF))

    def to_observation_dict(self) -> Dict[str, List[float]]:
        return {
            "arm_joint_positions": to_float_list(self.joint_positions),
            "arm_joint_velocities": to_float_list(self.joint_velocities),
            "arm_joint_torques": to_float_list(self.joint_torques),
            "arm_end_effector_pose": to_float_list(self.end_effector_pose),
        }


@dataclass(frozen=True)
class HandState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_currents_or_forces: np.ndarray
    raw_positions: np.ndarray | None = None
    raw_speeds: np.ndarray | None = None
    raw_currents: np.ndarray | None = None
    motor_states: list[Any] | None = None
    timestamp_unix: float | None = None
    timestamp_monotonic: float | None = None
    is_valid: bool = True
    error_message: str | None = None

    def __post_init__(self) -> None:
        positions = _as_float_vector(self.joint_positions, HAND_DOF, "hand_joint_positions")
        velocities = _as_float_vector(self.joint_velocities, HAND_DOF, "hand_joint_velocities")
        currents = _as_float_vector(
            self.joint_currents_or_forces,
            HAND_DOF,
            "hand_joint_currents_or_forces",
        )
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_velocities", velocities)
        object.__setattr__(self, "joint_currents_or_forces", currents)
        object.__setattr__(
            self,
            "raw_positions",
            _as_float_vector(
                positions * 1000.0 if self.raw_positions is None else self.raw_positions,
                HAND_DOF,
                "hand_joint_positions_raw",
            ),
        )
        object.__setattr__(
            self,
            "raw_speeds",
            _as_float_vector(
                velocities * 1000.0 if self.raw_speeds is None else self.raw_speeds,
                HAND_DOF,
                "hand_joint_velocities_raw",
            ),
        )
        object.__setattr__(
            self,
            "raw_currents",
            _as_float_vector(
                currents * 1000.0 if self.raw_currents is None else self.raw_currents,
                HAND_DOF,
                "hand_joint_currents_raw",
            ),
        )
        object.__setattr__(self, "timestamp_unix", time.time() if self.timestamp_unix is None else float(self.timestamp_unix))
        object.__setattr__(
            self,
            "timestamp_monotonic",
            time.monotonic() if self.timestamp_monotonic is None else float(self.timestamp_monotonic),
        )

    @classmethod
    def nan(cls, error_message: str | None = None) -> "HandState":
        return cls(
            _nan_vector(HAND_DOF),
            _nan_vector(HAND_DOF),
            _nan_vector(HAND_DOF),
            raw_positions=_nan_vector(HAND_DOF),
            raw_speeds=_nan_vector(HAND_DOF),
            raw_currents=_nan_vector(HAND_DOF),
            is_valid=False,
            error_message=error_message,
        )

    def to_observation_dict(self) -> Dict[str, Any]:
        observation: Dict[str, Any] = {
            "hand_joint_positions": to_float_list(self.joint_positions),
            "hand_joint_velocities": to_float_list(self.joint_velocities),
            "hand_joint_currents_or_forces": to_float_list(self.joint_currents_or_forces),
            "hand_joint_positions_raw": to_raw_number_list(self.raw_positions),
            "hand_joint_velocities_raw": to_raw_number_list(self.raw_speeds),
            "hand_joint_currents_raw": to_raw_number_list(self.raw_currents),
            "hand_timestamp_unix": float(self.timestamp_unix),
            "hand_timestamp_monotonic": float(self.timestamp_monotonic),
            "hand_is_valid": bool(self.is_valid),
        }
        if self.motor_states is not None:
            observation["hand_motor_states"] = _jsonable_sdk_value(self.motor_states)
        if self.error_message:
            observation["hand_error_message"] = self.error_message
        return observation


def normalize_arm_action(action: Iterable[float] | np.ndarray | None) -> np.ndarray:
    if action is None:
        return _nan_vector(ARM_DOF)
    return _as_float_vector(action, ARM_DOF, "arm_action")


def normalize_hand_action(action: Iterable[float] | np.ndarray | None) -> np.ndarray:
    if action is None:
        return _nan_vector(HAND_DOF)
    return _as_float_vector(action, HAND_DOF, "hand_action")


def build_frame_record(
    frame_index: int,
    timestamp_unix: float,
    timestamp_monotonic: float,
    image_head: str,
    arm_state: ArmState,
    hand_state: HandState,
    arm_action: np.ndarray,
    hand_action: np.ndarray,
    camera_timestamp_unix: float | None = None,
    camera_timestamp_msec: float | None = None,
) -> Dict[str, Any]:
    full_action = np.concatenate([arm_action, hand_action]).astype(np.float32)
    observation: Dict[str, List[float]] = {}
    observation.update(arm_state.to_observation_dict())
    observation.update(hand_state.to_observation_dict())
    record: Dict[str, Any] = {
        "frame_index": int(frame_index),
        "timestamp_unix": float(timestamp_unix),
        "timestamp_monotonic": float(timestamp_monotonic),
        "image_head": image_head,
        "observation": observation,
        "action": {
            "arm_action": to_float_list(arm_action),
            "hand_action": to_float_list(hand_action),
            "full_action": to_float_list(full_action),
        },
    }
    if camera_timestamp_unix is not None:
        record["camera_timestamp_unix"] = float(camera_timestamp_unix)
    if camera_timestamp_msec is not None:
        record["camera_timestamp_msec"] = float(camera_timestamp_msec)
    return record
