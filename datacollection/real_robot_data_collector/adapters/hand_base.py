from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from real_robot_data_collector.recorder.schema import HAND_DOF, HAND_TOTAL_DOF, HandState


class HandAdapter(ABC):
    name = "abstract_hand"
    active_dof = HAND_DOF
    total_dof = HAND_TOTAL_DOF
    state_source = "unknown"
    action_source = "unknown"
    sdk = "unknown"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> HandState:
        raise NotImplementedError

    @abstractmethod
    def get_action(self) -> np.ndarray:
        raise NotImplementedError


class NullHandAdapter(HandAdapter):
    name = "No hand"
    state_source = "nan_no_hand_adapter"
    action_source = "nan_no_hand_adapter"

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_state(self) -> HandState:
        return HandState.nan()

    def get_action(self) -> np.ndarray:
        return np.full((HAND_DOF,), np.nan, dtype=np.float32)
