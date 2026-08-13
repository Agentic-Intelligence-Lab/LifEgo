from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from real_robot_data_collector.recorder.schema import ARM_DOF, ArmState


class ArmAdapter(ABC):
    name = "abstract_arm"
    dof = ARM_DOF
    state_source = "unknown"
    action_source = "unknown"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> ArmState:
        raise NotImplementedError

    @abstractmethod
    def get_action(self) -> np.ndarray:
        raise NotImplementedError


class NullArmAdapter(ArmAdapter):
    name = "No arm"
    state_source = "nan_no_arm_adapter"
    action_source = "nan_no_arm_adapter"

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_state(self) -> ArmState:
        return ArmState.nan()

    def get_action(self) -> np.ndarray:
        return np.full((ARM_DOF,), np.nan, dtype=np.float32)
