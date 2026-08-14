from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from real_robot_data_collector.adapters.arm_base import ArmAdapter
from real_robot_data_collector.recorder.schema import ARM_DOF, ArmState


class DummyArmAdapter(ArmAdapter):
    name = "AgileX NERO dummy"
    state_source = "synthetic_random_walk"
    action_source = "synthetic_current_joint_positions"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        seed = int(self.config.get("seed", 7))
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._q = np.zeros((ARM_DOF,), dtype=np.float32)
        self._dq = np.zeros((ARM_DOF,), dtype=np.float32)
        self._last_t = time.monotonic()

    def connect(self) -> None:
        self._connected = True
        self._last_t = time.monotonic()

    def disconnect(self) -> None:
        self._connected = False

    def get_state(self) -> ArmState:
        if not self._connected:
            raise RuntimeError("DummyArmAdapter is not connected")
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-3)
        step = self._rng.normal(loc=0.0, scale=0.004, size=ARM_DOF).astype(np.float32)
        prev = self._q.copy()
        self._q = np.clip(self._q + step, -1.5, 1.5).astype(np.float32)
        self._dq = ((self._q - prev) / dt).astype(np.float32)
        self._last_t = now
        torques = self._rng.normal(loc=0.0, scale=0.02, size=ARM_DOF).astype(np.float32)
        pose = np.array([0.35 + self._q[0] * 0.01, self._q[1] * 0.01, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return ArmState(self._q.copy(), self._dq.copy(), torques, pose)

    def get_action(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("DummyArmAdapter is not connected")
        return self._q.copy()
