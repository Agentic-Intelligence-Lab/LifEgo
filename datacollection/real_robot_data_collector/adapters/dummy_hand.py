from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from real_robot_data_collector.adapters.hand_base import HandAdapter
from real_robot_data_collector.recorder.schema import HAND_DOF, HandState


class DummyHandAdapter(HandAdapter):
    name = "BrainCo Revo 2 dummy"
    state_source = "synthetic_random_walk"
    action_source = "synthetic_current_joint_positions"
    sdk = "dummy"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        seed = int(self.config.get("seed", 11))
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._q = np.zeros((HAND_DOF,), dtype=np.float32)
        self._dq = np.zeros((HAND_DOF,), dtype=np.float32)
        self._last_t = time.monotonic()

    def connect(self) -> None:
        self._connected = True
        self._last_t = time.monotonic()

    def disconnect(self) -> None:
        self._connected = False

    def get_state(self) -> HandState:
        if not self._connected:
            raise RuntimeError("DummyHandAdapter is not connected")
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-3)
        step = self._rng.normal(loc=0.0, scale=0.006, size=HAND_DOF).astype(np.float32)
        prev = self._q.copy()
        self._q = np.clip(self._q + step, 0.0, 1.0).astype(np.float32)
        self._dq = ((self._q - prev) / dt).astype(np.float32)
        self._last_t = now
        currents = self._rng.normal(loc=0.15, scale=0.01, size=HAND_DOF).astype(np.float32)
        return HandState(
            self._q.copy(),
            self._dq.copy(),
            currents,
            raw_positions=(self._q * 1000.0).astype(np.float32),
            raw_speeds=(self._dq * 1000.0).astype(np.float32),
            raw_currents=(currents * 1000.0).astype(np.float32),
        )

    def get_action(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("DummyHandAdapter is not connected")
        return self._q.copy()
