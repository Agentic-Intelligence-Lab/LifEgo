from __future__ import annotations

from typing import Any, Dict

import numpy as np

from real_robot_data_collector.adapters.hand_base import HandAdapter
from real_robot_data_collector.recorder.schema import HAND_DOF, HandState


class DeprecatedReNV2HandAdapter(HandAdapter):
    """Compatibility shim for old configs that used --hand-adapter renv2."""

    name = "Deprecated ReNV2"
    state_source = "deprecated_use_brainco_revo2"
    action_source = "deprecated_use_brainco_revo2"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)

    def connect(self) -> None:
        raise RuntimeError(
            "--hand-adapter renv2 is deprecated. The confirmed hand model is BrainCo Revo 2; "
            "use --hand-adapter brainco_revo2 and bc-stark-sdk==2.0.2."
        )

    def disconnect(self) -> None:
        return None

    def get_state(self) -> HandState:
        raise RuntimeError("Deprecated renv2 adapter cannot read state. Use BrainCoRevo2HandAdapter.")

    def get_action(self) -> np.ndarray:
        return np.full((HAND_DOF,), np.nan, dtype=np.float32)


ReNV2HandAdapter = DeprecatedReNV2HandAdapter
