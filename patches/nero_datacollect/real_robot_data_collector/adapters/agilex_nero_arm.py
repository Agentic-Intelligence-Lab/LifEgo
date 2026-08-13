from __future__ import annotations

from typing import Any, Dict

import numpy as np

from real_robot_data_collector.adapters.arm_base import ArmAdapter
from real_robot_data_collector.recorder.schema import ARM_DOF, ArmState


class AgilexNeroArmAdapter(ArmAdapter):
    """Skeleton for integrating the real AgileX NERO SDK.

    Keep all vendor SDK imports and calls inside this file. The collector should
    stay dependent only on ArmAdapter.
    """

    name = "AgileX NERO"
    state_source = "todo_vendor_sdk"
    action_source = "todo_vendor_sdk_or_controller_command"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client = None

    def connect(self) -> None:
        # TODO: Import and initialize the official AgileX NERO SDK here.
        # Example config fields may include ip, port, can_interface, ros_topic,
        # or calibration files, depending on the SDK transport you use.
        raise NotImplementedError(
            "AgilexNeroArmAdapter.connect() is a skeleton. Fill in the real AgileX NERO SDK initialization."
        )

    def disconnect(self) -> None:
        # TODO: Close the SDK client, CAN bus, ROS node, or network connection.
        self._client = None

    def get_state(self) -> ArmState:
        # TODO: Replace the placeholders below with SDK reads.
        # Required output:
        # - joint_positions: shape [7]
        # - joint_velocities: shape [7], or NaN if unavailable
        # - joint_torques: shape [7], or NaN if unavailable
        # - end_effector_pose: [x, y, z, qx, qy, qz, qw], or NaN if unavailable
        raise NotImplementedError("Fill in AgileX NERO joint and end-effector state SDK reads.")

    def get_action(self) -> np.ndarray:
        # TODO: Return the controller command actually sent to the arm.
        # If the SDK/control stack cannot expose commands, return NaN and use
        # exporter-side next-qpos action generation for imitation learning.
        return np.full((ARM_DOF,), np.nan, dtype=np.float32)
