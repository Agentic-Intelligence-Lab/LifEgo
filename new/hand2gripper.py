#!/usr/bin/env python3
"""Hand reconstruction to gripper target conversion utilities.

This module contains the hand-to-gripper definitions that were previously mixed
into WiLoR preprocessing.  WiLoR should only reconstruct hand geometry; modes in
this file decide how that geometry becomes an end-effector target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


@dataclass
class GripperTarget:
    mode: str
    T_hand_in_cam: np.ndarray
    T_eef_in_cam: np.ndarray
    grasp_state: int
    humanego_kpts_3d: np.ndarray
    wrist_pose_in_cam: np.ndarray | None = None
    confidence: float | None = None
    is_right: bool | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": to_jsonable(self.confidence),
            "is_right": to_jsonable(self.is_right),
            "grasp_state": to_jsonable(self.grasp_state),
            "T_hand_in_cam": to_jsonable(self.T_hand_in_cam),
            "T_eef_in_cam": to_jsonable(self.T_eef_in_cam),
            "wrist_pose_in_cam": to_jsonable(self.wrist_pose_in_cam),
            "humanego_kpts_3d": to_jsonable(self.humanego_kpts_3d),
        }


class HumanEgoMode:
    """Convert WiLoR hand geometry to the HumanEgo-style gripper target.

    The input 3D keypoints are expected in WiLoR/MANO order and camera frame.
    The output EEF pose is also in the camera frame; camera-to-robot-base
    conversion belongs to the export stage.
    """

    # WiLoR/MANO 21-keypoint order:
    #   0 wrist, 1 thumb CMC, 2 thumb MCP, 3 thumb IP, 4 thumb tip,
    #   5 index MCP, 6 index PIP, 7 index DIP, 8 index tip,
    #   9 middle MCP, 10 middle PIP, 11 middle DIP, 12 middle tip,
    #   13 ring MCP, 14 ring PIP, 15 ring DIP, 16 ring tip,
    #   17 pinky MCP, 18 pinky PIP, 19 pinky DIP, 20 pinky tip.
    #
    # HumanEgo's hand-to-EEF path expects the Aria-like keypoint order below:
    #   0 thumb tip, 1 index tip, 2 middle tip, 3 ring tip, 4 pinky tip,
    #   5 wrist, 6 thumb MCP, 7 thumb IP, 8 index MCP, 9 index PIP,
    #   10 index DIP, 11 middle MCP, 12 middle PIP, 13 middle DIP,
    #   14 ring MCP, 15 ring PIP, 16 ring DIP, 17 pinky MCP,
    #   18 pinky PIP, 19 pinky DIP, 20 palm center.
    WILOR_TO_HUMANEGO = [
        4,
        8,
        12,
        16,
        20,
        0,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
        17,
        18,
        19,
        -1,
    ]

    DEFAULT_T_HAND_FROM_EEF = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        T_hand_from_eef: np.ndarray | None = None,
        grasp_ratio_threshold: float = 1.0,
        keep_rotation_sign_consistency: bool = True,
    ) -> None:
        self.T_hand_from_eef = (
            np.array(T_hand_from_eef, dtype=np.float64)
            if T_hand_from_eef is not None
            else self.DEFAULT_T_HAND_FROM_EEF.copy()
        )
        if self.T_hand_from_eef.shape != (4, 4):
            raise ValueError("T_hand_from_eef must be a 4x4 matrix")
        self.T_eef_from_hand = np.linalg.inv(self.T_hand_from_eef)
        self.grasp_ratio_threshold = float(grasp_ratio_threshold)
        self.keep_rotation_sign_consistency = bool(keep_rotation_sign_consistency)
        self._prev_mid_rotation: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_mid_rotation = None

    def _remap_wilor_to_humanego(self, kpts_wilor: np.ndarray) -> np.ndarray:
        kpts_wilor = np.asarray(kpts_wilor, dtype=np.float64)
        if kpts_wilor.shape[0] != 21:
            raise ValueError(f"expected 21 WiLoR keypoints, got shape {kpts_wilor.shape}")

        kpts_humanego = np.zeros((21, kpts_wilor.shape[1]), dtype=np.float64)
        for humanego_idx in range(20):
            kpts_humanego[humanego_idx] = kpts_wilor[self.WILOR_TO_HUMANEGO[humanego_idx]]
        kpts_humanego[20] = (kpts_wilor[0] + kpts_wilor[5] + kpts_wilor[9]) / 3.0
        return kpts_humanego

    def _build_wrist_pose(self, kpts_humanego: np.ndarray) -> np.ndarray | None:
        wrist = kpts_humanego[5]
        palm = kpts_humanego[20]
        index_mcp = kpts_humanego[8]
        middle_mcp = kpts_humanego[11]

        y_axis = normalize(palm - wrist)
        if y_axis is None:
            return None
        lateral = index_mcp - middle_mcp
        x_axis = normalize(np.cross(y_axis, lateral))
        if x_axis is None:
            return None
        z_axis = normalize(np.cross(x_axis, y_axis))
        if z_axis is None:
            return None
        y_axis = normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None
        return make_pose(np.column_stack([x_axis, y_axis, z_axis]), wrist)

    def _build_midpoint_pose(
        self,
        kpts_humanego: np.ndarray,
        fallback_rotation: np.ndarray | None = None,
        prev_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        thumb_tip = kpts_humanego[0]
        index_tip = kpts_humanego[1]
        thumb_base = kpts_humanego[6]
        index_base = kpts_humanego[8]
        wrist = kpts_humanego[5]

        midpoint = (thumb_tip + index_tip) * 0.5
        base_midpoint = (thumb_base + index_base) * 0.5

        x_axis = normalize(index_base - thumb_base)
        arm = base_midpoint - wrist
        if x_axis is not None and float(np.linalg.norm(arm)) >= 1e-5:
            y_proj = arm - float(np.dot(arm, x_axis)) * x_axis
            y_axis = normalize(y_proj)
            if y_axis is not None:
                z_axis = normalize(np.cross(x_axis, y_axis))
                if z_axis is not None:
                    y_axis = normalize(np.cross(z_axis, x_axis))
                    if y_axis is not None:
                        if prev_rotation is not None and float(np.dot(prev_rotation[:, 0], x_axis)) < 0.0:
                            x_axis = -x_axis
                            y_axis = -y_axis
                            z_axis = np.cross(x_axis, y_axis)
                        rotation = np.column_stack([x_axis, y_axis, z_axis])
                        return make_pose(rotation, midpoint), rotation

        rotation = prev_rotation if prev_rotation is not None else fallback_rotation
        if rotation is None:
            return None, prev_rotation
        return make_pose(rotation, midpoint), rotation

    def _compute_grasp_state(self, kpts_humanego: np.ndarray) -> int:
        thumb_tip = kpts_humanego[0]
        index_tip = kpts_humanego[1]
        wrist = kpts_humanego[5]
        middle_mcp = kpts_humanego[11]
        tip_distance = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(middle_mcp - wrist))
        if palm_size > 0.01:
            return int(tip_distance / palm_size < self.grasp_ratio_threshold)
        return int(tip_distance < 0.105)

    def __call__(
        self,
        kpts_3d_wilor: np.ndarray,
        *,
        confidence: float | None = None,
        is_right: bool | None = None,
    ) -> GripperTarget | None:
        kpts_humanego = self._remap_wilor_to_humanego(kpts_3d_wilor)
        wrist_pose = self._build_wrist_pose(kpts_humanego)
        prev_rotation = self._prev_mid_rotation if self.keep_rotation_sign_consistency else None
        T_hand_in_cam, mid_rotation = self._build_midpoint_pose(
            kpts_humanego,
            fallback_rotation=wrist_pose[:3, :3] if wrist_pose is not None else None,
            prev_rotation=prev_rotation,
        )
        if T_hand_in_cam is None:
            return None

        if self.keep_rotation_sign_consistency:
            self._prev_mid_rotation = mid_rotation

        T_eef_in_cam = T_hand_in_cam @ self.T_eef_from_hand
        grasp_state = self._compute_grasp_state(kpts_humanego)
        return GripperTarget(
            mode="humanego",
            T_hand_in_cam=T_hand_in_cam,
            T_eef_in_cam=T_eef_in_cam,
            grasp_state=grasp_state,
            humanego_kpts_3d=kpts_humanego,
            wrist_pose_in_cam=wrist_pose,
            confidence=confidence,
            is_right=is_right,
        )

    def from_hand_record(self, hand: dict[str, Any]) -> GripperTarget | None:
        kpts = hand.get("kpts_3d")
        if kpts is None:
            kpts = hand.get("wilor_kpts_3d")
        if kpts is None:
            return None
        return self(
            np.array(kpts, dtype=np.float64),
            confidence=hand.get("confidence"),
            is_right=hand.get("is_right"),
        )
