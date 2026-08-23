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


# Named indices into the HumanEgo 21-keypoint order (see WILOR_TO_HUMANEGO below).
HE_THUMB_TIP = 0
HE_INDEX_TIP = 1
HE_WRIST = 5
HE_THUMB_MCP = 6
HE_INDEX_MCP = 8
HE_MIDDLE_MCP = 11
HE_RING_MCP = 14
HE_PINKY_MCP = 17
HE_PALM_CENTER = 20


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

    This mirrors the upstream HumanEgo implementation (TX-Leo/HumanEgo,
    ``preprocess/AriaHandsTypes.py::MidpointFrameBuilder``) line for line.  Kept
    as the reference mode; see PinchPlaneMode for the corrected forward axis.
    """

    MODE_NAME = "humanego"

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

    @property
    def mode_label(self) -> str:
        """Identifier recorded in the exported JSON, so a run is traceable to its mode."""
        return self.MODE_NAME

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

    def _forward_seed(self, kpts_humanego: np.ndarray) -> np.ndarray:
        """Seed vector for the forward axis, before Gram-Schmidt against the jaw axis.

        Only the component orthogonal to the jaw axis survives the projection, so
        what this has to get right is the *plane* the seed spans with that axis,
        not the seed's own direction.

        HumanEgo's original choice: wrist -> midpoint of the thumb/index MCPs.
        See PinchPlaneMode for why that midpoint is a problematic reference.
        """
        wrist = kpts_humanego[HE_WRIST]
        base_midpoint = (kpts_humanego[HE_THUMB_MCP] + kpts_humanego[HE_INDEX_MCP]) * 0.5
        return base_midpoint - wrist

    def _build_midpoint_pose(
        self,
        kpts_humanego: np.ndarray,
        fallback_rotation: np.ndarray | None = None,
        prev_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        thumb_tip = kpts_humanego[HE_THUMB_TIP]
        index_tip = kpts_humanego[HE_INDEX_TIP]
        thumb_base = kpts_humanego[HE_THUMB_MCP]
        index_base = kpts_humanego[HE_INDEX_MCP]

        midpoint = (thumb_tip + index_tip) * 0.5

        x_axis = normalize(index_base - thumb_base)
        arm = self._forward_seed(kpts_humanego)
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
            mode=self.mode_label,
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


class PinchPlaneMode(HumanEgoMode):
    """HumanEgoMode with the thumb removed from the forward-axis reference.

    Everything else is inherited: the jaw axis stays ``index_MCP - thumb_MCP``
    (MCPs, not fingertips, so it does not degenerate as the tips meet), as does
    the grasp ratio, the sign-consistency guard and the fallback chain.

    Why the seed changes
    --------------------
    HumanEgo seeds the forward axis with ``(thumb_MCP + index_MCP)/2 - wrist``.
    The thumb MCP is a poor reference for two reasons: it does not lie in the
    plane of the other MCPs (a static tilt), and it is not rigid relative to the
    palm - the thumb's carpometacarpal joint moves it during opposition, so the
    forward axis swings as the hand opens and closes.

    Measured over 3 sessions / ~600 WiLoR frames of stack_object_horizontal,
    as pitch away from a rigid palm plane built from wrist + the four finger MCPs:

        seed                             std          swing open-vs-closed
        (thumb_MCP+index_MCP)/2 - wrist  5.0/3.6/4.0  -7.7 / -5.6 / -6.1 deg
        index_tip - thumb_MCP            2.5/1.9/2.1  +2.4 / +2.9 / +3.1 deg
        four-finger MCP centroid - wrist 0.7/1.4/0.7  +0.2 / +1.4 / +0.9 deg

    The three seeds below all agree to within ~2 deg of each other and differ
    from HumanEgo's by 5-10 deg, which is what identifies HumanEgo's as the
    outlier rather than any of these.

    ``index_tip``
        ``index_tip - thumb_MCP``, i.e. the plane through thumb MCP, index MCP
        and index tip - the plane the pinch actually happens in.  Note the
        ``thumb_MCP -> index_MCP`` leg lies exactly along the jaw axis and is
        therefore annihilated by the Gram-Schmidt projection, making this seed
        *identical* to ``index_tip - index_MCP``: the thumb does not enter the
        forward axis at all.  Fingertip noise enters, but the flexion that
        closes a pinch runs mostly along the jaw axis, which is projected out.
    ``finger_mcp_centroid``
        ``mean(index, middle, ring, pinky MCP) - wrist``.  Rigid throughout the
        grasp, so it is the steadier of the two, at the cost of describing the
        palm plane rather than the pinch plane.

    Which one transfers better is an empirical question for the eval harness
    (``rho_rot`` and the L2b per-anchor ``err_rot_deg``), not for this file.
    """

    MODE_NAME = "pinch_plane"
    FORWARD_SEEDS = ("index_tip", "finger_mcp_centroid")
    DEFAULT_FORWARD_SEED = "index_tip"

    def __init__(self, *args: Any, forward_seed: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = self.DEFAULT_FORWARD_SEED if forward_seed is None else str(forward_seed)
        if seed not in self.FORWARD_SEEDS:
            raise ValueError(f"forward_seed must be one of {self.FORWARD_SEEDS}, got {seed!r}")
        self.seed_kind = seed

    @property
    def mode_label(self) -> str:
        return f"{self.MODE_NAME}:{self.seed_kind}"

    def _forward_seed(self, kpts_humanego: np.ndarray) -> np.ndarray:
        if self.seed_kind == "index_tip":
            return kpts_humanego[HE_INDEX_TIP] - kpts_humanego[HE_THUMB_MCP]
        centroid = (
            kpts_humanego[HE_INDEX_MCP]
            + kpts_humanego[HE_MIDDLE_MCP]
            + kpts_humanego[HE_RING_MCP]
            + kpts_humanego[HE_PINKY_MCP]
        ) / 4.0
        return centroid - kpts_humanego[HE_WRIST]


MODES = {
    HumanEgoMode.MODE_NAME: HumanEgoMode,
    PinchPlaneMode.MODE_NAME: PinchPlaneMode,
}


def make_mode(
    name: str = HumanEgoMode.MODE_NAME,
    *,
    forward_seed: str | None = None,
    **kwargs: Any,
) -> HumanEgoMode:
    """Build a hand-to-gripper mode by name.  ``forward_seed`` applies to pinch_plane only."""
    try:
        cls = MODES[name]
    except KeyError:
        raise ValueError(f"unknown hand2gripper mode {name!r}; choose from {sorted(MODES)}") from None
    if cls is PinchPlaneMode:
        return cls(forward_seed=forward_seed, **kwargs)
    if forward_seed is not None:
        raise ValueError(f"--forward-seed is only meaningful for mode 'pinch_plane', not {name!r}")
    return cls(**kwargs)
