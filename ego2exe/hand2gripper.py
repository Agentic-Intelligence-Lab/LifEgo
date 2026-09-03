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
HE_MIDDLE_TIP = 2
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

    # Hysteresis band on tip_distance / palm_size. Two thresholds rather than one:
    # a single threshold produces a burst of spurious toggles whenever the ratio
    # hovers near it, and no single value serves every task (stack_bowl's ratio
    # peaks at 0.85-0.94 while stack_object reaches 1.4, so the old 1.0 default
    # left stack_bowl permanently "closed"). These defaults score 17/20 and 28/30
    # on the two tasks; override per task rather than chasing one global value.
    DEFAULT_GRASP_CLOSE_RATIO = 0.45
    DEFAULT_GRASP_OPEN_RATIO = 0.55
    DEFAULT_GRASP_MIN_FRAMES = 1

    def __init__(
        self,
        T_hand_from_eef: np.ndarray | None = None,
        grasp_ratio_threshold: float | None = None,
        grasp_close_ratio: float | None = None,
        grasp_open_ratio: float | None = None,
        grasp_min_frames: int | None = None,
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

        # grasp_ratio_threshold is the old single-threshold knob: honour it by
        # collapsing the band onto that value, so existing callers keep working.
        if grasp_ratio_threshold is not None and grasp_close_ratio is None and grasp_open_ratio is None:
            grasp_close_ratio = grasp_open_ratio = float(grasp_ratio_threshold)
        self.grasp_close_ratio = float(
            grasp_close_ratio if grasp_close_ratio is not None else self.DEFAULT_GRASP_CLOSE_RATIO
        )
        self.grasp_open_ratio = float(
            grasp_open_ratio if grasp_open_ratio is not None else self.DEFAULT_GRASP_OPEN_RATIO
        )
        if self.grasp_open_ratio < self.grasp_close_ratio:
            raise ValueError(
                f"grasp_open_ratio ({self.grasp_open_ratio}) must be >= grasp_close_ratio "
                f"({self.grasp_close_ratio}); the open threshold is the upper edge of the band"
            )
        self.grasp_min_frames = int(
            grasp_min_frames if grasp_min_frames is not None else self.DEFAULT_GRASP_MIN_FRAMES
        )
        # Kept for callers that read it back; it has no meaning once the band is wide.
        self.grasp_ratio_threshold = 0.5 * (self.grasp_close_ratio + self.grasp_open_ratio)

        self.keep_rotation_sign_consistency = bool(keep_rotation_sign_consistency)
        self._prev_mid_rotation: np.ndarray | None = None
        self._grasp_state: int | None = None

    @property
    def mode_label(self) -> str:
        """Identifier recorded in the exported JSON, so a run is traceable to its mode."""
        return self.MODE_NAME

    def reset(self) -> None:
        self._prev_mid_rotation = None
        # The hysteresis state is per-episode; leaving it set would carry the last
        # frame of one session into the first frame of the next.
        self._grasp_state = None

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

    def _position(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        """Gripper position. HumanEgo's choice: midpoint of thumb tip and index tip."""
        return (kpts_humanego[HE_THUMB_TIP] + kpts_humanego[HE_INDEX_TIP]) * 0.5

    def _jaw_axis_raw(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        """Unnormalized open/close axis. HumanEgo's choice: thumb MCP -> index MCP.

        MCPs, not fingertips, so this does not degenerate as the tips converge
        during a pinch.
        """
        return kpts_humanego[HE_INDEX_MCP] - kpts_humanego[HE_THUMB_MCP]

    def _forward_seed(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
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
        is_right: bool | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        midpoint = self._position(kpts_humanego, is_right=is_right)

        x_axis = normalize(self._jaw_axis_raw(kpts_humanego, is_right=is_right))
        arm = self._forward_seed(kpts_humanego, is_right=is_right)
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

    def grasp_ratio(self, kpts_humanego: np.ndarray) -> float | None:
        """tip_distance / palm_size, or None when the palm is too small to trust.

        Dividing by palm size is what makes the number comparable across hand
        sizes and camera distances; the raw tip distance is not.
        """
        tip_distance = float(np.linalg.norm(kpts_humanego[0] - kpts_humanego[1]))
        palm_size = float(np.linalg.norm(kpts_humanego[11] - kpts_humanego[5]))
        return tip_distance / palm_size if palm_size > 0.01 else None

    def _compute_grasp_state(self, kpts_humanego: np.ndarray) -> int:
        """Closed/open with hysteresis, carried frame to frame.

        Below grasp_close_ratio it closes, above grasp_open_ratio it opens, and
        inside the band it holds the previous state. A single threshold flips
        repeatedly whenever the ratio sits near it, which shows up downstream as
        a burst of spurious gripper events.
        """
        ratio = self.grasp_ratio(kpts_humanego)
        if ratio is None:
            tip_distance = float(np.linalg.norm(kpts_humanego[0] - kpts_humanego[1]))
            return int(tip_distance < 0.105)

        if self._grasp_state is None:
            # Seed from whichever edge the first frame falls outside; inside the
            # band, start open - these episodes begin with the hand approaching.
            self._grasp_state = 1 if ratio < self.grasp_close_ratio else 0
        elif self._grasp_state == 0 and ratio < self.grasp_close_ratio:
            self._grasp_state = 1
        elif self._grasp_state == 1 and ratio > self.grasp_open_ratio:
            self._grasp_state = 0
        return self._grasp_state

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
            is_right=is_right,
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

    def _forward_seed(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        if self.seed_kind == "index_tip":
            return kpts_humanego[HE_INDEX_TIP] - kpts_humanego[HE_THUMB_MCP]
        centroid = (
            kpts_humanego[HE_INDEX_MCP]
            + kpts_humanego[HE_MIDDLE_MCP]
            + kpts_humanego[HE_RING_MCP]
            + kpts_humanego[HE_PINKY_MCP]
        ) / 4.0
        return centroid - kpts_humanego[HE_WRIST]


class FingerCenterForwardPrimaryMode(PinchPlaneMode):
    """Finger-center frame with the Gram-Schmidt priority reversed.

    ``PinchPlaneMode`` keeps the MCP jaw direction fixed and projects the
    forward seed off that jaw.  This ablation keeps the rigid palm-forward
    direction (wrist -> four-finger MCP centroid) fixed instead, then projects
    the MCP jaw direction off forward.  Position and grasp detection remain
    identical to HumanEgo/finger_center; only orientation construction changes.
    """

    MODE_NAME = "finger_center_f_primary"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, forward_seed="finger_mcp_centroid", **kwargs)

    @property
    def mode_label(self) -> str:
        return self.MODE_NAME

    def _build_midpoint_pose(
        self,
        kpts_humanego: np.ndarray,
        fallback_rotation: np.ndarray | None = None,
        prev_rotation: np.ndarray | None = None,
        is_right: bool | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        midpoint = self._position(kpts_humanego, is_right=is_right)

        forward = normalize(self._forward_seed(kpts_humanego, is_right=is_right))
        jaw = self._jaw_axis_raw(kpts_humanego, is_right=is_right)
        if forward is not None and float(np.linalg.norm(jaw)) >= 1e-5:
            # Reversed Gram-Schmidt: preserve forward (col1), project jaw (col0).
            jaw_proj = jaw - float(np.dot(jaw, forward)) * forward
            jaw_axis = normalize(jaw_proj)
            if jaw_axis is not None:
                up_axis = normalize(np.cross(jaw_axis, forward))
                if up_axis is not None:
                    jaw_axis = normalize(np.cross(forward, up_axis))
                    if jaw_axis is not None:
                        if prev_rotation is not None and float(
                            np.dot(prev_rotation[:, 0], jaw_axis)
                        ) < 0.0:
                            jaw_axis = -jaw_axis
                            forward = -forward
                            up_axis = np.cross(jaw_axis, forward)
                        rotation = np.column_stack([jaw_axis, forward, up_axis])
                        return make_pose(rotation, midpoint), rotation

        rotation = prev_rotation if prev_rotation is not None else fallback_rotation
        if rotation is None:
            return None, prev_rotation
        return make_pose(rotation, midpoint), rotation


class QwenMode(HumanEgoMode):
    """Hand-to-gripper definition from Qwen-RobotManip (arxiv 2606.17846, Eq. 1-2).

    A baseline alongside HumanEgoMode / PinchPlaneMode: a different published
    choice of which hand landmarks define the gripper frame, for A/B comparison
    in the eval harness. Qwen-RobotManip's own retargeting code was not released
    (see the repo README); this is rebuilt directly from the paper's formulas,
    then relabelled into this module's (jaw, forward, up) column convention so it
    drops through the same T_hand_from_eef / axis-correction pipeline as the
    other modes rather than needing its own.

    As given in the paper (``kvf`` = virtual fingertip, ``s`` = handedness sign):

        kvf = 0.7 * index_tip + 0.3 * middle_tip
        p   = (thumb_tip + kvf) / 2                    gripper position
        w   = ||thumb_tip - kvf||                      gripper width
        z   = s * normalize(thumb_tip - kvf)            jaw axis   (s: +1 right hand, -1 left)
        d   = kvf - wrist
        y   = normalize(z x d)                          gripper normal
        x   = y x z                                     approach axis

    ``_jaw_axis_raw`` below implements ``z`` with the sign flipped
    (``kvf - thumb_tip``, not ``thumb_tip - kvf``) - see its docstring for why:
    the paper's sign is tied to its own, unreleased retargeting target, and this
    codebase's calibration expects the opposite one.

    ``x = y x z`` makes ``(z, x, y)`` a right-handed orthonormal triple (cyclic:
    ``y x z = x``, ``z x x = y``), and algebraically ``x`` is exactly the
    Gram-Schmidt projection of ``d`` onto the plane orthogonal to ``z`` - the
    same operation ``_forward_seed`` + projection already perform in the base
    class. So ``(z, x, y)`` relabels straight onto this module's
    ``(jaw, forward, up)`` = ``(col0, col1, col2)``, and this mode is just the
    same three hooks with the paper's choice of landmarks, not a separate
    construction.

    Where this differs from PinchPlaneMode: the jaw axis is fingertip-to-fingertip
    (``thumb_tip - kvf``), not MCP-to-MCP, so - unlike PinchPlaneMode, which keeps
    HumanEgo's MCP jaw axis specifically to avoid this - it is expected to get
    noisier as the pinch closes and the tips converge. Note also that the grasp
    hysteresis thresholds (``DEFAULT_GRASP_CLOSE_RATIO`` / ``_OPEN_RATIO``) were
    tuned against the thumb-tip/index-tip ratio, not this mode's thumb-tip/kvf
    ratio; they carry over unchanged here and may need separate tuning.
    """

    MODE_NAME = "qwen"
    KVF_INDEX_WEIGHT = 0.7
    KVF_MIDDLE_WEIGHT = 0.3

    def _virtual_fingertip(self, kpts_humanego: np.ndarray) -> np.ndarray:
        return (
            self.KVF_INDEX_WEIGHT * kpts_humanego[HE_INDEX_TIP]
            + self.KVF_MIDDLE_WEIGHT * kpts_humanego[HE_MIDDLE_TIP]
        )

    def _position(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        return (kpts_humanego[HE_THUMB_TIP] + self._virtual_fingertip(kpts_humanego)) * 0.5

    def _jaw_axis_raw(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        # The paper writes z = s * (thumb_tip - kvf): kvf-towards-thumb. That is
        # the opposite direction from HumanEgoMode._jaw_axis_raw's thumb-towards-
        # index (index_MCP - thumb_MCP), and this module's downstream calibration
        # (DEFAULT_T_HAND_FROM_EEF, DEFAULT_C_REAL_FROM_HUMANEGO) is tuned against
        # that convention, not the paper's. Flipping the sign here - kvf minus
        # thumb_tip rather than thumb_tip minus kvf - realigns it: this axis is
        # still exactly the thumb<->virtual-fingertip line the paper specifies,
        # only which end counts as "positive" changes, matching how
        # keep_rotation_sign_consistency already treats jaw-axis sign as a
        # convention to be fixed rather than part of the geometry. Confirmed by
        # measurement: before this fix every exported episode across all three
        # tasks carried a ~150-175 deg orientation error whose fitted axis was
        # the jaw axis (col0) - a single-axis near-180 deg flip, not noise -
        # while D_pos stayed in line with the other modes (~58mm vs ~58.5mm on
        # stack_object_horizontal), narrowing the bug to this one sign.
        s = -1.0 if is_right is False else 1.0
        return s * (self._virtual_fingertip(kpts_humanego) - kpts_humanego[HE_THUMB_TIP])

    def _forward_seed(self, kpts_humanego: np.ndarray, *, is_right: bool | None = None) -> np.ndarray:
        return self._virtual_fingertip(kpts_humanego) - kpts_humanego[HE_WRIST]

    def grasp_ratio(self, kpts_humanego: np.ndarray) -> float | None:
        tip_distance = float(
            np.linalg.norm(kpts_humanego[HE_THUMB_TIP] - self._virtual_fingertip(kpts_humanego))
        )
        palm_size = float(np.linalg.norm(kpts_humanego[HE_MIDDLE_MCP] - kpts_humanego[HE_WRIST]))
        return tip_distance / palm_size if palm_size > 0.01 else None


MODES = {
    HumanEgoMode.MODE_NAME: HumanEgoMode,
    PinchPlaneMode.MODE_NAME: PinchPlaneMode,
    FingerCenterForwardPrimaryMode.MODE_NAME: FingerCenterForwardPrimaryMode,
    QwenMode.MODE_NAME: QwenMode,
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
