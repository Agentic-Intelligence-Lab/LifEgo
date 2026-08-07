#!/usr/bin/env python3
"""Scene / sensor assets used for HumanEgo ↔ Nero calibration and correction.

Central place for:
  - RGB camera intrinsics / extrinsics (OpenCV convention)
  - Robot platform geometry (table, base frame, TCP offset, …)
  - Other fixed transforms that exports and MuJoCo replay should share

Fill numbers here when calibrations are ready; callers should import from this
module instead of scattering hard-coded defaults.

Convention notes (Nero setup, current pipeline):
  - Robot base: origin at table height under the arm; +z up; +x back, -x front;
    +y robot-right, -y robot-left.
  - Camera optical (OpenCV): +x image-right, +y image-down, +z optical forward.
  - Tool-centric TCP: p_tcp = p_flange + R_flange @ [0.13, 0, 0] (link7 +X).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def empty_vec3() -> np.ndarray:
    return np.full(3, np.nan, dtype=np.float64)


def empty_mat3() -> np.ndarray:
    return np.full((3, 3), np.nan, dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole + optional radial/tangential distortion (OpenCV).

    Units: pixels for fx/fy/cx/cy; distortion coefficients OpenCV-order.
    Leave fields as None / empty until a calibration is written.
    """

    # Image size used for the calibration (not necessarily capture resolution).
    width: int | None = None
    height: int | None = None

    # Focal length and principal point.
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    # OpenCV distortion: [k1, k2, p1, p2, k3, ...]  (empty = unused)
    dist_coeffs: np.ndarray | None = None

    # Free-form notes (serial number, capture date, …).
    notes: str = ""

    def K(self) -> np.ndarray | None:
        """3×3 camera matrix, or None if incomplete."""
        if None in (self.fx, self.fy, self.cx, self.cy):
            return None
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def is_filled(self) -> bool:
        return self.K() is not None


@dataclass
class CameraExtrinsics:
    """Camera pose relative to the robot base / table.

    Prefer filling T_cam_in_base (4×4) when full SE(3) is available.
    The pitch / height / table-target fields are the softer geometric
    description used by the current export scripts — optional once SE(3)
    is measured.
    """

    # Full transform: p_cam expressed in robot base, R columns = cam axes in base.
    T_cam_in_base: np.ndarray | None = None  # (4, 4)

    # Soft geometric description (table frame):
    camera_height_m: float | None = None
    pitch_down_deg: float | None = None
    # Optical-axis direction projected onto table (xy), typically toward -x.
    optical_projection_base: np.ndarray | None = None  # (3,)
    # Table hit point of the optical axis [x, y, 0].
    camera_target_base: np.ndarray | None = None  # (3,)

    notes: str = ""

    def is_filled(self) -> bool:
        return self.T_cam_in_base is not None


@dataclass
class CameraAsset:
    """One RGB (or other) camera on the platform."""

    name: str = "scene_rgb"
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    # Capture stream metadata (fps, rotate, backend) — fill when useful.
    capture: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AprilTag board (metric ruler / base anchor)
# ---------------------------------------------------------------------------

@dataclass
class AprilTagBoard:
    """Known AprilTag geometry for scale + T_cam_in_base estimation.

    Measured / printed sizes live here; detection is in
    patches/estimate_scale_apriltag.py.
    """

    dictionary: str = "DICT_APRILTAG_36h11"
    # Physical black-square edge length in meters (required for metric PnP).
    tag_size_m: float | None = None
    # Tag ids present (empty = any).
    tag_ids: list[int] = field(default_factory=list)
    # Pose of primary tag (id = tag_ids[0] if set) in robot base.
    T_tag_in_base: np.ndarray | None = None  # (4, 4)
    notes: str = ""


@dataclass
class TcpOffset:
    """TCP definition in the flange (link7) frame.

    Tool-centric pipeline default is along flange +X; controller JSONL often
    logged along flange +Z — keep both places blank until pinned.
    """

    # Translation flange → TCP in flange frame [m].
    t_flange_m: np.ndarray | None = None  # (3,)
    # Optional fixed rotation flange → TCP (as 3×3 or leave None = identity).
    R_flange: np.ndarray | None = None  # (3, 3)
    notes: str = ""


@dataclass
class RobotPlatform:
    """Fixed geometry of one Nero (or compatible) tabletop setup."""

    name: str = "nero_table"

    # Robot base pose in world / table frame (usually identity on z=0).
    T_base_in_world: np.ndarray | None = None  # (4, 4)

    # Table plane: point + normal in base/world (z-up tables: n = [0,0,1]).
    table_point_m: np.ndarray | None = None  # (3,)
    table_normal: np.ndarray | None = None  # (3,)

    # Workcell extents / origin hints (optional for viz).
    workspace_center_m: np.ndarray | None = None  # (3,)
    workspace_half_size_m: np.ndarray | None = None  # (3,)

    # TCP definition used for IK / site:tcp.
    tcp: TcpOffset = field(default_factory=TcpOffset)

    # Gripper open/closed widths [m] for binary grasp map (optional defaults).
    gripper_open_m: float | None = None
    gripper_closed_m: float | None = None

    # URDF / mesh root on this machine (optional path string).
    urdf_root: str | None = None

    notes: str = ""


# ---------------------------------------------------------------------------
# Bundle: one calibrated cell (platform + cameras + misc)
# ---------------------------------------------------------------------------

@dataclass
class SceneAssets:
    """Everything needed to transform HumanEgo data into robot base for a cell."""

    name: str = ""
    platform: RobotPlatform = field(default_factory=RobotPlatform)
    cameras: dict[str, CameraAsset] = field(default_factory=dict)
    apriltag: AprilTagBoard = field(default_factory=AprilTagBoard)

    # Extra named rigid transforms (e.g. hand→EE, T_align) — fill later.
    # Values should be (4, 4) float64 when set.
    extra_transforms: dict[str, np.ndarray | None] = field(default_factory=dict)

    notes: str = ""

    def camera(self, name: str = "scene_rgb") -> CameraAsset | None:
        return self.cameras.get(name)


# ---------------------------------------------------------------------------
# Registry — fill entries as calibrations land
# ---------------------------------------------------------------------------

# Blank primary cell: Nero table + external scene RGB (HumanEgo capture).
# Replace Nones with measured values; do not invent numbers here.
NERO_TABLE_V1 = SceneAssets(
    name="nero_table_v1",
    platform=RobotPlatform(
        name="nero_table",
        T_base_in_world=None,
        table_point_m=None,  # e.g. np.array([0.0, 0.0, 0.0])
        table_normal=None,  # e.g. np.array([0.0, 0.0, 1.0])
        workspace_center_m=None,
        workspace_half_size_m=None,
        tcp=TcpOffset(
            t_flange_m=None,  # e.g. tool-centric [0.13, 0.0, 0.0]
            R_flange=None,
            notes="Tool-centric TCP is flange +X 0.13 m in current MuJoCo scene; "
            "controller JSONL historically used flange +Z 0.13 m.",
        ),
        gripper_open_m=None,
        gripper_closed_m=None,
        urdf_root=None,  # e.g. "/home/ymq/code/agx_arm_urdf/nero"
        notes="",
    ),
    cameras={
        "scene_rgb": CameraAsset(
            name="scene_rgb",
            intrinsics=CameraIntrinsics(
                width=None,
                height=None,
                fx=None,
                fy=None,
                cx=None,
                cy=None,
                dist_coeffs=None,
                notes="",
            ),
            extrinsics=CameraExtrinsics(
                T_cam_in_base=None,
                camera_height_m=None,
                pitch_down_deg=None,
                optical_projection_base=None,
                camera_target_base=None,
                notes="Soft extrinsic used by export_robot_eef_from_wilor.py defaults "
                "until T_cam_in_base is measured.",
            ),
            capture={},
        ),
    },
    apriltag=AprilTagBoard(
        dictionary="DICT_APRILTAG_36h11",
        tag_size_m=None,  # measure printed black square edge [m]
        tag_ids=[1],  # nero_pick_place_human currently detects id=1
        T_tag_in_base=None,  # measure tag center pose in robot base
        notes="Use patches/estimate_scale_apriltag.py to fill T_cam_in_base + scale.",
    ),
    extra_transforms={
        # Hand mid-point local frame → EE/TCP local correction (4×4), if any.
        "T_hand_to_ee": None,
        # Optional axis-align remaining after main hand→cam pipeline.
        "T_ee_axis_correct": None,
    },
    notes="Primary Nero humanego→robot cell. All numeric fields intentionally blank.",
)


# Active default for scripts that want a single entry point.
DEFAULT_ASSETS: SceneAssets = NERO_TABLE_V1


def get_assets(name: str | None = None) -> SceneAssets:
    """Look up a named asset bundle; name=None → DEFAULT_ASSETS."""
    if name is None or name == DEFAULT_ASSETS.name:
        return DEFAULT_ASSETS
    registry = {
        NERO_TABLE_V1.name: NERO_TABLE_V1,
    }
    if name not in registry:
        raise KeyError(f"Unknown assets bundle {name!r}; known: {sorted(registry)}")
    return registry[name]


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraAsset",
    "AprilTagBoard",
    "TcpOffset",
    "RobotPlatform",
    "SceneAssets",
    "NERO_TABLE_V1",
    "DEFAULT_ASSETS",
    "get_assets",
    "eye4",
    "empty_vec3",
    "empty_mat3",
]
