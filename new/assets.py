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
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = REPO_ROOT / "new"
MUJOCO_NERO_SCENE = NEW_ROOT / "assets" / "mujoco_nero_scene" / "scene.xml"
MUJOCO_NERO_DUAL_SCENE = NEW_ROOT / "assets" / "mujoco_nero_scene" / "scene_dual_compare.xml"


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
        table_point_m=np.array([0.0, 0.0, 0.0]),
        table_normal=np.array([0.0, 0.0, 1.0]),
        workspace_center_m=None,
        workspace_half_size_m=None,
        tcp=TcpOffset(
            t_flange_m=np.array([0.13, 0.0, 0.0]),
            R_flange=None,
            notes="Tool-centric TCP is flange +X 0.13 m in current MuJoCo scene; "
            "controller JSONL historically used flange +Z 0.13 m. "
            "Verified against site:tcp with patches/view_nero_zero_pose_frames.py.",
        ),
        # Copied from existing pipeline defaults (solve_nero_eef_ik.py --gripper-open-m /
        # --gripper-closed-m); not independently re-measured here.
        gripper_open_m=0.1,
        gripper_closed_m=0.0,
        urdf_root=None,  # handled via AGX_ARM_URDF_ROOT env var in build_nero_mujoco_scene.py
        notes="",
    ),
    cameras={
        "scene_rgb": CameraAsset(
            name="scene_rgb",
            intrinsics=CameraIntrinsics(
                width=1280,
                height=720,
                fx=909.57177734375,
                fy=908.9627075195312,
                cx=656.6723022460938,
                cy=357.9732666015625,
                dist_coeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                notes="RealSense D435I (serial 243222074905, fw 5.15.1.55), 1280x720 Color "
                "stream, factory intrinsics from patches/camera_insrinsics.json. "
                "No distortion coefficients supplied; assumed zero.",
            ),
            extrinsics=CameraExtrinsics(
                T_cam_in_base=np.array(
                    [
                        [0.03183660052227389, 0.7508783599965531, -0.6596727365565986, 0.08190806562366636],
                        [0.9982977885613242, 0.00837981341705542, 0.05771745039990049, -0.3030280283129042],
                        [0.04886671894812236, -0.6603873614901953, -0.749333421490903, 0.6040042835648946],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                camera_height_m=None,
                pitch_down_deg=None,
                optical_projection_base=None,
                camera_target_base=None,
                notes="Solved by patches/calibrate_realsense_extrinsic_from_apriltags.py from "
                "tags 1+2 top_left corners (examples/calib/tag_corners_base.json) + "
                "patches/nero_datacollect/pyAgxArm-master/output4.png, reprojection RMSE "
                "1.012 px. Full result: outputs/camera_extrinsics/camera_extrinsics.json. "
                "Supersedes the soft camera_height_m/pitch_down_deg estimate used by "
                "export_robot_eef_from_wilor.py's CLI defaults. "
                "NOTE: camera position shifted substantially vs. the prior output1.png-based "
                "solve (z: 0.831m -> 0.604m, ~23cm), and RMSE roughly doubled (0.477 -> 1.012 px) "
                "-- verify the camera/rig hasn't moved and this is the intended calibration before trusting it.",
            ),
            capture={},
        ),
    },
    apriltag=AprilTagBoard(
        dictionary="DICT_APRILTAG_36h11",
        tag_size_m=0.05,
        tag_ids=[1, 2],
        # Not filled: our calibration solves each tag's yaw jointly with the camera pose
        # (see solved_yaw_deg_by_tag in camera_extrinsics.json) but doesn't need or emit a
        # single T_tag_in_base — T_cam_in_base above was solved directly, so this is not
        # required for the current pipeline. Fill in only if some other script needs it.
        T_tag_in_base=None,
        notes="tag_corners_base.json collected via "
        "patches/nero_datacollect/pyAgxArm-master/collect_apriltag_corners.py "
        "(top_left corner only, tcp_offset=0.13 m). "
        "Use patches/estimate_scale_apriltag.py to fill VGGT depth scale separately.",
    ),
    extra_transforms={
        # = inv(export_robot_eef_from_wilor.DEFAULT_T_ALIGN); this rotation is self-inverse.
        "T_hand_to_ee": np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        # = apply_humanego_eef_axis_correction.DEFAULT_C_REAL_FROM_HUMANEGO, embedded as a
        # 4x4 (rotation only, zero translation): real x=HumanEgo z, real y=-HumanEgo y,
        # real z=HumanEgo x. Empirically observed, not derived from a calibration script.
        "T_ee_axis_correct": np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    },
    notes="Primary Nero humanego→robot cell. Camera intrinsics calibrated 2026-08-10; "
    "extrinsics re-synced 2026-08-13 from outputs/camera_extrinsics/camera_extrinsics.json "
    "(solved 2026-08-12 against output4.png). AprilTag geometry calibrated 2026-08-10; "
    "workspace_center/half_size and T_base_in_world still blank (not needed by the current "
    "pipeline).",
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
    "MUJOCO_NERO_SCENE",
    "MUJOCO_NERO_DUAL_SCENE",
    "get_assets",
    "eye4",
    "empty_vec3",
    "empty_mat3",
]
