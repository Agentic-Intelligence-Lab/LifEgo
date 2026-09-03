#!/usr/bin/env python3
"""Scene / sensor assets used for HumanEgo <-> ARX AC one conversion.

Convention notes:
  - ARX base: +X is the normal forward workspace direction, +Z is up.
  - Camera optical (OpenCV): +x image-right, +y image-down, +z optical forward.
  - HEAD camera is mounted on the middle pillar and looks 45 deg downward
    toward the +X workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import struct

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EGO2EXE_ARX_ROOT = Path(__file__).resolve().parent

ARX_ACONE_URDF_ROOT = REPO_ROOT / "thirdparty" / "ARX_Model" / "AC one" / "URDF" / "ACone"
MUJOCO_ARX_SCENE = EGO2EXE_ARX_ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"

# base_link.STL contains the rectangular support base. Its bottom is at about
# z=-0.085 m and its top/mounting plane is around z=0 in the robot base frame.
ARX_BASE_BOTTOM_Z_M = -0.085
ARX_TABLE_TOP_Z_M = ARX_BASE_BOTTOM_Z_M
ARX_TABLE_HALF_THICKNESS_M = 0.025
ARX_FLOOR_BELOW_TABLE_M = 0.70


def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def empty_vec3() -> np.ndarray:
    return np.full(3, np.nan, dtype=np.float64)


def empty_mat3() -> np.ndarray:
    return np.full((3, 3), np.nan, dtype=np.float64)


def _read_stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) >= 84:
        tri_count = struct.unpack("<I", raw[80:84])[0]
        expected_len = 84 + tri_count * 50
        if expected_len == len(raw):
            vertices = np.empty((tri_count * 3, 3), dtype=np.float64)
            offset = 84
            out_i = 0
            for _ in range(tri_count):
                offset += 12
                for _ in range(3):
                    vertices[out_i] = struct.unpack("<3f", raw[offset : offset + 12])
                    offset += 12
                    out_i += 1
                offset += 2
            return vertices

    vertices = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"Could not read STL vertices from {path}")
    return np.asarray(vertices, dtype=np.float64)


def estimate_head_camera_extrinsics_from_base_mesh(
    arx_root: Path = ARX_ACONE_URDF_ROOT,
    *,
    pitch_down_deg: float = 45.0,
    head_min_z_m: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate HEAD camera pose from the high center component in base_link.STL.

    The AC one URDF does not expose a separate camera link. The no-handle
    base_link mesh contains the middle head/camera component, whose high vertices
    identify the mounted camera housing. Position is estimated at the front face
    center of that high component. Orientation uses the stated mechanical
    convention: optical +Z points toward +X and down by pitch_down_deg.

    Returns:
      T_cam_in_base, selected_bounds_min, selected_bounds_max.
    """

    mesh_path = arx_root / "meshes" / "base_link.STL"
    vertices = _read_stl_vertices(mesh_path)
    high = vertices[vertices[:, 2] > head_min_z_m]
    if len(high) == 0:
        raise ValueError(f"No head/camera vertices above z={head_min_z_m} in {mesh_path}")

    bounds_min = high.min(axis=0)
    bounds_max = high.max(axis=0)
    pitch = np.deg2rad(float(pitch_down_deg))

    cam_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_z = np.array([np.cos(pitch), 0.0, -np.sin(pitch)], dtype=np.float64)
    cam_y = np.cross(cam_z, cam_x)
    cam_y /= np.linalg.norm(cam_y)

    T_cam_in_base = np.eye(4, dtype=np.float64)
    T_cam_in_base[:3, 0] = cam_x
    T_cam_in_base[:3, 1] = cam_y
    T_cam_in_base[:3, 2] = cam_z
    T_cam_in_base[:3, 3] = np.array(
        [
            bounds_max[0],
            0.5 * (bounds_min[1] + bounds_max[1]),
            0.5 * (bounds_min[2] + bounds_max[2]),
        ],
        dtype=np.float64,
    )
    return T_cam_in_base, bounds_min, bounds_max


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics plus distortion metadata."""

    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_model: str | None = None
    dist_coeffs: np.ndarray | None = None
    notes: str = ""

    def K(self) -> np.ndarray | None:
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
    """Camera pose relative to ARX base."""

    T_cam_in_base: np.ndarray | None = None
    camera_height_m: float | None = None
    pitch_down_deg: float | None = None
    optical_projection_base: np.ndarray | None = None
    camera_target_base: np.ndarray | None = None
    source_mesh_bounds_min_m: np.ndarray | None = None
    source_mesh_bounds_max_m: np.ndarray | None = None
    notes: str = ""

    def is_filled(self) -> bool:
        return self.T_cam_in_base is not None


@dataclass
class CameraAsset:
    name: str = "head_rgb"
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    capture: dict[str, Any] = field(default_factory=dict)


@dataclass
class TcpOffset:
    t_flange_m: np.ndarray | None = None
    R_flange: np.ndarray | None = None
    notes: str = ""


@dataclass
class RobotPlatform:
    name: str = "arx_acone"
    T_base_in_world: np.ndarray | None = None
    table_top_z_m: float = 0.0
    floor_z_m: float = -0.70
    table_center_m: np.ndarray | None = None
    table_half_size_m: np.ndarray | None = None
    workspace_center_m: np.ndarray | None = None
    workspace_half_size_m: np.ndarray | None = None
    left_tcp: TcpOffset = field(default_factory=TcpOffset)
    right_tcp: TcpOffset = field(default_factory=TcpOffset)
    gripper_open_m: float | None = None
    gripper_closed_m: float | None = None
    urdf_root: str | None = None
    notes: str = ""


@dataclass
class SceneAssets:
    name: str = ""
    platform: RobotPlatform = field(default_factory=RobotPlatform)
    cameras: dict[str, CameraAsset] = field(default_factory=dict)
    extra_transforms: dict[str, np.ndarray | None] = field(default_factory=dict)
    notes: str = ""

    def camera(self, name: str = "head_rgb") -> CameraAsset | None:
        return self.cameras.get(name)


def _make_head_camera_extrinsics() -> CameraExtrinsics:
    T_cam_in_base, bounds_min, bounds_max = estimate_head_camera_extrinsics_from_base_mesh()
    optical = T_cam_in_base[:3, 2]
    pos = T_cam_in_base[:3, 3]
    table_top_z = ARX_TABLE_TOP_Z_M
    t_table = (table_top_z - pos[2]) / optical[2]
    target = pos + t_table * optical
    return CameraExtrinsics(
        T_cam_in_base=T_cam_in_base,
        camera_height_m=float(pos[2] - table_top_z),
        pitch_down_deg=45.0,
        optical_projection_base=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        camera_target_base=target,
        source_mesh_bounds_min_m=bounds_min,
        source_mesh_bounds_max_m=bounds_max,
        notes="Estimated from URDF/ACone/meshes/base_link.STL high head/camera vertices "
        "(z > 0.20 m). Position is the high component front-face center; orientation uses "
        "the stated AC one HEAD camera mount: optical axis toward +X, pitched down 45 deg.",
    )


ARX_ACONE_HEAD_V1 = SceneAssets(
    name="arx_acone_head_v1",
    platform=RobotPlatform(
        name="arx_acone",
        T_base_in_world=None,
        table_top_z_m=ARX_TABLE_TOP_Z_M,
        floor_z_m=ARX_TABLE_TOP_Z_M - ARX_FLOOR_BELOW_TABLE_M,
        table_center_m=np.array(
            [0.30, 0.0, ARX_TABLE_TOP_Z_M - ARX_TABLE_HALF_THICKNESS_M],
            dtype=np.float64,
        ),
        table_half_size_m=np.array([0.90, 0.55, ARX_TABLE_HALF_THICKNESS_M], dtype=np.float64),
        workspace_center_m=np.array([0.40, 0.0, ARX_TABLE_TOP_Z_M], dtype=np.float64),
        workspace_half_size_m=np.array([0.32, 0.28, 0.002], dtype=np.float64),
        left_tcp=TcpOffset(
            t_flange_m=np.array([0.105, 0.0018, -0.0063], dtype=np.float64),
            R_flange=None,
            notes="Approximate midpoint between left gripper fingers in left_link6 frame.",
        ),
        right_tcp=TcpOffset(
            t_flange_m=np.array([0.105, 0.0018, -0.0063], dtype=np.float64),
            R_flange=None,
            notes="Approximate midpoint between right gripper fingers in right_link16 frame.",
        ),
        gripper_open_m=0.088,
        gripper_closed_m=0.0,
        urdf_root=str(ARX_ACONE_URDF_ROOT),
        notes="ARX AC one no-handle model. Base convention: +X forward workspace, +Z up. "
        "Vendor URDF names put left_* at positive Y and right_* at negative Y.",
    ),
    cameras={
        "head_rgb": CameraAsset(
            name="head_rgb",
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=393.030548,
                fy=392.679291,
                cx=312.918335,
                cy=240.937149,
                distortion_model="inverse_brown_conrady",
                dist_coeffs=np.array(
                    [-0.050604139, 0.056275558, 0.000794928, 0.000940709, -0.018167889],
                    dtype=np.float64,
                ),
                notes="HEAD camera RealSense D405 intrinsics for 640x480.",
            ),
            extrinsics=_make_head_camera_extrinsics(),
            capture={"device": "RealSense D405", "resolution": [640, 480]},
        ),
    },
    extra_transforms={
        "T_hand_to_ee": np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "T_ee_axis_correct": np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    },
    notes="Primary ARX AC one HumanEgo cell using the built-in HEAD camera.",
)


DEFAULT_ASSETS: SceneAssets = ARX_ACONE_HEAD_V1


def get_assets(name: str | None = None) -> SceneAssets:
    if name is None or name == DEFAULT_ASSETS.name:
        return DEFAULT_ASSETS
    registry = {
        ARX_ACONE_HEAD_V1.name: ARX_ACONE_HEAD_V1,
    }
    if name not in registry:
        raise KeyError(f"Unknown assets bundle {name!r}; known: {sorted(registry)}")
    return registry[name]


__all__ = [
    "ARX_ACONE_HEAD_V1",
    "ARX_ACONE_URDF_ROOT",
    "CameraAsset",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "DEFAULT_ASSETS",
    "MUJOCO_ARX_SCENE",
    "RobotPlatform",
    "SceneAssets",
    "TcpOffset",
    "empty_mat3",
    "empty_vec3",
    "estimate_head_camera_extrinsics_from_base_mesh",
    "eye4",
    "get_assets",
]
