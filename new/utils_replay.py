#!/usr/bin/env python3
"""Shared MuJoCo replay helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

cv2 = None
mujoco = None


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_runtime(viewer: bool, gl_backend: str) -> None:
    global cv2, mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    elif viewer:
        os.environ.setdefault("MUJOCO_GL", "glfw")

    import cv2 as cv2_module
    import mujoco as mujoco_module

    if not hasattr(mujoco_module, "Renderer"):
        from mujoco.rendering.classic.renderer import Renderer

        mujoco_module.Renderer = Renderer
    if viewer:
        import mujoco.viewer  # noqa: F401

    cv2 = cv2_module
    mujoco = mujoco_module


def require_runtime():
    if cv2 is None or mujoco is None:
        raise RuntimeError("call load_runtime() before using MuJoCo replay helpers")
    return cv2, mujoco


def quat_xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


def _add_capsule(scn, mujoco_module, p0, p1, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco_module.mjtCatBit.mjCAT_DECOR
    mujoco_module.mjv_connector(
        geom,
        mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(scn, mujoco_module, pos, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        mujoco_module.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco_module.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def draw_marker_path(
    scn,
    points,
    *,
    rgba=(0.05, 0.75, 1.0, 0.72),
    radius: float = 0.004,
    stride: int = 1,
    start_rgba=(0.0, 1.0, 1.0, 1.0),
    end_rgba=(0.25, 0.1, 1.0, 1.0),
) -> None:
    """Draw a lightweight trajectory overlay in a MuJoCo user scene."""
    _, mujoco_module = require_runtime()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return
    step = max(int(stride), 1)
    sampled = pts[::step]
    if len(sampled) == 0 or not np.allclose(sampled[-1], pts[-1]):
        sampled = np.vstack([sampled, pts[-1]])
    for p0, p1 in zip(sampled[:-1], sampled[1:]):
        _add_capsule(scn, mujoco_module, p0, p1, rgba, radius)
    _add_sphere(scn, mujoco_module, pts[0], start_rgba, radius * 3.0)
    _add_sphere(scn, mujoco_module, pts[-1], end_rgba, radius * 3.0)
