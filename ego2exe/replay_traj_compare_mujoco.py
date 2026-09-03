#!/usr/bin/env python3
"""Compare raw ego EEF, corrected ego EEF, and real-robot EEF in MuJoCo.

The robot stays at its home pose. This script only draws static EEF paths in
the shared robot-base frame; it neither loads nor displays IK:

  - raw ego EEF       cyan     ``--eef``
  - corrected ego EEF orange   ``--corrected-eef``
  - real-robot EEF    white     ``--realbot``

Grasp toggles use the same definition as correction/eval: anchor k is the
sample immediately before grasp[i] changes to grasp[i+1]. The same anchor has
the same large-sphere color on all tracks, with optional thin links between
corresponding points. The HUD reports raw-to-real and corrected-to-real anchor
distances. Use ``--viewer`` for an interactive view; otherwise render an MP4.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import MUJOCO_NERO_SCENE
from utils_replay import as_abs, draw_marker_path, load_runtime, require_runtime


RAW_RGBA = (0.05, 0.85, 0.95, 0.92)
CORRECTED_RGBA = (1.0, 0.55, 0.05, 0.95)
REAL_RGBA = (0.92, 0.92, 0.92, 0.95)
START_RGBA = (0.15, 0.95, 0.25, 1.0)
END_RGBA = (0.95, 0.15, 0.15, 1.0)
ANCHOR_RGBA = (
    (1.0, 0.9, 0.05, 1.0),
    (0.75, 0.2, 1.0, 1.0),
    (0.1, 1.0, 0.45, 1.0),
    (1.0, 0.2, 0.35, 1.0),
    (0.1, 0.55, 1.0, 1.0),
)


@dataclass
class Track:
    role: str
    name: str
    points: np.ndarray
    rotations: np.ndarray
    grasp: np.ndarray
    events: np.ndarray
    rgba: tuple[float, float, float, float]

    def anchor_point(self, anchor: int) -> np.ndarray:
        return self.points[int(self.events[anchor])]

    def anchor_rotation(self, anchor: int) -> np.ndarray:
        return self.rotations[int(self.events[anchor])]

    def transition(self, anchor: int) -> tuple[int, int]:
        i = int(self.events[anchor])
        return int(self.grasp[i]), int(self.grasp[i + 1])


def grasp_events(grasp: np.ndarray) -> np.ndarray:
    binary = (np.asarray(grasp) > 0.5).astype(np.int32)
    return np.where(np.diff(binary) != 0)[0]


def load_eef_track(path: Path, name: str, rgba) -> Track:
    data = json.loads(path.read_text(encoding="utf-8"))
    points, rotations, grasp = [], [], []
    for rec in data.get("records", []):
        if not rec.get("valid") or rec.get("T_ee_in_base") is None or rec.get("grasp") is None:
            continue
        T = np.asarray(rec["T_ee_in_base"]["T"], dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            continue
        points.append(T[:3, 3])
        rotations.append(T[:3, :3])
        grasp.append(int(float(rec["grasp"]) > 0.5))
    if not points:
        raise RuntimeError(f"No valid EEF positions with grasp states in {path}")
    grasp = np.asarray(grasp, dtype=np.int32)
    return Track("ego", name, np.asarray(points), np.asarray(rotations), grasp, grasp_events(grasp), rgba)


def flange_pose6_to_matrix(pose6: list[float]) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose6
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load_real_track(path: Path, tcp_offset: np.ndarray) -> Track:
    """Read flange poses and action_grasp directly; no IK/FK fallback."""
    points, rotations, grasp = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        flange = rec.get("poses", {}).get("flange_pose")
        action = rec.get("gripper", {}).get("action_grasp")
        if flange is None or action is None:
            continue
        T = flange_pose6_to_matrix(flange)
        points.append(T[:3, 3] + T[:3, :3] @ tcp_offset)
        rotations.append(T[:3, :3])
        grasp.append(int(float(action) > 0.5))
    if not points:
        raise RuntimeError(
            f"No samples with poses.flange_pose and gripper.action_grasp in {path}"
        )
    grasp = np.asarray(grasp, dtype=np.int32)
    return Track("real", "Real robot EEF", np.asarray(points), np.asarray(rotations), grasp,
                 grasp_events(grasp), REAL_RGBA)


def load_tracks(args) -> list[Track]:
    tracks = []
    for value, name, color in (
        (args.eef, args.eef_name, RAW_RGBA),
        (args.corrected_eef, args.corrected_eef_name, CORRECTED_RGBA),
    ):
        if not value:
            continue
        path = as_abs(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
        tracks.append(load_eef_track(path, name, color))
    if args.realbot:
        path = as_abs(args.realbot)
        if not path.is_file():
            raise FileNotFoundError(f"Real robot JSONL not found: {path}")
        tracks.append(load_real_track(path, np.asarray(args.tcp_offset_m, dtype=np.float64)))
    return tracks


def resolve_anchors(tracks: list[Track], requested: list[int] | None) -> list[int]:
    anchors = list(range(min(len(t.events) for t in tracks))) if requested is None else sorted(requested)
    if any(k < 0 for k in anchors) or len(set(anchors)) != len(anchors):
        raise ValueError(f"--anchors must be unique non-negative indices, got {anchors}")
    for track in tracks:
        missing = [k for k in anchors if k >= len(track.events)]
        if missing:
            raise ValueError(
                f"{track.name} has {len(track.events)} grasp event(s); anchors {missing} do not exist"
            )
    return anchors


def fit_camera(points: np.ndarray, elevation: float, azimuth: float, margin: float):
    _, mujoco = require_runtime()
    center = points.mean(axis=0)
    extent = float(np.max(np.linalg.norm(points - center, axis=1)))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = max(extent * margin, 0.25)
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def add_sphere(scn, position: np.ndarray, rgba, radius: float) -> None:
    _, mujoco = require_runtime()
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0]), np.asarray(position), np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def add_link(scn, p0: np.ndarray, p1: np.ndarray, rgba, radius: float) -> None:
    _, mujoco = require_runtime()
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, float(radius), np.asarray(p0), np.asarray(p1)
    )
    scn.ngeom += 1


def segment_bounds(track: Track, args) -> tuple[int, int]:
    """Inclusive sample bounds, optionally selected by grasp-event indices."""
    lo = 0 if args.segment_start is None else int(track.events[args.segment_start])
    hi = len(track.points) - 1 if args.segment_end is None else int(track.events[args.segment_end])
    if hi < lo:
        raise ValueError(f"invalid segment for {track.name}: {lo}..{hi}")
    return lo, hi


def draw_pose_axes(scn, point: np.ndarray, rotation: np.ndarray, args) -> None:
    colors = ((1.0, 0.05, 0.05, 0.9), (0.05, 1.0, 0.05, 0.9), (0.1, 0.35, 1.0, 0.9))
    for axis, color in enumerate(colors):
        add_link(scn, point, point + rotation[:, axis] * args.pose_axis_length,
                 color, args.pose_axis_radius)


def draw_tracks(scn, tracks: list[Track], anchors: list[int], args) -> None:
    # Reserve visibility for anchors before dense paths consume user geometries.
    for anchor in anchors:
        color = ANCHOR_RGBA[anchor % len(ANCHOR_RGBA)]
        points = [track.anchor_point(anchor) for track in tracks]
        if not args.no_anchor_links:
            for p0, p1 in zip(points[:-1], points[1:]):
                add_link(scn, p0, p1, (*color[:3], 0.38), args.anchor_link_radius)
        for point in points:
            add_sphere(scn, point, color, args.anchor_radius)
    for track in tracks:
        lo, hi = segment_bounds(track, args)
        draw_marker_path(
            scn, track.points[lo:hi + 1], rgba=track.rgba, radius=args.path_radius,
            stride=args.path_stride, start_rgba=START_RGBA, end_rgba=END_RGBA,
        )
        pose_indices = list(range(lo, hi + 1, args.pose_stride))
        if hi not in pose_indices:
            pose_indices.append(hi)
        for i in pose_indices:
            draw_pose_axes(scn, track.points[i], track.rotations[i], args)


def anchor_lines(tracks: list[Track], anchors: list[int]) -> list[str]:
    ego = [track for track in tracks if track.role == "ego"]
    raw = ego[0] if ego else None
    corrected = ego[1] if len(ego) > 1 else None
    real = next((track for track in tracks if track.role == "real"), None)
    lines = []
    for anchor in anchors:
        reference = corrected or raw or real
        before, after = reference.transition(anchor)
        action = f"{'OPEN' if before == 0 else 'CLOSED'}->{'OPEN' if after == 0 else 'CLOSED'}"
        parts = [f"A{anchor} {action}"]
        if raw is not None and real is not None:
            d = 1000 * np.linalg.norm(raw.anchor_point(anchor) - real.anchor_point(anchor))
            dr = np.degrees((R.from_matrix(real.anchor_rotation(anchor)).inv() *
                             R.from_matrix(raw.anchor_rotation(anchor))).magnitude())
            parts.append(f"raw-real {d:.1f}mm/{dr:.1f}deg")
        if corrected is not None and real is not None:
            d = 1000 * np.linalg.norm(corrected.anchor_point(anchor) - real.anchor_point(anchor))
            dr = np.degrees((R.from_matrix(real.anchor_rotation(anchor)).inv() *
                             R.from_matrix(corrected.anchor_rotation(anchor))).magnitude())
            parts.append(f"corrected-real {d:.1f}mm/{dr:.1f}deg")
        lines.append("   ".join(parts))
    return lines


def rgba_to_bgr(rgba) -> tuple[int, int, int]:
    r, g, b, _ = rgba
    return int(255 * b), int(255 * g), int(255 * r)


def draw_legend(frame: np.ndarray, tracks: list[Track], anchors: list[int], header: str) -> np.ndarray:
    cv2, _ = require_runtime()
    x, y = 18, 28

    def text(line: str, color=(245, 245, 245), scale=0.52) -> None:
        nonlocal y
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)
        y += 24

    text(header, scale=0.58)
    text("segment only; axes: X=red Y=green Z=blue; large spheres=anchors", (220, 220, 220), 0.48)
    for track in tracks:
        cv2.rectangle(frame, (x, y - 12), (x + 18, y + 2), rgba_to_bgr(track.rgba), -1)
        text(f"      {track.name} ({len(track.points)} pts, {len(track.events)} events)")
    for anchor, line in zip(anchors, anchor_lines(tracks, anchors)):
        cv2.circle(frame, (x + 9, y - 6), 7,
                   rgba_to_bgr(ANCHOR_RGBA[anchor % len(ANCHOR_RGBA)]), -1, cv2.LINE_AA)
        text(f"      {line}")
    return frame


def viewer_text(tracks: list[Track], anchors: list[int]) -> str:
    lines = [f"{t.name}: {len(t.points)} pts / {len(t.events)} events" for t in tracks]
    return "\n".join(lines + anchor_lines(tracks, anchors))


def launch_viewer(model, data, tracks: list[Track], anchors: list[int], camera, args) -> None:
    print("Launching MuJoCo viewer. Drag to rotate; close the window to stop.")
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = camera.type
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_tracks(viewer.user_scn, tracks, anchors, args)
            viewer.set_texts((None, None, "EEF trajectory compare", viewer_text(tracks, anchors)))
            viewer.sync()
            time.sleep(1 / 30)


def render_mp4(model, data, tracks: list[Track], anchors: list[int], camera, args) -> None:
    cv2, mujoco = require_runtime()
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    n_frames = max(int(round(args.seconds * args.fps)), 1)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")
    try:
        for i in range(n_frames):
            camera.azimuth = args.azimuth_start + args.orbit_degrees * i / max(n_frames - 1, 1)
            renderer.update_scene(data, camera=camera)
            if hasattr(renderer, "scene"):
                draw_tracks(renderer.scene, tracks, anchors, args)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            header = f"EEF compare {i + 1}/{n_frames} azimuth={camera.azimuth:.0f}deg"
            writer.write(draw_legend(frame, tracks, anchors, header))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    parser.add_argument("--eef", default="", help="Raw ego robot_eef_trajectory.json")
    parser.add_argument("--corrected-eef", default="", help="Corrected ego robot_eef_trajectory.json")
    parser.add_argument("--eef-name", default="Raw ego EEF")
    parser.add_argument("--corrected-eef-name", default="Corrected ego EEF")
    parser.add_argument("--realbot", default="", help="Real-robot episode JSONL")
    parser.add_argument("--anchors", nargs="*", type=int, default=None,
                        help="Event indices to mark; default: all anchors common to every track")
    parser.add_argument("--segment-start", type=int, default=None,
                        help="draw from this grasp-event index (inclusive)")
    parser.add_argument("--segment-end", type=int, default=None,
                        help="draw through this grasp-event index (inclusive)")
    parser.add_argument("--tcp-offset-m", nargs=3, type=float, default=(0.18, 0.0, 0.0))
    parser.add_argument("--out", default="outputs/replays/eef_raw_corrected_real_compare.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--orbit-degrees", type=float, default=360.0)
    parser.add_argument("--azimuth-start", type=float, default=130.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--camera-margin", type=float, default=1.6)
    parser.add_argument("--path-stride", type=int, default=1)
    parser.add_argument("--path-radius", type=float, default=0.0035)
    parser.add_argument("--pose-stride", type=int, default=30,
                        help="draw one EEF coordinate triad every N samples")
    parser.add_argument("--pose-axis-length", type=float, default=0.04)
    parser.add_argument("--pose-axis-radius", type=float, default=0.0013)
    parser.add_argument("--anchor-radius", type=float, default=0.012)
    parser.add_argument("--anchor-link-radius", type=float, default=0.0012)
    parser.add_argument("--no-anchor-links", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gl-backend", choices=("auto", "glfw", "egl", "osmesa"), default="auto")
    args = parser.parse_args()

    if not (args.eef or args.corrected_eef or args.realbot):
        raise SystemExit("Provide at least one of --eef, --corrected-eef, or --realbot")
    if args.path_stride <= 0 or args.pose_stride <= 0 or min(
            args.path_radius, args.anchor_radius, args.anchor_link_radius,
            args.pose_axis_length, args.pose_axis_radius) <= 0:
        raise SystemExit("Path stride and marker radii must be positive")

    tracks = load_tracks(args)
    anchors = resolve_anchors(tracks, args.anchors)
    if (args.segment_start is None) != (args.segment_end is None):
        raise SystemExit("--segment-start and --segment-end must be supplied together")
    if args.segment_start is not None:
        resolve_anchors(tracks, [args.segment_start, args.segment_end])
    print("[replay_traj_compare] loaded tracks:")
    for track in tracks:
        print(f"  {track.name}: {len(track.points)} points, {len(track.events)} grasp events")
    print(f"[replay_traj_compare] anchors: {anchors}")
    for line in anchor_lines(tracks, anchors):
        print(f"  {line}")

    load_runtime(args.viewer, args.gl_backend)
    _, mujoco = require_runtime()
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    visible_points = []
    for track in tracks:
        lo, hi = segment_bounds(track, args)
        visible_points.append(track.points[lo:hi + 1])
    points = np.concatenate(visible_points, axis=0)
    camera = fit_camera(points, args.elevation, args.azimuth_start, args.camera_margin)
    if args.viewer:
        launch_viewer(model, data, tracks, anchors, camera, args)
    else:
        render_mp4(model, data, tracks, anchors, camera, args)


if __name__ == "__main__":
    main()
