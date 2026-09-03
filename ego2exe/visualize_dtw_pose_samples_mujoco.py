#!/usr/bin/env python3
"""Visualize pose errors at uniformly sampled position-DTW correspondences.

The real-robot segment is the common reference.  For every ego method, a DTW
path is fitted using position only.  ``--samples`` real frames are selected
uniformly from the requested grasp-event segment, and the corresponding ego
frame is read from that method's DTW path.  MuJoCo shows paths, correspondence
links and local EEF axes; a sidecar JSON records the exact poses and errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import MUJOCO_NERO_SCENE
from replay_traj_compare_mujoco import add_link, add_sphere, fit_camera, rgba_to_bgr
from utils_replay import as_abs, draw_marker_path, load_runtime, require_runtime

EVAL_DIR = Path(__file__).resolve().parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from traj_metrics import dtw_pairs  # noqa: E402


REAL_RGBA = (0.95, 0.95, 0.95, 0.95)
METHOD_COLORS = {
    "finger_center": (0.05, 0.85, 0.95, 0.95),
    "qwen": (1.0, 0.55, 0.05, 0.95),
    "humanego": (0.95, 0.15, 0.75, 0.95),
}
AXIS_COLORS = (
    (1.0, 0.05, 0.05, 0.95),
    (0.05, 1.0, 0.05, 0.95),
    (0.1, 0.35, 1.0, 0.95),
)
DIRECTION_COLORS = {
    "real": (0.95, 0.95, 0.95, 1.0),
    "finger_center": (0.05, 0.85, 0.95, 1.0),
    "qwen": (1.0, 0.55, 0.05, 1.0),
    "humanego": (0.95, 0.15, 0.75, 1.0),
}


@dataclass
class PoseTrack:
    name: str
    points: np.ndarray
    rotations: np.ndarray
    grasp: np.ndarray
    events: np.ndarray
    rgba: tuple[float, float, float, float]


def grasp_events(grasp: np.ndarray) -> np.ndarray:
    return np.where(np.diff((np.asarray(grasp) > 0.5).astype(np.int32)) != 0)[0]


def load_ego(path: Path, name: str) -> PoseTrack:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points, rotations, grasp = [], [], []
    for rec in payload.get("records", []):
        pose = rec.get("T_ee_in_base")
        if not rec.get("valid") or pose is None or rec.get("grasp") is None:
            continue
        T = np.asarray(pose["T"], dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            continue
        points.append(T[:3, 3])
        rotations.append(T[:3, :3])
        grasp.append(int(float(rec["grasp"]) > 0.5))
    if not points:
        raise RuntimeError(f"no valid EEF poses in {path}")
    g = np.asarray(grasp, dtype=np.int32)
    return PoseTrack(name, np.asarray(points), np.asarray(rotations), g,
                     grasp_events(g), METHOD_COLORS[name])


def load_real(path: Path, tcp_offset_m: float) -> PoseTrack:
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
        p = np.asarray(flange[:3], dtype=np.float64)
        rot = R.from_euler("xyz", flange[3:]).as_matrix()
        points.append(p + rot @ np.array([tcp_offset_m, 0.0, 0.0]))
        rotations.append(rot)
        grasp.append(int(float(action) > 0.5))
    if not points:
        raise RuntimeError(f"no real-robot flange poses in {path}")
    g = np.asarray(grasp, dtype=np.int32)
    return PoseTrack("real", np.asarray(points), np.asarray(rotations), g,
                     grasp_events(g), REAL_RGBA)


def segment_slice(track: PoseTrack, start_event: int, end_event: int) -> slice:
    if not (0 <= start_event < len(track.events) and 0 <= end_event < len(track.events)):
        raise ValueError(
            f"{track.name} has {len(track.events)} events; cannot select "
            f"{start_event}->{end_event}"
        )
    lo, hi = int(track.events[start_event]), int(track.events[end_event])
    if hi <= lo:
        raise ValueError(f"invalid segment for {track.name}: frames {lo}->{hi}")
    return slice(lo, hi + 1)


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees((R.from_matrix(a).inv() * R.from_matrix(b)).magnitude()))


def axis_errors_deg(a: np.ndarray, b: np.ndarray) -> list[float]:
    dots = np.sum(a * b, axis=0).clip(-1.0, 1.0)
    return np.degrees(np.arccos(dots)).tolist()


def correspondences(real: PoseTrack, ego: PoseTrack, real_sl: slice, ego_sl: slice,
                    count: int) -> list[dict]:
    real_pos = real.points[real_sl]
    ego_pos = ego.points[ego_sl]
    ego_path, real_path = dtw_pairs(ego_pos, real_pos)
    targets = np.rint(np.linspace(0, len(real_pos) - 1, count)).astype(int)
    result = []
    for sample_id, real_local in enumerate(targets):
        distance = np.abs(real_path - real_local)
        candidates = np.flatnonzero(distance == distance.min())
        path_i = int(candidates[len(candidates) // 2])
        ego_local = int(ego_path[path_i])
        real_global = int(real_sl.start + real_local)
        ego_global = int(ego_sl.start + ego_local)
        real_R, ego_R = real.rotations[real_global], ego.rotations[ego_global]
        result.append({
            "sample": sample_id,
            "real_frame": real_global,
            "ego_frame": ego_global,
            "real_position_m": real.points[real_global].tolist(),
            "ego_position_m": ego.points[ego_global].tolist(),
            "real_quat_xyzw": R.from_matrix(real_R).as_quat().tolist(),
            "ego_quat_xyzw": R.from_matrix(ego_R).as_quat().tolist(),
            "position_error_mm": float(1000 * np.linalg.norm(
                real.points[real_global] - ego.points[ego_global])),
            "rotation_error_deg": geodesic_deg(real_R, ego_R),
            "axis_error_deg": dict(zip("xyz", axis_errors_deg(real_R, ego_R))),
        })
    return result


def draw_direction_arrow(scene, center: np.ndarray, direction: np.ndarray, rgba, args) -> None:
    """Draw a normalized direction from a common comparison-sphere center."""
    endpoint = center + direction * args.orientation_sphere_radius
    add_link(scene, center, endpoint, rgba, args.direction_radius)
    add_sphere(scene, endpoint, rgba, args.direction_tip_radius)


def draw_orientation_sphere(scene, center: np.ndarray, real_rotation: np.ndarray,
                            method_items: dict[str, dict], args) -> None:
    """Overlay world axes and method directions at one real reference point."""
    add_sphere(scene, center, (0.75, 0.75, 0.75, 0.13), args.orientation_sphere_radius)

    # Fixed robot-base axes, deliberately shorter/thinner than pose directions.
    for axis, rgba in enumerate(AXIS_COLORS):
        add_link(scene, center,
                 center + np.eye(3)[:, axis] * args.orientation_sphere_radius * 0.72,
                 (*rgba[:3], 0.55), args.world_axis_radius)

    axis = "xyz".index(args.orientation_axis)
    draw_direction_arrow(scene, center, real_rotation[:, axis], DIRECTION_COLORS["real"], args)
    for name in ("finger_center", "qwen"):
        rotation = R.from_quat(method_items[name]["ego_quat_xyzw"]).as_matrix()
        draw_direction_arrow(scene, center, rotation[:, axis], DIRECTION_COLORS[name], args)
    if args.show_humanego_direction:
        rotation = R.from_quat(method_items["humanego"]["ego_quat_xyzw"]).as_matrix()
        draw_direction_arrow(scene, center, rotation[:, axis], DIRECTION_COLORS["humanego"], args)


def draw_scene(scene, real: PoseTrack, methods: list[PoseTrack], slices: dict[str, slice],
               samples: dict[str, list[dict]], args) -> None:
    tracks = [real, *methods]
    for track in tracks:
        sl = slices[track.name]
        draw_marker_path(scene, track.points[sl], rgba=track.rgba, radius=args.path_radius,
                         stride=args.path_stride)

    # Every orientation is moved to the corresponding real point, so direction
    # differences can be read directly on one common normalized sphere.
    real_samples = samples[methods[0].name]
    for i, real_item in enumerate(real_samples):
        rp = np.asarray(real_item["real_position_m"])
        rr = R.from_quat(real_item["real_quat_xyzw"]).as_matrix()
        method_items = {method.name: samples[method.name][i] for method in methods}
        draw_orientation_sphere(scene, rp, rr, method_items, args)
        for method in methods:
            item = samples[method.name][i]
            ep = np.asarray(item["ego_position_m"])
            if not args.hide_position_links:
                add_link(scene, rp, ep, (*method.rgba[:3], 0.25), args.link_radius)
                add_sphere(scene, ep, (*method.rgba[:3], 0.75), args.sample_radius * 0.55)


def summary_lines(samples: dict[str, list[dict]]) -> list[str]:
    lines = []
    for name, rows in samples.items():
        pos = np.mean([x["position_error_mm"] for x in rows])
        rot = np.mean([x["rotation_error_deg"] for x in rows])
        axes = [np.mean([x["axis_error_deg"][a] for x in rows]) for a in "xyz"]
        lines.append(f"{name}: pos {pos:.1f}mm  rot {rot:.1f}deg  axes {axes[0]:.1f}/{axes[1]:.1f}/{axes[2]:.1f}")
    return lines


def draw_hud(frame: np.ndarray, tracks: list[PoseTrack], samples: dict[str, list[dict]],
             frame_label: str, args) -> np.ndarray:
    cv2, _ = require_runtime()
    direction_names = "real=white finger=cyan qwen=orange"
    if args.show_humanego_direction:
        direction_names += " humanego=magenta"
    lines = [frame_label, "position-DTW; orientation spheres are centered on real samples",
             f"fixed base axes: X=red Y=green Z=blue; comparing local {args.orientation_axis.upper()} direction",
             direction_names, *summary_lines(samples)]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (245, 245, 245), 1, cv2.LINE_AA)
        y += 23
    for track in tracks:
        cv2.rectangle(frame, (x, y - 12), (x + 18, y + 2), rgba_to_bgr(track.rgba), -1)
        cv2.putText(frame, track.name, (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (245, 245, 245), 1, cv2.LINE_AA)
        y += 22
    return frame


def render_video(model, data, camera, real, methods, slices, samples, args) -> None:
    cv2, mujoco = require_runtime()
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    n_frames = max(1, int(round(args.seconds * args.fps)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out}")
    try:
        for i in range(n_frames):
            camera.azimuth = args.azimuth_start + args.orbit_degrees * i / max(n_frames - 1, 1)
            renderer.update_scene(data, camera=camera)
            draw_scene(renderer.scene, real, methods, slices, samples, args)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            writer.write(draw_hud(frame, [real, *methods], samples,
                                  f"stack_bowl DTW pose samples {i + 1}/{n_frames}", args))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def launch_viewer(model, data, camera, real, methods, slices, samples, args) -> None:
    import mujoco.viewer
    print("Viewer: drag to orbit, scroll to zoom, close window to exit")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = camera.type
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_scene(viewer.user_scn, real, methods, slices, samples, args)
            viewer.set_texts((None, None, "DTW pose samples", "\n".join(summary_lines(samples))))
            viewer.sync()
            time.sleep(1 / 30)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--realbot", required=True, help="real-robot episode JSONL")
    p.add_argument("--finger-center", required=True, help="finger_center EEF JSON")
    p.add_argument("--qwen", required=True, help="Qwen EEF JSON")
    p.add_argument("--humanego", required=True, help="HumanEgo EEF JSON")
    p.add_argument("--segment-start", type=int, default=0)
    p.add_argument("--segment-end", type=int, default=1)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--tcp-offset-m", type=float, default=0.18)
    p.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    p.add_argument("--out", default="outputs/replays/stack_bowl_dtw_pose_samples.mp4")
    p.add_argument("--json-out", default="", help="default: <out stem>.json")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--gl-backend", choices=("auto", "glfw", "egl", "osmesa"), default="auto")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth-start", type=float, default=130.0)
    p.add_argument("--elevation", type=float, default=-25.0)
    p.add_argument("--orbit-degrees", type=float, default=270.0)
    p.add_argument("--camera-margin", type=float, default=1.7)
    p.add_argument("--path-stride", type=int, default=2)
    p.add_argument("--path-radius", type=float, default=0.0028)
    p.add_argument("--sample-radius", type=float, default=0.008)
    p.add_argument("--link-radius", type=float, default=0.0007)
    p.add_argument("--orientation-axis", choices=("x", "y", "z"), default="z",
                   help="local EEF direction compared on each sphere (default: z)")
    p.add_argument("--orientation-sphere-radius", type=float, default=0.028)
    p.add_argument("--direction-radius", type=float, default=0.0018)
    p.add_argument("--direction-tip-radius", type=float, default=0.0038)
    p.add_argument("--world-axis-radius", type=float, default=0.00065)
    p.add_argument("--show-humanego-direction", action="store_true",
                   help="also draw HumanEgo direction in magenta (hidden by default)")
    p.add_argument("--hide-position-links", action="store_true",
                   help="hide real-to-ego DTW position links for a cleaner orientation view")
    args = p.parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")

    real = load_real(as_abs(args.realbot), args.tcp_offset_m)
    methods = [
        load_ego(as_abs(args.finger_center), "finger_center"),
        load_ego(as_abs(args.qwen), "qwen"),
        load_ego(as_abs(args.humanego), "humanego"),
    ]
    slices = {real.name: segment_slice(real, args.segment_start, args.segment_end)}
    for method in methods:
        slices[method.name] = segment_slice(method, args.segment_start, args.segment_end)
    samples = {
        method.name: correspondences(real, method, slices[real.name], slices[method.name], args.samples)
        for method in methods
    }

    out = as_abs(args.out)
    json_out = as_abs(args.json_out) if args.json_out else out.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "realbot": str(as_abs(args.realbot)),
        "segment": [args.segment_start, args.segment_end],
        "samples": args.samples,
        "methods": samples,
    }
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {json_out}")
    for line in summary_lines(samples):
        print(line)

    load_runtime(args.viewer, args.gl_backend)
    _, mujoco = require_runtime()
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    visible = [real.points[slices[real.name]]]
    visible.extend(method.points[slices[method.name]] for method in methods)
    camera = fit_camera(np.concatenate(visible), args.elevation, args.azimuth_start, args.camera_margin)
    if args.viewer:
        launch_viewer(model, data, camera, real, methods, slices, samples, args)
    else:
        render_video(model, data, camera, real, methods, slices, samples, args)


if __name__ == "__main__":
    main()
