#!/usr/bin/env python3
"""Overlay EEF/IK/real-bot trajectories as static 3D paths for a sanity check.

Unlike `replay_eef_mujoco.py` / `replay_ik_mujoco.py` / `replay_realbot_mujoco.py`,
which step the robot (or a marker) frame-by-frame through time, this script does
not drive the robot at all. It loads up to three trajectory sources, places them
all in the fixed single-arm scene (for table/robot scale reference), and draws
each as a colored 3D polyline so you can judge by eye whether the shapes/paths
line up:

  - EEF target   (cyan)    from `robot_eef_trajectory.json`  (`T_ee_in_base`)
  - IK target    (orange)  from an IK `.npz`                 (`target_pos_m`)
  - IK achieved  (magenta) same IK `.npz`, `joint_qpos` forward-kinematics'd
                            through `site:tcp` -- what the solver actually reached
  - Real robot   (white)   a real-bot JSONL. Deliberately does NOT use the
                            controller's own `poses.tcp_pose` -- the real
                            controller's `tcp_offset_flange_frame` uses a
                            different point definition than hand2gripper.py's
                            target (see assets.py `TcpOffset` notes), so the
                            two aren't apples-to-apples. Instead this computes
                            `p_flange + R_flange @ tcp_offset_m` from each
                            sample's `poses.flange_pose`, using the same
                            flange-frame offset (`--tcp-offset-m`, default
                            0.18 m along flange +X) as `site:tcp` in the
                            MuJoCo scene, so this track is directly comparable
                            to "IK achieved". Falls back to FK'ing
                            `follower.position_rad` through `site:tcp` only if
                            a JSONL doesn't carry `poses.flange_pose`.

The robot itself stays at its home pose (never driven); the IK track is FK'd
internally just to place it in 3D, not to animate anything. Any subset of the
three sources may be supplied. `--viewer` opens an interactive window you can
freely rotate; without it, an MP4 is rendered with the camera slowly orbiting
the combined point cloud (auto-framed) so depth is still readable in a static
video.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import MUJOCO_NERO_SCENE
from replay_eef_mujoco import load_eef
from replay_ik_mujoco import ARM_JOINTS, load_ik, qpos_addrs
from replay_realbot_mujoco import load_realbot
from utils_replay import as_abs, draw_marker_path, load_runtime, require_runtime

EEF_RGBA = (0.05, 0.85, 0.95, 0.95)         # cyan
IK_TARGET_RGBA = (1.0, 0.75, 0.05, 0.95)    # orange
IK_ACHIEVED_RGBA = (0.95, 0.15, 0.85, 0.95)  # magenta
REALBOT_RGBA = (0.92, 0.92, 0.92, 0.95)     # near-white
START_RGBA = (0.15, 0.95, 0.25, 1.0)        # green: first sample of every track
END_RGBA = (0.95, 0.15, 0.15, 1.0)          # red: last sample of every track


def rgba_to_bgr255(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    r, g, b, _ = rgba
    return int(b * 255), int(g * 255), int(r * 255)


def tcp_site_id(model) -> int:
    _, mujoco = require_runtime()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if sid < 0:
        raise RuntimeError("fixed scene is missing site: tcp")
    return sid


def compute_tcp_positions(model, fk_data, arm_addrs: np.ndarray, tcp_sid: int, q_traj: np.ndarray) -> np.ndarray:
    """Forward-kinematics a joint trajectory through site:tcp. Does not touch render_data."""
    _, mujoco = require_runtime()
    q_traj = np.asarray(q_traj, dtype=np.float64)
    out = np.zeros((len(q_traj), 3), dtype=np.float64)
    for i, q in enumerate(q_traj):
        fk_data.qpos[arm_addrs] = q
        mujoco.mj_forward(model, fk_data)
        out[i] = fk_data.site_xpos[tcp_sid]
    return out


def flange_pose6_to_matrix(pose6: list[float]) -> np.ndarray:
    """`[x, y, z, roll, pitch, yaw]` (base frame) -> 4x4, matching visualize_realbot_tcp.py."""
    x, y, z, roll, pitch, yaw = pose6
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load_realbot_tcp_positions(path: Path, tcp_offset_flange: np.ndarray) -> np.ndarray | None:
    """Compute TCP xyz from each sample's `poses.flange_pose` + a flange-frame offset.

    Deliberately does NOT read the JSONL's own `poses.tcp_pose` -- the real
    controller bakes in its own `tcp_offset_flange_frame`, which is a
    different point definition than hand2gripper.py's target (see assets.py
    `TcpOffset` notes). This instead applies
    `p_flange + R_flange @ tcp_offset_flange`, the same convention `site:tcp`
    uses in the MuJoCo scene, so the result is directly comparable to the
    "IK achieved" track. Returns None if no sample carries `poses.flange_pose`,
    so the caller can fall back to FK'ing `follower.position_rad` instead.
    """
    positions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            flange_pose = rec.get("poses", {}).get("flange_pose")
            if flange_pose is None:
                continue
            T_flange = flange_pose6_to_matrix(flange_pose)
            positions.append(T_flange[:3, 3] + T_flange[:3, :3] @ tcp_offset_flange)
    if not positions:
        return None
    return np.asarray(positions, dtype=np.float64)


def load_tracks(args, model, fk_data) -> list[tuple[str, np.ndarray, tuple[float, float, float, float]]]:
    arm_addrs = qpos_addrs(model, ARM_JOINTS)
    tcp_sid = tcp_site_id(model)
    tracks: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = []

    if args.eef:
        path = as_abs(args.eef)
        if path.is_file():
            _, pos, _ = load_eef(path)
            tracks.append(("EEF target", pos, EEF_RGBA))
        else:
            print(f"[replay_traj_compare] skipping --eef: not found: {path}")

    if args.ik:
        path = as_abs(args.ik)
        if path.is_file():
            ik = load_ik(path)
            tracks.append(("IK target", np.asarray(ik["target_pos_m"], dtype=np.float64), IK_TARGET_RGBA))
            achieved = compute_tcp_positions(model, fk_data, arm_addrs, tcp_sid, ik["joint_qpos"])
            tracks.append(("IK achieved (FK)", achieved, IK_ACHIEVED_RGBA))
        else:
            print(f"[replay_traj_compare] skipping --ik: not found: {path}")

    if args.realbot:
        path = as_abs(args.realbot)
        if path.is_file():
            tcp_offset_flange = np.asarray(args.tcp_offset_m, dtype=np.float64)
            tcp_pos = load_realbot_tcp_positions(path, tcp_offset_flange)
            if tcp_pos is not None:
                tracks.append((f"Real robot (flange+{tcp_offset_flange[0]:g})", tcp_pos, REALBOT_RGBA))
            else:
                print(f"[replay_traj_compare] {path} has no poses.flange_pose; falling back to FK from follower.position_rad")
                real = load_realbot(path)
                achieved = compute_tcp_positions(model, fk_data, arm_addrs, tcp_sid, real["joint_qpos"])
                tracks.append(("Real robot (FK)", achieved, REALBOT_RGBA))
        else:
            print(f"[replay_traj_compare] skipping --realbot: not found: {path}")

    return tracks


def fit_camera(points: np.ndarray, elevation: float, azimuth: float, margin: float):
    _, mujoco = require_runtime()
    center = points.mean(axis=0)
    extent = float(np.max(np.linalg.norm(points - center, axis=1))) if len(points) else 0.3
    distance = max(extent * margin, 0.25)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def draw_tracks(scn, tracks, path_stride: int, radius: float) -> None:
    for _, pts, rgba in tracks:
        draw_marker_path(scn, pts, rgba=rgba, radius=radius, stride=path_stride, start_rgba=START_RGBA, end_rgba=END_RGBA)


def draw_legend(frame_bgr: np.ndarray, tracks, header: str) -> np.ndarray:
    cv2, _ = require_runtime()
    x, y = 18, 28
    cv2.putText(frame_bgr, header, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, header, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(frame_bgr, "robot is static (home pose); green=start red=end of every track", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, "robot is static (home pose); green=start red=end of every track", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    y += 24
    for name, pts, rgba in tracks:
        color = rgba_to_bgr255(rgba)
        cv2.rectangle(frame_bgr, (x, y - 12), (x + 18, y + 2), color, -1)
        label = f"{name}  ({len(pts)} pts)"
        cv2.putText(frame_bgr, label, (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        y += 24
    return frame_bgr


def launch_viewer(model, render_data, tracks, camera, args) -> None:
    _, mujoco = require_runtime()
    print("Launching MuJoCo viewer. Drag to rotate freely. Close the window to stop.")
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, render_data) as viewer:
        viewer.cam.type = camera.type
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_tracks(viewer.user_scn, tracks, args.path_stride, args.path_radius)
            viewer.set_texts((None, None, "Trajectory compare (static robot)", "\n".join(name for name, _, _ in tracks)))
            viewer.sync()
            time.sleep(1.0 / 30.0)


def render_mp4(model, render_data, tracks, camera, args) -> None:
    cv2, mujoco = require_runtime()
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    n_frames = max(int(round(args.seconds * args.fps)), 1)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out}")
    try:
        for i in range(n_frames):
            frac = i / max(n_frames - 1, 1)
            camera.azimuth = args.azimuth_start + args.orbit_degrees * frac
            renderer.update_scene(render_data, camera=camera)
            if hasattr(renderer, "scene"):
                draw_tracks(renderer.scene, tracks, args.path_stride, args.path_radius)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            header = f"Trajectory compare  frame {i + 1}/{n_frames}  azimuth={camera.azimuth:.0f} deg"
            writer.write(draw_legend(frame, tracks, header))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    parser.add_argument("--eef", default="outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json", help="robot_eef_trajectory.json, or '' to skip")
    parser.add_argument("--ik", default="outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz", help="IK .npz, or '' to skip")
    parser.add_argument("--realbot", default="", help="Real-bot JSONL, or '' to skip")
    parser.add_argument(
        "--tcp-offset-m",
        nargs=3,
        type=float,
        default=(0.18, 0.0, 0.0),
        help="Flange-frame offset applied to poses.flange_pose to get the real-robot TCP "
        "point (same convention as site:tcp / hand2gripper.py target; independent of the "
        "controller's own tcp_offset_flange_frame baked into poses.tcp_pose).",
    )
    parser.add_argument("--out", default="outputs/new_pipeline/replays/traj_compare.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0, help="MP4 duration; camera orbits over this time")
    parser.add_argument("--orbit-degrees", type=float, default=360.0)
    parser.add_argument("--azimuth-start", type=float, default=130.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--camera-margin", type=float, default=1.6, help="Auto-fit distance = point-cloud radius * margin")
    parser.add_argument("--path-stride", type=int, default=1, help="Draw every Nth point of each track (thin out dense trajectories)")
    parser.add_argument("--path-radius", type=float, default=0.0035)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gl-backend", choices=["auto", "glfw", "egl", "osmesa"], default="auto")
    args = parser.parse_args()

    if not (args.eef or args.ik or args.realbot):
        raise SystemExit("at least one of --eef / --ik / --realbot must be given")

    load_runtime(args.viewer, args.gl_backend)
    _, mujoco = require_runtime()

    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    fk_data = mujoco.MjData(model)
    tracks = load_tracks(args, model, fk_data)
    if not tracks:
        raise SystemExit("none of the given trajectory sources could be loaded")

    all_points = np.concatenate([pts for _, pts, _ in tracks], axis=0)
    camera = fit_camera(all_points, args.elevation, args.azimuth_start, args.camera_margin)

    render_data = mujoco.MjData(model)
    mujoco.mj_forward(model, render_data)

    if args.viewer:
        launch_viewer(model, render_data, tracks, camera, args)
    else:
        render_mp4(model, render_data, tracks, camera, args)


if __name__ == "__main__":
    main()
