#!/usr/bin/env python3
"""Overlay hand2gripper EEF targets (position/orientation/grasp) on RGB frames.

Unlike `visualize_wilor_hands.py`, which only draws the raw WiLoR 21-keypoint
skeleton, this script runs the actual `hand2gripper.HumanEgoMode` conversion
(the same one `preprocess_export_eef.py` uses) on each frame's `kpts_3d` and
draws the resulting gripper (EEF) target directly in the camera frame:

  - EEF origin, projected with the session's camera intrinsics `K`.
  - EEF axes (X=red, Y=green, Z=blue), so orientation is visible frame-to-frame.
    These are drawn *after* the same axis correction used for the base-frame
    text (unless `--no-axis-correction`), so red is always the real gripper's
    pointing/approach axis -- matching the color convention used by
    `visualize_realbot_tcp.py` for the real robot's TCP axes, so the two are
    directly comparable by eye.
  - Grasp state (open/closed) plus the thumb-tip/index-tip pinch distance that
    drives it, drawn as a small marker + line so the open/close decision is
    visually checkable against the actual finger gap.
  - A text block with idx/hand/confidence/grasp/EEF pose printed in the robot
    *base* frame (`T_ee_in_base`, same axis-corrected pose `preprocess_export_eef.py`
    exports by default) rather than the camera frame, since that's what
    actually drives the robot.
  - A small fixed-corner "compass" gizmo showing which screen direction each
    robot-base axis (X/Y/Z) points toward, since the base origin itself
    usually falls outside the camera's field of view.

Reads the same per-frame `rgb.png` + `wilor_hands.json` (or
`wilor_hands_processed.json`) layout as `visualize_wilor_hands.py`, under
`<session>/preprocess/all_data/<frame>/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS
from hand2gripper import GripperTarget, HumanEgoMode

# wilor_hands_config.json reports keypoint_order "wilor_mano_21": index 0 is
# the wrist, followed by 4 joints each for thumb/index/middle/ring/pinky.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

RIGHT_COLOR = (0, 220, 0)    # green, BGR
LEFT_COLOR = (220, 0, 220)   # magenta, BGR
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X red, Y green, Z blue (BGR)
OPEN_COLOR = (0, 220, 0)
CLOSED_COLOR = (0, 0, 220)

# preprocess_wilor_hands.build_hand_json's "depth_source" values, shortened for
# on-screen display: "pred_cam_t_full" is WiLoR's own camera-translation depth,
# "assets_intrinsics_wrist_middle_mcp_scale" is the wrist->middle_mcp physical-size
# fallback used when that prediction is missing or out of the plausible range.
DEPTH_SOURCE_LABELS = {
    "pred_cam_t_full": "cam_t",
    "assets_intrinsics_wrist_middle_mcp_scale": "scale_fallback",
}


def make_hand2gripper_mode() -> HumanEgoMode:
    """Mirror preprocess_export_eef.make_hand2gripper_mode so the visualized
    EEF target matches what the export stage actually produces."""
    t_hand_to_ee = DEFAULT_ASSETS.extra_transforms.get("T_hand_to_ee")
    t_hand_from_eef = None if t_hand_to_ee is None else np.linalg.inv(np.array(t_hand_to_ee, dtype=np.float64))
    return HumanEgoMode(T_hand_from_eef=t_hand_from_eef)


def load_camera_to_base(apply_axis_correction: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mirror preprocess_export_eef's T_cam_in_base + axis-correction lookup.

    Returns (T_cam_in_base, T_base_in_cam, axis_correction_3x3_or_None).
    """
    camera = DEFAULT_ASSETS.camera()
    if camera is None or not camera.extrinsics.is_filled():
        raise ValueError(
            "patches/assets.py DEFAULT_ASSETS camera extrinsics are not filled; "
            "cannot express EEF pose in the robot base frame."
        )
    T_cam_in_base = np.array(camera.extrinsics.T_cam_in_base, dtype=np.float64)
    T_base_in_cam = np.linalg.inv(T_cam_in_base)
    axis_correction = None
    if apply_axis_correction:
        t_ee_axis_correct = DEFAULT_ASSETS.extra_transforms.get("T_ee_axis_correct")
        if t_ee_axis_correct is not None:
            axis_correction = np.array(t_ee_axis_correct, dtype=np.float64)[:3, :3]
    return T_cam_in_base, T_base_in_cam, axis_correction


def eef_pose_in_base(
    T_eef_in_cam: np.ndarray,
    T_cam_in_base: np.ndarray,
    axis_correction: np.ndarray | None,
) -> np.ndarray:
    T_ee_in_base = T_cam_in_base @ T_eef_in_cam
    if axis_correction is not None:
        T_ee_in_base = T_ee_in_base.copy()
        T_ee_in_base[:3, :3] = T_ee_in_base[:3, :3] @ axis_correction
    return T_ee_in_base


def project(K: np.ndarray, p_cam: np.ndarray) -> tuple[int, int] | None:
    x, y, z = p_cam
    if z <= 1e-4:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return int(round(u)), int(round(v))


def draw_skeleton(frame: np.ndarray, hand: dict, color: tuple[int, int, int]) -> None:
    kpts = np.asarray(hand["kpts_2d"], dtype=np.float64)
    for a, b in HAND_BONES:
        pa, pb = kpts[a], kpts[b]
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)
    for x, y in kpts:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def draw_gripper_axes(frame: np.ndarray, K: np.ndarray, T_eef_in_cam: np.ndarray, axis_length_m: float) -> tuple[int, int] | None:
    origin = T_eef_in_cam[:3, 3]
    rotation = T_eef_in_cam[:3, :3]
    origin_2d = project(K, origin)
    if origin_2d is None:
        return None
    for axis_idx, color in enumerate(AXIS_COLORS):
        tip = origin + rotation[:, axis_idx] * axis_length_m
        tip_2d = project(K, tip)
        if tip_2d is None:
            continue
        cv2.arrowedLine(frame, origin_2d, tip_2d, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(frame, origin_2d, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, origin_2d, 5, (0, 0, 0), 1, cv2.LINE_AA)
    return origin_2d


def draw_base_axes_gizmo(
    frame: np.ndarray,
    R_base_in_cam: np.ndarray,
    center: tuple[int, int],
    radius: int = 46,
    axis_names: tuple[str, str, str] = ("Xb", "Yb", "Zb"),
) -> None:
    """Draw a fixed-position compass showing which screen direction each robot
    base axis points toward. The robot base origin itself is usually outside
    the camera's field of view, so this is a rotation-only gizmo (screen-space
    orthographic projection of R_base_in_cam's columns), not a true reprojection.
    """
    bg_radius = radius + 22
    overlay = frame.copy()
    cv2.circle(overlay, center, bg_radius, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.circle(frame, center, bg_radius, (200, 200, 200), 1, cv2.LINE_AA)

    order = np.argsort(R_base_in_cam[2, :])  # draw far-pointing axes first, near ones on top
    for axis_idx in order:
        v = R_base_in_cam[:, axis_idx]
        tip = (center[0] + int(round(v[0] * radius)), center[1] + int(round(v[1] * radius)))
        color = AXIS_COLORS[axis_idx]
        cv2.arrowedLine(frame, center, tip, color, 2, cv2.LINE_AA, tipLength=0.3)
        if v[2] < 0:  # pointing toward the camera (out of the screen)
            cv2.circle(frame, tip, 5, color, -1, cv2.LINE_AA)
        else:  # pointing away from the camera (into the screen)
            cv2.circle(frame, tip, 5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, axis_names[axis_idx], (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        frame, "robot base axes", (center[0] - bg_radius, center[1] + bg_radius + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
    )


def pinch_stats(target: GripperTarget) -> tuple[float, float, float]:
    """Reproduce hand2gripper.HumanEgoMode._compute_grasp_state's inputs for display."""
    kpts = target.humanego_kpts_3d
    thumb_tip, index_tip, wrist, middle_mcp = kpts[0], kpts[1], kpts[5], kpts[11]
    tip_distance = float(np.linalg.norm(thumb_tip - index_tip))
    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    ratio = tip_distance / palm_size if palm_size > 0.01 else float("nan")
    return tip_distance, palm_size, ratio


def draw_pinch(frame: np.ndarray, K: np.ndarray, target: GripperTarget) -> None:
    kpts = target.humanego_kpts_3d
    thumb_2d = project(K, kpts[0])
    index_2d = project(K, kpts[1])
    color = CLOSED_COLOR if target.grasp_state else OPEN_COLOR
    if thumb_2d is not None and index_2d is not None:
        cv2.line(frame, thumb_2d, index_2d, color, 2, cv2.LINE_AA)
    for pt in (thumb_2d, index_2d):
        if pt is not None:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def draw_info_block(frame: np.ndarray, lines: list[str], top_left: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = top_left
    line_h = 20
    pad = 6
    w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines) + 2 * pad
    h = line_h * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 6
        cv2.putText(frame, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_hand(
    frame: np.ndarray,
    K: np.ndarray,
    hand: dict,
    mode: HumanEgoMode,
    label: str,
    skel_color: tuple[int, int, int],
    axis_length_m: float,
    min_confidence: float,
    info_corner: tuple[int, int],
    show_skeleton: bool,
    T_cam_in_base: np.ndarray,
    axis_correction: np.ndarray | None,
) -> bool:
    confidence = float(hand.get("confidence", 0.0))
    if confidence < min_confidence:
        return False

    if show_skeleton and hand.get("kpts_2d") is not None:
        draw_skeleton(frame, hand, skel_color)

    target = mode.from_hand_record(hand)
    if target is None:
        draw_info_block(frame, [f"{label} conf={confidence:.2f}", "hand2gripper: FAILED"], info_corner, skel_color)
        return True

    draw_pinch(frame, K, target)
    T_eef_in_cam_display = target.T_eef_in_cam.copy()
    if axis_correction is not None:
        # Same local-frame correction as eef_pose_in_base, applied here in the
        # camera frame so the drawn arrows use the real-gripper axis convention
        # (red = pointing axis) instead of hand2gripper's raw HumanEgo layout.
        T_eef_in_cam_display[:3, :3] = T_eef_in_cam_display[:3, :3] @ axis_correction
    draw_gripper_axes(frame, K, T_eef_in_cam_display, axis_length_m)

    tip_distance, palm_size, ratio = pinch_stats(target)
    grasp_label = "CLOSED" if target.grasp_state else "OPEN"
    T_ee_in_base = eef_pose_in_base(target.T_eef_in_cam, T_cam_in_base, axis_correction)
    bx, by, bz = T_ee_in_base[:3, 3]
    rx, ry, rz = R.from_matrix(T_ee_in_base[:3, :3]).as_rotvec()
    corrected_tag = "corrected" if axis_correction is not None else "raw"
    depth_source = DEPTH_SOURCE_LABELS.get(hand.get("depth_source"), hand.get("depth_source") or "unknown")
    lines = [
        f"{label} conf={confidence:.2f} grasp={grasp_label}",
        f"eef_base(m,{corrected_tag})=({bx:.3f},{by:.3f},{bz:.3f})",
        f"rotvec_base(rad)=({rx:.2f},{ry:.2f},{rz:.2f})",
        f"pinch={tip_distance*100:.1f}cm palm={palm_size*100:.1f}cm ratio={ratio:.2f}",
        f"depth={depth_source}",
    ]
    text_color = CLOSED_COLOR if target.grasp_state else OPEN_COLOR
    draw_info_block(frame, lines, info_corner, text_color)
    return True


def load_config(session_dir: Path) -> dict:
    cfg_path = session_dir / "preprocess" / "wilor_hands_config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Session dir, e.g. outputs/new_pipeline/horizontal1")
    parser.add_argument("--out", default="", help="Output MP4 path. Default: <session>/preprocess/vis/hand2gripper_vis.mp4")
    parser.add_argument("--processed", action="store_true", help="Use wilor_hands_processed.json (smoothed) instead of raw wilor_hands.json")
    parser.add_argument("--fps", type=float, default=0.0, help="Override output fps. Default: fps from wilor_hands_config.json, else 30")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip a hand below this confidence")
    parser.add_argument("--axis-length-m", type=float, default=0.05, help="EEF axis arrow length in meters")
    parser.add_argument("--no-skeleton", action="store_true", help="Don't draw the raw WiLoR 21-keypoint skeleton")
    parser.add_argument(
        "--no-axis-correction", action="store_true",
        help="Print the raw camera->base EEF pose instead of the axis-corrected one preprocess_export_eef.py exports by default",
    )
    args = parser.parse_args()

    session_dir = Path(args.session)
    all_data_dir = session_dir / "preprocess" / "all_data"
    frame_dirs = sorted(p for p in all_data_dir.iterdir() if p.is_dir())
    if not frame_dirs:
        raise SystemExit(f"No frames found under {all_data_dir}")

    config = load_config(session_dir)
    fps = args.fps if args.fps > 0 else float(config.get("fps", 30.0))
    K = np.array(config.get("K"), dtype=np.float64) if config.get("K") is not None else None
    if K is None:
        raise SystemExit(f"wilor_hands_config.json under {session_dir} is missing camera intrinsics 'K'")
    json_name = "wilor_hands_processed.json" if args.processed else "wilor_hands.json"

    out_path = Path(args.out) if args.out else session_dir / "preprocess" / "vis" / "hand2gripper_vis.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_r = make_hand2gripper_mode()
    mode_l = make_hand2gripper_mode()
    T_cam_in_base, T_base_in_cam, axis_correction = load_camera_to_base(not args.no_axis_correction)

    writer: cv2.VideoWriter | None = None
    n_written = 0
    n_right = 0
    n_left = 0
    for frame_dir in frame_dirs:
        rgb_path = frame_dir / "rgb.png"
        json_path = frame_dir / json_name
        if not rgb_path.is_file() or not json_path.is_file():
            continue
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            continue
        record = json.loads(json_path.read_text())
        width = frame.shape[1]

        hand_r = record.get("hand_r")
        if hand_r is not None:
            n_right += int(render_hand(
                frame, K, hand_r, mode_r, "R", RIGHT_COLOR, args.axis_length_m,
                args.min_confidence, (12, 40), not args.no_skeleton,
                T_cam_in_base, axis_correction,
            ))

        hand_l = record.get("hand_l")
        if hand_l is not None:
            n_left += int(render_hand(
                frame, K, hand_l, mode_l, "L", LEFT_COLOR, args.axis_length_m,
                args.min_confidence, (max(width - 260, 12), 40), not args.no_skeleton,
                T_cam_in_base, axis_correction,
            ))

        cv2.putText(
            frame, frame_dir.name, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        height = frame.shape[0]
        draw_base_axes_gizmo(frame, T_base_in_cam[:3, :3], (70, height - 90))

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        n_written += 1

    if writer is not None:
        writer.release()

    print(f"[visualize_hand2gripper] frames written: {n_written} (right hand: {n_right}, left hand: {n_left})")
    print(f"[visualize_hand2gripper] output: {out_path}")


if __name__ == "__main__":
    main()
