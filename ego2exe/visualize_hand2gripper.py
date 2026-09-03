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
  - With ``--eef``, the exact exported/debounced grasp state used by correction,
    an anchor-event banner around every toggle, and a bottom timeline marking
    A0/A1/etc.  The anchor coordinate is the frame immediately before the
    transition; the following frame is labelled as the new-state frame.
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
from hand2gripper import MODES, GripperTarget, HumanEgoMode, PinchPlaneMode, make_mode

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

EVENT_COLOR = (0, 215, 255)  # amber, BGR
TEXT_COLOR = (245, 245, 245)


def load_eef_grasp_events(path: Path) -> tuple[dict[int, int], list[dict]]:
    """Load the exact exported/debounced grasp sequence used by correction.

    An anchor is the valid trajectory sample immediately *before* a binary
    transition, matching ``eval.traj_metrics.grasp_events`` and
    ``correct_eef_with_real_anchors.py``.  Keeping both before/after frame IDs
    lets the video distinguish the anchor coordinate from the first frame in
    the new state.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[tuple[int, int]] = []
    for rec in data.get("records", []):
        if not rec.get("valid") or rec.get("grasp") is None:
            continue
        samples.append((int(rec["idx"]), int(float(rec["grasp"]) > 0.5)))
    if not samples:
        raise ValueError(f"No valid grasp samples found in EEF trajectory: {path}")

    states = {idx: state for idx, state in samples}
    events = []
    for i in range(len(samples) - 1):
        before_frame, before = samples[i]
        after_frame, after = samples[i + 1]
        if before == after:
            continue
        events.append({
            "anchor_index": len(events),
            "before_frame": before_frame,
            "after_frame": after_frame,
            "before_state": before,
            "after_state": after,
        })
    return states, events


def state_name(state: int | None) -> str:
    if state is None:
        return "N/A"
    return "CLOSED" if state else "OPEN"


def draw_grasp_event_overlay(
    frame: np.ndarray,
    *,
    frame_idx: int,
    state: int | None,
    events: list[dict],
    selected_anchors: set[int],
    hold_frames: int,
    frame_range: tuple[int, int] | None,
    source_label: str,
) -> None:
    """Draw current state, an event flash, and an anchor timeline."""
    h, w = frame.shape[:2]
    color = CLOSED_COLOR if state else OPEN_COLOR if state is not None else (180, 180, 180)
    status = f"GRASP: {state_name(state)}  [{source_label}]"
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    x0 = max((w - tw) // 2 - 12, 4)
    x1 = min(x0 + tw + 24, w - 4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 8), (x1, 8 + th + 18), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, 8), (x1, 8 + th + 18), color, 2, cv2.LINE_AA)
    cv2.putText(frame, status, (x0 + 12, 8 + th + 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, color, 2, cv2.LINE_AA)

    nearby = [
        event for event in events
        if event["anchor_index"] in selected_anchors
        and event["before_frame"] - hold_frames <= frame_idx <= event["after_frame"] + hold_frames
    ]
    if nearby:
        event = min(
            nearby,
            key=lambda item: min(abs(frame_idx - item["before_frame"]), abs(frame_idx - item["after_frame"])),
        )
        anchor = event["anchor_index"]
        exact_anchor = frame_idx == event["before_frame"]
        changed = frame_idx == event["after_frame"]
        if exact_anchor:
            phase = "ANCHOR COORDINATE (transition after this frame)"
        elif changed:
            phase = "GRASP CHANGED (first frame in new state)"
        elif frame_idx < event["before_frame"]:
            phase = f"approaching anchor in {event['before_frame'] - frame_idx} frame(s)"
        else:
            phase = f"{frame_idx - event['after_frame']} frame(s) after transition"
        title = (
            f"ANCHOR[{anchor}] / EVENT {anchor + 1}: "
            f"{state_name(event['before_state'])} -> {state_name(event['after_state'])}"
        )
        subtitle = (
            f"{phase}   frames {event['before_frame']} -> {event['after_frame']}"
        )
        scale = 0.68 if w >= 900 else 0.52
        title_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        sub_w = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        box_w = min(max(title_w, sub_w) + 28, w - 16)
        bx0 = max((w - box_w) // 2, 8)
        by0 = 58
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx0, by0), (bx0 + box_w, by0 + 62), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        thickness = 4 if exact_anchor or changed else 2
        cv2.rectangle(frame, (bx0, by0), (bx0 + box_w, by0 + 62), EVENT_COLOR,
                      thickness, cv2.LINE_AA)
        cv2.putText(frame, title, (bx0 + 14, by0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, EVENT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (bx0 + 14, by0 + 49), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, TEXT_COLOR, 1, cv2.LINE_AA)

    if frame_range is None:
        return
    first_frame, last_frame = frame_range
    span = max(last_frame - first_frame, 1)
    tx0, tx1, ty = 150, max(w - 30, 151), h - 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx0 - 8, ty - 23), (tx1 + 8, ty + 18), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (tx0, ty), (tx1, ty), (170, 170, 170), 2, cv2.LINE_AA)

    def timeline_x(idx: int) -> int:
        alpha = np.clip((idx - first_frame) / span, 0.0, 1.0)
        return int(round(tx0 + alpha * (tx1 - tx0)))

    for event in events:
        anchor = event["anchor_index"]
        if anchor not in selected_anchors:
            continue
        x = timeline_x(event["before_frame"])
        cv2.line(frame, (x, ty - 10), (x, ty + 10), EVENT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, f"A{anchor}", (x - 9, ty - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, EVENT_COLOR, 1, cv2.LINE_AA)
    current_x = timeline_x(frame_idx)
    cv2.circle(frame, (current_x, ty), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "grasp anchors", (12, ty + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, TEXT_COLOR, 1, cv2.LINE_AA)


def make_hand2gripper_mode(
    mode: str = HumanEgoMode.MODE_NAME,
    forward_seed: str | None = None,
) -> HumanEgoMode:
    """Mirror preprocess_export_eef.make_hand2gripper_mode so the visualized
    EEF target matches what the export stage actually produces."""
    t_hand_to_ee = DEFAULT_ASSETS.extra_transforms.get("T_hand_to_ee")
    t_hand_from_eef = None if t_hand_to_ee is None else np.linalg.inv(np.array(t_hand_to_ee, dtype=np.float64))
    return make_mode(mode, forward_seed=forward_seed, T_hand_from_eef=t_hand_from_eef)


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
) -> tuple[bool, int | None]:
    confidence = float(hand.get("confidence", 0.0))
    if confidence < min_confidence:
        return False, None

    if show_skeleton and hand.get("kpts_2d") is not None:
        draw_skeleton(frame, hand, skel_color)

    target = mode.from_hand_record(hand)
    if target is None:
        draw_info_block(frame, [f"{label} conf={confidence:.2f}", "hand2gripper: FAILED"], info_corner, skel_color)
        return True, None

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
    return True, int(target.grasp_state)


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
    parser.add_argument(
        "--hand2gripper-mode", default=None, choices=sorted(MODES),
        help="Hand-to-gripper definition. With --eef, infer it from trajectory metadata; "
             f"otherwise default to {HumanEgoMode.MODE_NAME}.",
    )
    parser.add_argument(
        "--forward-seed", default=None, choices=list(PinchPlaneMode.FORWARD_SEEDS),
        help=f"Forward-axis seed for pinch_plane (default: {PinchPlaneMode.DEFAULT_FORWARD_SEED})",
    )
    parser.add_argument(
        "--eef", default="",
        help="Optional robot_eef_trajectory.json whose exported/debounced grasp events define "
             "the anchors. Pass a corrected trajectory to show exactly the anchors used by correction.",
    )
    parser.add_argument(
        "--anchors", type=int, nargs="*", default=None,
        help="Only highlight these zero-based grasp-event indices (default: all events).",
    )
    parser.add_argument(
        "--event-hold-frames", type=int, default=12,
        help="Show the event banner this many frames before/after a transition (default: 12).",
    )
    parser.add_argument(
        "--event-hand", choices=("hand_r", "hand_l"), default="hand_r",
        help="Hand used for the live grasp overlay when --eef is omitted (default: hand_r).",
    )
    parser.add_argument(
        "--no-grasp-event-overlay", action="store_true",
        help="Disable the large grasp-state/event banner and bottom anchor timeline.",
    )
    args = parser.parse_args()

    if args.event_hold_frames < 0:
        raise SystemExit("--event-hold-frames must be non-negative")
    if args.anchors is not None and (
        any(k < 0 for k in args.anchors) or len(set(args.anchors)) != len(args.anchors)
    ):
        raise SystemExit(f"--anchors must be unique non-negative indices, got {args.anchors}")

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

    eef_states: dict[int, int] = {}
    grasp_events: list[dict] = []
    if args.eef:
        eef_path = Path(args.eef)
        if not eef_path.is_file():
            raise SystemExit(f"--eef trajectory not found: {eef_path}")
        eef_states, grasp_events = load_eef_grasp_events(eef_path)
        eef_metadata = json.loads(eef_path.read_text(encoding="utf-8")).get("metadata", {})
        mode_spec = str(eef_metadata.get("hand2gripper_mode") or "")
        if args.hand2gripper_mode is None and mode_spec:
            mode_name, _, mode_detail = mode_spec.partition(":")
            if mode_name in MODES:
                args.hand2gripper_mode = mode_name
            if args.forward_seed is None and mode_detail in PinchPlaneMode.FORWARD_SEEDS:
                args.forward_seed = mode_detail
        if args.anchors is not None:
            missing = [k for k in args.anchors if k >= len(grasp_events)]
            if missing:
                raise SystemExit(
                    f"--eef contains {len(grasp_events)} grasp event(s), so anchors {missing} do not exist"
                )
        print(f"[visualize_hand2gripper] EEF grasp events: {len(grasp_events)} from {eef_path}")
        for event in grasp_events:
            print(
                f"  anchor[{event['anchor_index']}] frame {event['before_frame']} -> "
                f"{event['after_frame']}: {state_name(event['before_state'])} -> "
                f"{state_name(event['after_state'])}"
            )
    if args.hand2gripper_mode is None:
        args.hand2gripper_mode = HumanEgoMode.MODE_NAME
    print(
        f"[visualize_hand2gripper] hand2gripper mode: {args.hand2gripper_mode}"
        + (f":{args.forward_seed}" if args.hand2gripper_mode == PinchPlaneMode.MODE_NAME and args.forward_seed else "")
    )
    selected_anchors = set(args.anchors) if args.anchors is not None else set(range(len(grasp_events)))

    out_path = Path(args.out) if args.out else session_dir / "preprocess" / "vis" / "hand2gripper_vis.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_r = make_hand2gripper_mode(args.hand2gripper_mode, args.forward_seed)
    mode_l = make_hand2gripper_mode(args.hand2gripper_mode, args.forward_seed)
    T_cam_in_base, T_base_in_cam, axis_correction = load_camera_to_base(not args.no_axis_correction)

    writer: cv2.VideoWriter | None = None
    n_written = 0
    n_right = 0
    n_left = 0
    eef_last_state: int | None = None
    live_last_state: int | None = None
    live_last_frame: int | None = None
    live_events: list[dict] = []
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
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            frame_idx = n_written

        hand_r = record.get("hand_r")
        state_r = None
        if hand_r is not None:
            rendered, state_r = render_hand(
                frame, K, hand_r, mode_r, "R", RIGHT_COLOR, args.axis_length_m,
                args.min_confidence, (12, 40), not args.no_skeleton,
                T_cam_in_base, axis_correction,
            )
            n_right += int(rendered)

        hand_l = record.get("hand_l")
        state_l = None
        if hand_l is not None:
            rendered, state_l = render_hand(
                frame, K, hand_l, mode_l, "L", LEFT_COLOR, args.axis_length_m,
                args.min_confidence, (max(width - 260, 12), 40), not args.no_skeleton,
                T_cam_in_base, axis_correction,
            )
            n_left += int(rendered)

        cv2.putText(
            frame, frame_dir.name, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        height = frame.shape[0]
        draw_base_axes_gizmo(frame, T_base_in_cam[:3, :3], (70, height - 90))

        if not args.no_grasp_event_overlay:
            if eef_states:
                if frame_idx in eef_states:
                    eef_last_state = eef_states[frame_idx]
                display_state = eef_last_state
                display_events = grasp_events
                display_selected = selected_anchors
                frame_range = (min(eef_states), max(eef_states))
                source_label = "exported/debounced EEF"
            else:
                display_state = state_r if args.event_hand == "hand_r" else state_l
                if display_state is not None:
                    if live_last_state is not None and display_state != live_last_state:
                        live_events.append({
                            "anchor_index": len(live_events),
                            "before_frame": live_last_frame if live_last_frame is not None else frame_idx - 1,
                            "after_frame": frame_idx,
                            "before_state": live_last_state,
                            "after_state": display_state,
                        })
                    live_last_state = display_state
                    live_last_frame = frame_idx
                display_events = live_events
                display_selected = (
                    set(args.anchors) if args.anchors is not None else set(range(len(live_events)))
                )
                try:
                    frame_range = (int(frame_dirs[0].name), int(frame_dirs[-1].name))
                except ValueError:
                    frame_range = (0, len(frame_dirs) - 1)
                source_label = f"live {args.event_hand} (not debounced)"
            draw_grasp_event_overlay(
                frame,
                frame_idx=frame_idx,
                state=display_state,
                events=display_events,
                selected_anchors=display_selected,
                hold_frames=args.event_hold_frames,
                frame_range=frame_range,
                source_label=source_label,
            )

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
