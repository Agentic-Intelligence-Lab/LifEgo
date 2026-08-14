#!/usr/bin/env python3
"""Run WiLoR on an ordinary RGB video and export 3D hand reconstructions.

The output layout intentionally stays compatible with `preprocess_export_eef.py`:

    outputs/<video_stem>/preprocess/all_data/<frame>/rgb.png
    outputs/<video_stem>/preprocess/all_data/<frame>/wilor_hands.json

No HumanEgo/Aria camera wrapper is used here.  WiLoR inference reads each video
frame directly, and hand-to-gripper conversion belongs in `hand2gripper.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

from assets import DEFAULT_ASSETS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "outputs" / ".hf_cache"))

DEFAULT_WILOR_CACHE = REPO_ROOT / ".cache" / "wilor_mini"
WILOR_DEFAULT_CONFIDENCE = 0.85
HAND_SIZE_WRIST_TO_MIDDLE_MCP_M = 0.085


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def import_wilor_pipeline():
    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except ImportError as exc:
        raise ImportError(
            "wilor_mini is not available in this Python environment. "
            "Install the package with `pip install wilor_mini`."
        ) from exc
    return WiLorHandPose3dEstimationPipeline


def make_wilor_pipeline(pretrained_dir: Path, hand_conf: float):
    cls = import_wilor_pipeline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = cls(
        device=device,
        dtype=dtype,
        verbose=False,
        wilor_pretrained_dir=str(pretrained_dir),
    )
    return pipe, device, hand_conf


def scaled_camera_intrinsics(width: int, height: int) -> np.ndarray:
    camera = DEFAULT_ASSETS.camera()
    if camera is None or not camera.intrinsics.is_filled():
        raise ValueError("DEFAULT_ASSETS.camera().intrinsics must be filled for WiLoR depth recovery")
    intr = camera.intrinsics
    sx = width / intr.width if intr.width else 1.0
    sy = height / intr.height if intr.height else 1.0
    return np.array(
        [
            [intr.fx * sx, 0.0, intr.cx * sx],
            [0.0, intr.fy * sy, intr.cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def recover_absolute_3d_from_intrinsics(
    kpts_2d: np.ndarray,
    kpts_3d_rel: np.ndarray,
    K: np.ndarray,
) -> np.ndarray | None:
    wrist_2d = kpts_2d[0]
    middle_mcp_2d = kpts_2d[9]
    wrist_3d = kpts_3d_rel[0]
    middle_mcp_3d = kpts_3d_rel[9]

    physical_dist = float(np.linalg.norm(middle_mcp_3d - wrist_3d))
    if physical_dist < 0.01:
        physical_dist = HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

    pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
    if pixel_dist < 5.0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    focal = (fx + fy) * 0.5
    z_wrist = focal * physical_dist / pixel_dist
    if z_wrist < 0.05 or z_wrist > 3.0:
        return None

    wrist_cam = np.array(
        [
            (wrist_2d[0] - cx) * z_wrist / fx,
            (wrist_2d[1] - cy) * z_wrist / fy,
            z_wrist,
        ],
        dtype=np.float64,
    )
    kpts_3d_cam = wrist_cam[None, :] + (kpts_3d_rel - kpts_3d_rel[0:1])
    kpts_3d_cam[:, 2] = np.clip(kpts_3d_cam[:, 2], 0.01, None)
    return kpts_3d_cam


def build_hand_json(det: dict[str, Any], K: np.ndarray) -> dict[str, Any] | None:
    preds = det.get("wilor_preds")
    if preds is None or "pred_keypoints_3d" not in preds or "pred_cam_t_full" not in preds:
        return None

    kpts_3d_rel = np.asarray(preds["pred_keypoints_3d"][0], dtype=np.float64)
    cam_t = np.asarray(preds["pred_cam_t_full"][0], dtype=np.float64)
    kpts_2d = None
    if "pred_keypoints_2d" in preds:
        kpts_2d = np.asarray(preds["pred_keypoints_2d"][0], dtype=np.float64)

    kpts_3d_cam = kpts_3d_rel + cam_t[None, :]
    depth_source = "pred_cam_t_full"
    wrist_z = float(kpts_3d_cam[0, 2]) if kpts_3d_cam.shape == (21, 3) else np.inf
    if not np.all(np.isfinite(kpts_3d_cam)) or not (0.05 < wrist_z < 3.0):
        if kpts_2d is None:
            return None
        kpts_3d_cam = recover_absolute_3d_from_intrinsics(kpts_2d, kpts_3d_rel, K)
        depth_source = "assets_intrinsics_wrist_middle_mcp_scale"
    if kpts_3d_cam is None or kpts_3d_cam.shape != (21, 3) or not np.all(np.isfinite(kpts_3d_cam)):
        return None

    is_right = bool(float(det.get("is_right", 1.0)) >= 0.5)

    hand = {
        "source": "wilor_mini_direct",
        "is_right": is_right,
        "confidence": WILOR_DEFAULT_CONFIDENCE,
        "keypoint_order": "wilor_mano_21",
        "coordinate_frame": "camera",
        "depth_source": depth_source,
        "kpts_3d": to_jsonable(kpts_3d_cam),
        "kpts_2d": to_jsonable(kpts_2d),
        "pred_keypoints_3d_root_relative": to_jsonable(kpts_3d_rel),
        "pred_cam_t_full": to_jsonable(cam_t),
        "hand_bbox": det.get("hand_bbox"),
    }
    for src_key, dst_key in [
        ("pred_vertices", "wilor_vertices"),
        ("global_orient", "wilor_global_orient"),
        ("hand_pose", "wilor_hand_pose"),
        ("pred_cam", "wilor_pred_cam"),
    ]:
        if src_key in preds:
            hand[dst_key] = to_jsonable(np.asarray(preds[src_key][0], dtype=np.float64))
    return hand


def select_best(current: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if float(candidate.get("confidence", 0.0)) > float(current.get("confidence", 0.0)) else current


class WiLoRHandTrajectoryPostProcessor:
    """Clean and smooth per-frame WiLoR hand reconstruction records.

    This mirrors the trajectory-level intent of HumanEgo's WiLoR postprocess
    without adding gripper or EEF semantics to the reconstruction stage.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        min_segment_frames: int = 5,
        max_interp_gap: int = 3,
        smooth_window: int = 9,
        smooth_polyorder: int = 2,
        ema_alpha: float = 0.35,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.min_segment_frames = int(min_segment_frames)
        self.max_interp_gap = int(max_interp_gap)
        self.smooth_window = int(smooth_window)
        self.smooth_polyorder = int(smooth_polyorder)
        self.ema_alpha = float(ema_alpha)

    def __call__(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = [json.loads(json.dumps(rec)) for rec in records]
        for hand_key in ("hand_r", "hand_l"):
            self._filter_by_confidence(processed, hand_key)
            self._suppress_short_segments(processed, hand_key)
            self._interpolate_small_gaps(processed, hand_key)
            self._smooth_keypoints(processed, hand_key)
        for rec in processed:
            rec["postprocess"] = {
                "source": "WiLoRHandTrajectoryPostProcessor",
                "confidence_threshold": self.confidence_threshold,
                "min_segment_frames": self.min_segment_frames,
                "max_interp_gap": self.max_interp_gap,
                "smooth_window": self.smooth_window,
                "smooth_polyorder": self.smooth_polyorder,
                "ema_alpha": self.ema_alpha,
            }
        return processed

    def _filter_by_confidence(self, records: list[dict[str, Any]], hand_key: str) -> None:
        for rec in records:
            hand = rec.get(hand_key)
            if hand is not None and float(hand.get("confidence", 0.0)) < self.confidence_threshold:
                rec[hand_key] = None

    def _presence_segments(self, records: list[dict[str, Any]], hand_key: str) -> list[tuple[int, int]]:
        segments = []
        start = None
        for i, rec in enumerate(records):
            if rec.get(hand_key) is not None:
                if start is None:
                    start = i
            elif start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(records)))
        return segments

    def _suppress_short_segments(self, records: list[dict[str, Any]], hand_key: str) -> None:
        for start, end in self._presence_segments(records, hand_key):
            if end - start < self.min_segment_frames:
                for i in range(start, end):
                    records[i][hand_key] = None

    def _interpolate_small_gaps(self, records: list[dict[str, Any]], hand_key: str) -> None:
        present = [rec.get(hand_key) is not None for rec in records]
        indices = np.flatnonzero(present)
        if len(indices) < 2:
            return
        for left, right in zip(indices[:-1], indices[1:]):
            gap = int(right - left - 1)
            if gap <= 0 or gap > self.max_interp_gap:
                continue
            left_hand = records[int(left)][hand_key]
            right_hand = records[int(right)][hand_key]
            if not self._can_interpolate(left_hand, right_hand):
                continue
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                records[int(left + j)][hand_key] = self._interpolate_hand(left_hand, right_hand, t)

    @staticmethod
    def _can_interpolate(left_hand: dict[str, Any], right_hand: dict[str, Any]) -> bool:
        for key in ("kpts_3d", "kpts_2d", "pred_keypoints_3d_root_relative"):
            if left_hand.get(key) is None or right_hand.get(key) is None:
                return False
            if np.asarray(left_hand[key]).shape != np.asarray(right_hand[key]).shape:
                return False
        return True

    def _interpolate_hand(self, left_hand: dict[str, Any], right_hand: dict[str, Any], t: float) -> dict[str, Any]:
        out = json.loads(json.dumps(left_hand))
        out["source"] = f"{left_hand.get('source', 'wilor_mini_direct')}+postprocessed"
        out["postprocess_state"] = "interpolated"
        out["confidence"] = (1.0 - t) * float(left_hand.get("confidence", 0.0)) + t * float(right_hand.get("confidence", 0.0))
        for key in ("kpts_3d", "kpts_2d", "pred_keypoints_3d_root_relative", "pred_cam_t_full"):
            if left_hand.get(key) is not None and right_hand.get(key) is not None:
                left = np.asarray(left_hand[key], dtype=np.float64)
                right = np.asarray(right_hand[key], dtype=np.float64)
                if left.shape == right.shape:
                    out[key] = to_jsonable((1.0 - t) * left + t * right)
        out["depth_source"] = left_hand.get("depth_source")
        return out

    def _smooth_keypoints(self, records: list[dict[str, Any]], hand_key: str) -> None:
        for start, end in self._presence_segments(records, hand_key):
            segment = records[start:end]
            n = len(segment)
            if n < 3:
                continue
            for key in ("kpts_3d", "kpts_2d", "pred_keypoints_3d_root_relative"):
                if any(rec[hand_key].get(key) is None for rec in segment):
                    continue
                arr = np.asarray([rec[hand_key][key] for rec in segment], dtype=np.float64)
                smoothed = self._smooth_array(arr)
                for rec, value in zip(segment, smoothed):
                    rec[hand_key][key] = to_jsonable(value)
                    source = rec[hand_key].get("source", "wilor_mini_direct")
                    if not str(source).endswith("+postprocessed"):
                        rec[hand_key]["source"] = f"{source}+postprocessed"
                    if rec[hand_key].get("postprocess_state") != "interpolated":
                        rec[hand_key]["postprocess_state"] = "smoothed"

    def _smooth_array(self, arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(arr.shape[0], -1)
        out = flat.copy()
        window = self._valid_savgol_window(len(flat))
        if window is not None:
            out = savgol_filter(out, window_length=window, polyorder=min(self.smooth_polyorder, window - 1), axis=0, mode="interp")
        if 0.0 < self.ema_alpha < 1.0:
            ema = out.copy()
            for i in range(1, len(ema)):
                ema[i] = self.ema_alpha * out[i] + (1.0 - self.ema_alpha) * ema[i - 1]
            out = ema
        return out.reshape(arr.shape)

    def _valid_savgol_window(self, n: int) -> int | None:
        window = min(self.smooth_window, n if n % 2 == 1 else n - 1)
        min_window = self.smooth_polyorder + 2
        if min_window % 2 == 0:
            min_window += 1
        if window < min_window:
            return None
        if window % 2 == 0:
            window -= 1
        return max(window, min_window)


def process_video(video_path: Path, out_root: Path, pipe: Any, hand_conf: float) -> Path:
    session_dir = out_root / video_path.stem
    all_data_dir = session_dir / "preprocess" / "all_data"
    all_data_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    K = scaled_camera_intrinsics(width, height)

    idx = 0
    valid_r = 0
    valid_l = 0
    raw_records: list[dict[str, Any]] = []

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"WiLoR {video_path.stem}")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_dir = all_data_dir / f"{idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_dir / "rgb.png"), frame_bgr)

        hand_r = None
        hand_l = None
        try:
            detections = pipe.predict(frame_bgr, hand_conf=hand_conf)
        except Exception as exc:
            print(f"[preprocess_wilor_hands] frame {idx:05d}: WiLoR failed: {exc}")
            detections = []

        for det in detections:
            is_right = bool(float(det.get("is_right", 1.0)) >= 0.5)
            hand = build_hand_json(det, K)
            if hand is None:
                continue
            if is_right:
                hand_r = select_best(hand_r, hand)
            else:
                hand_l = select_best(hand_l, hand)

        valid_r += int(hand_r is not None)
        valid_l += int(hand_l is not None)
        ts = int(idx * 1_000_000_000 / fps)
        record = {"idx": idx, "ts": ts, "hand_r": hand_r, "hand_l": hand_l}
        raw_records.append(record)
        (frame_dir / "wilor_hands.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if idx == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    postprocessor = WiLoRHandTrajectoryPostProcessor()
    processed_records = postprocessor(raw_records)
    processed_valid_r = 0
    processed_valid_l = 0
    for rec in processed_records:
        processed_valid_r += int(rec.get("hand_r") is not None)
        processed_valid_l += int(rec.get("hand_l") is not None)
        frame_dir = all_data_dir / f"{int(rec['idx']):05d}"
        (frame_dir / "wilor_hands_processed.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    summary = {
        "source": "preprocess_wilor_hands.py",
        "source_video": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": idx,
        "valid_right_frames": valid_r,
        "valid_left_frames": valid_l,
        "processed_valid_right_frames": processed_valid_r,
        "processed_valid_left_frames": processed_valid_l,
        "coordinate_frame": "WiLoR camera frame",
        "keypoint_order": "wilor_mano_21",
        "camera_intrinsics_source": "assets",
        "K": K.tolist(),
        "raw_output_file": "wilor_hands.json",
        "processed_output_file": "wilor_hands_processed.json",
    }
    (session_dir / "preprocess" / "wilor_hands_config.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="examples/ego_nero_easy.mp4",
        help="Input RGB video path.",
    )
    parser.add_argument("--out", default="outputs", help="Output root directory.")
    parser.add_argument("--wilor-pretrained-dir", default=None, help="WiLoR model cache directory.")
    parser.add_argument(
        "--hand-conf",
        type=float,
        default=0.3,
        help="WiLoR internal YOLO hand detector confidence threshold.",
    )
    args = parser.parse_args()

    out_root = as_abs(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    pretrained_dir = as_abs(args.wilor_pretrained_dir) if args.wilor_pretrained_dir else DEFAULT_WILOR_CACHE

    pipe, device, hand_conf = make_wilor_pipeline(pretrained_dir, args.hand_conf)
    print(f"[preprocess_wilor_hands] device: {device}")
    print(f"[preprocess_wilor_hands] WiLoR pretrained dir: {pretrained_dir}")

    video_path = as_abs(args.video)
    print(f"[preprocess_wilor_hands] Processing {video_path}")
    session_dir = process_video(video_path, out_root, pipe, hand_conf)

    print("[preprocess_wilor_hands] Output:")
    print(f"  {session_dir / 'preprocess' / 'all_data'}")


if __name__ == "__main__":
    main()
