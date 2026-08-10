#!/usr/bin/env python3
"""Estimate camera-tag geometry and VGGT depth scale from AprilTags.

Detects AprilTags in RGB frames (HumanEgo session / image dir / video), solves
PnP with known tag edge length, and optionally:

  - estimates T_cam_in_base given a known T_tag_in_base
  - estimates a global metric scale factor s by comparing tag edge length
    reconstructed from a VGGT-Ω depth map against the known tag size
  - rewrites HumanEgo EEF trajectory with T_cam_in_base and/or scale
    (corrects hand EEF positions in robot base)

OpenCV: DICT_APRILTAG_36h11 (id=1) has been verified on nero_pick_place_human.

Examples:

  # Detect + PnP only (use session intrinsic from aria_cam_rgb.json when present)
  $PY patches/estimate_scale_apriltag.py \\
    --session outputs/nero_pick_place_human \\
    --tag-size-m 0.08 \\
    --out outputs/nero_pick_place_human/apriltag_calib

  # + VGGT depth scale
  $PY patches/estimate_scale_apriltag.py \\
    --session outputs/nero_pick_place_human \\
    --tag-size-m 0.08 \\
    --vggt-npz outputs/nero_pick_place_human/vggt_omega/predictions.npz \\
    --out outputs/nero_pick_place_human/apriltag_calib

  # + known tag pose in robot base → T_cam_in_base + rescale EEF
  $PY patches/estimate_scale_apriltag.py \\
    --session outputs/nero_pick_place_human \\
    --tag-size-m 0.08 \\
    --tag-in-base-json path/to/T_tag_in_base.json \\
    --eef-in outputs/nero_pick_place_human/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \\
    --eef-out outputs/nero_pick_place_human/robot_eef_apriltag_corrected \\
    --apply-scale \\
    --out outputs/nero_pick_place_human/apriltag_calib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS

REPO_ROOT = Path(__file__).resolve().parents[1]

# OpenCV ArUco / AprilTag: corners = TL, TR, BR, BL of the marker face.
# Object frame: origin at tag center, +X to the right, +Y down in image when
# tag is face-on upright, +Z out of the tag (toward camera when facing it).
DEFAULT_DICT = "DICT_APRILTAG_36h11"


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    Rm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = eye4()
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_record(T: np.ndarray) -> dict:
    quat = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": quat.tolist(),
        "rotvec": R.from_matrix(T[:3, :3]).as_rotvec().tolist(),
    }


def invert_T(T: np.ndarray) -> np.ndarray:
    R_m = T[:3, :3]
    t = T[:3, 3]
    out = eye4()
    out[:3, :3] = R_m.T
    out[:3, 3] = -R_m.T @ t
    return out


def load_K_from_aria_cam(path: Path) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(data["k"], dtype=np.float64)
    d = np.asarray(data.get("d", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    if d.size < 4:
        d = np.zeros(5, dtype=np.float64)
    h = int(data.get("h", 0))
    w = int(data.get("w", 0))
    return K, d, h, w


def estimate_K_from_vfov(width: int, height: int, vfov_deg: float) -> np.ndarray:
    fy = (height * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
    fx = fy
    return np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def K_from_assets(width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    """scene_rgb intrinsics from patches/assets.py, scaled to (width, height)."""
    intr = DEFAULT_ASSETS.camera().intrinsics
    if not intr.is_filled():
        return None
    sx = width / intr.width if intr.width else 1.0
    sy = height / intr.height if intr.height else 1.0
    K = np.array(
        [[intr.fx * sx, 0.0, intr.cx * sx], [0.0, intr.fy * sy, intr.cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64) if intr.dist_coeffs is None else intr.dist_coeffs.copy()
    return K, dist


def parse_K_cli(args: argparse.Namespace, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if args.fx is not None and args.fy is not None:
        cx = args.cx if args.cx is not None else width * 0.5
        cy = args.cy if args.cy is not None else height * 0.5
        K = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.zeros(5, dtype=np.float64)
    if args.K_json:
        data = json.loads(as_abs(args.K_json).read_text(encoding="utf-8"))
        K = np.asarray(data["K"] if "K" in data else data["k"], dtype=np.float64)
        d = np.asarray(data.get("d", data.get("dist", [0, 0, 0, 0, 0])), dtype=np.float64).reshape(-1)
        return K, d
    if args.k_fallback == "assets":
        from_assets = K_from_assets(width, height)
        if from_assets is not None:
            print("[apriltag] K from patches/assets.py (DEFAULT_ASSETS scene_rgb)")
            return from_assets
        print("[apriltag] assets camera intrinsics not filled; falling back to --vfov-deg guess.")
    return estimate_K_from_vfov(width, height, args.vfov_deg), np.zeros(5, dtype=np.float64)


def get_dictionary(name: str):
    key = name if name.startswith("DICT_") else f"DICT_{name}"
    # OpenCV accepts DICT_APRILTAG_36h11
    val = getattr(aruco, key, None)
    if val is None:
        # try uppercase H
        alt = key.replace("h", "H")
        val = getattr(aruco, alt, None)
    if val is None:
        raise ValueError(f"Unknown AprilTag/ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(val)


def marker_object_points(tag_size_m: float) -> np.ndarray:
    s = float(tag_size_m)
    # TL, TR, BR, BL matching OpenCV corner order
    return np.array(
        [
            [-s / 2, s / 2, 0.0],
            [s / 2, s / 2, 0.0],
            [s / 2, -s / 2, 0.0],
            [-s / 2, -s / 2, 0.0],
        ],
        dtype=np.float64,
    )


def collect_frames(args: argparse.Namespace) -> list[tuple[int, Path]]:
    """Return list of (frame_idx, image_path)."""
    frames: list[tuple[int, Path]] = []
    if args.session:
        all_data = as_abs(args.session) / "preprocess" / "all_data"
        if not all_data.is_dir():
            raise FileNotFoundError(f"missing all_data: {all_data}")
        for d in sorted(p for p in all_data.iterdir() if p.is_dir() and p.name.isdigit()):
            rgb = d / "rgb.png"
            if rgb.is_file():
                frames.append((int(d.name), rgb))
    elif args.images:
        img_dir = as_abs(args.images)
        paths = sorted(img_dir.glob("*/rgb.png"))
        if not paths:
            paths = sorted(
                p
                for p in img_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name != "depth.png"
            )
        for i, p in enumerate(paths):
            try:
                idx = int(p.parent.name)
            except ValueError:
                idx = i
            frames.append((idx, p))
    elif args.video:
        video = as_abs(args.video)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video}")
        tmp = as_abs("outputs/apriltag/_tmp_frames") / video.stem
        tmp.mkdir(parents=True, exist_ok=True)
        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(args.frame_stride, 1) == 0:
                out = tmp / f"{saved:06d}.png"
                cv2.imwrite(str(out), frame)
                frames.append((saved, out))
                saved += 1
                if args.max_frames > 0 and saved >= args.max_frames:
                    break
            idx += 1
        cap.release()
        return frames
    else:
        raise RuntimeError("Provide --session, --images, or --video")

    if args.frame_stride > 1:
        frames = frames[:: args.frame_stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    return frames


def solve_marker_pose(
    corners_2d: np.ndarray,
    tag_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    obj = marker_object_points(tag_size_m)
    img_pts = np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return bool(ok), np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def average_SE3(Ts: list[np.ndarray]) -> np.ndarray:
    if not Ts:
        raise ValueError("empty SE3 list")
    if len(Ts) == 1:
        return Ts[0].copy()
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    quats = R.from_matrix([T[:3, :3] for T in Ts]).as_quat()  # xyzw
    # naive mean with sign flip for hemisphere
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q = quats.mean(axis=0)
    q /= np.linalg.norm(q)
    T = eye4()
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def sample_depth_at_pixels(depth: np.ndarray, u: np.ndarray, v: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear-ish nearest sample; depth HxW or HxWx1, coords in image pixel space of K."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    dh, dw = depth.shape[:2]
    # map from image (W,H) to depth (dw,dh)
    xs = np.clip(np.round(u * (dw - 1) / max(W - 1, 1)).astype(int), 0, dw - 1)
    ys = np.clip(np.round(v * (dh - 1) / max(H - 1, 1)).astype(int), 0, dh - 1)
    return depth[ys, xs].astype(np.float64)


def unproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def scale_from_vggt_depth(
    corners_2d: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    tag_size_m: float,
) -> float | None:
    """Estimate s = L_true / L_depth from quadrilateral opposite/adjacent edges."""
    pts = corners_2d.reshape(4, 2)
    us, vs = pts[:, 0], pts[:, 1]
    zs = sample_depth_at_pixels(depth_map, us, vs, img_h, img_w)
    if np.any(~np.isfinite(zs)) or np.any(zs <= 1e-4):
        return None
    p3 = np.stack([unproject(us[i], vs[i], zs[i], K) for i in range(4)], axis=0)
    # edge lengths TL-TR, TR-BR, BR-BL, BL-TL (should ≈ tag_size)
    edges = [
        np.linalg.norm(p3[0] - p3[1]),
        np.linalg.norm(p3[1] - p3[2]),
        np.linalg.norm(p3[2] - p3[3]),
        np.linalg.norm(p3[3] - p3[0]),
    ]
    L_pred = float(np.median(edges))
    if L_pred < 1e-6:
        return None
    return float(tag_size_m / L_pred)


def load_T_tag_in_base(args: argparse.Namespace) -> np.ndarray | None:
    if args.tag_in_base_json:
        data = json.loads(as_abs(args.tag_in_base_json).read_text(encoding="utf-8"))
        if "T" in data:
            return np.asarray(data["T"], dtype=np.float64)
        if "T_tag_in_base" in data:
            return np.asarray(data["T_tag_in_base"], dtype=np.float64)
        raise ValueError("tag-in-base json must contain 'T' or 'T_tag_in_base'")
    if args.tag_origin_base is not None:
        T = eye4()
        T[:3, 3] = np.asarray(args.tag_origin_base, dtype=np.float64)
        if args.tag_rpy_deg is not None:
            roll, pitch, yaw = [math.radians(x) for x in args.tag_rpy_deg]
            T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        return T
    return None


def apply_to_eef(
    eef_path: Path,
    out_dir: Path,
    T_cam_in_base: np.ndarray | None,
    scale: float | None,
    apply_scale: bool,
    args: argparse.Namespace,
) -> dict:
    """Recompute T_ee_in_base from cam-space EE using optional scale + new extrinsics."""
    data = json.loads(eef_path.read_text(encoding="utf-8"))
    records = data.get("records", data if isinstance(data, list) else [])
    s = float(scale) if (apply_scale and scale is not None) else 1.0

    # Prefer explicit T_cam_in_base; else keep whatever T was encoded per-record
    # by reconstructing from T_ee_in_cam if present.
    n_ok = 0
    for rec in records:
        if not rec.get("valid"):
            continue
        T_ee_cam = rec.get("T_ee_in_cam")
        if T_ee_cam is None:
            continue
        T_c = np.asarray(T_ee_cam["T"] if isinstance(T_ee_cam, dict) else T_ee_cam, dtype=np.float64)
        T_c_scaled = T_c.copy()
        T_c_scaled[:3, 3] *= s
        if T_cam_in_base is not None:
            T_base = T_cam_in_base @ T_c_scaled
        else:
            # only scale existing base pose if no new extrinsic
            T_old = rec.get("T_ee_in_base")
            if T_old is None:
                continue
            T_b = np.asarray(T_old["T"] if isinstance(T_old, dict) else T_old, dtype=np.float64)
            T_base = T_b.copy()
            T_base[:3, 3] = T_b[:3, 3] * s
            # orientation unchanged; pure scale of positions relative to origin
        rec["T_ee_in_cam"] = {
            **(T_ee_cam if isinstance(T_ee_cam, dict) else {}),
            **T_to_record(T_c_scaled),
        }
        rec["T_ee_in_base"] = T_to_record(T_base)
        rec["apriltag_scale"] = s
        n_ok += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "robot_eef_trajectory.json"
    payload = data if isinstance(data, dict) else {"records": records}
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["records"] = records
        payload["apriltag_correction"] = {
            "source_eef": str(eef_path),
            "scale": s,
            "apply_scale": bool(apply_scale and scale is not None),
            "T_cam_in_base": None if T_cam_in_base is None else T_cam_in_base.tolist(),
            "tag_size_m": float(args.tag_size_m),
        }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # CSV of positions for quick plots
    csv_path = out_dir / "robot_eef_trajectory.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("idx,valid,grasp,x_m,y_m,z_m,scale\n")
        for rec in records:
            idx = rec.get("idx")
            valid = int(bool(rec.get("valid")))
            grasp = rec.get("grasp")
            T = rec.get("T_ee_in_base")
            if T is None:
                f.write(f"{idx},{valid},{grasp},,,,\n")
                continue
            t = T["translation_m"] if isinstance(T, dict) else list(np.asarray(T)[:3, 3])
            f.write(f"{idx},{valid},{grasp},{t[0]},{t[1]},{t[2]},{s}\n")

    return {"n_records": len(records), "n_corrected": n_ok, "out_json": str(out_json), "scale_applied": s}


def run(args: argparse.Namespace) -> None:
    frames = collect_frames(args)
    if not frames:
        raise RuntimeError("No frames found")

    # Intrinsics bootstrap from first frame / session cam
    sample = cv2.imread(str(frames[0][1]))
    if sample is None:
        raise RuntimeError(f"Failed to read {frames[0][1]}")
    img_h, img_w = sample.shape[:2]

    K: np.ndarray | None = None
    dist = np.zeros(5, dtype=np.float64)
    if args.session:
        cam_json = as_abs(args.session) / "preprocess" / "all_data" / f"{frames[0][0]:05d}" / "aria_cam_rgb.json"
        loaded = load_K_from_aria_cam(cam_json)
        if loaded is None:
            # try frame 0 always
            loaded = load_K_from_aria_cam(
                as_abs(args.session) / "preprocess" / "all_data" / "00000" / "aria_cam_rgb.json"
            )
        if loaded is not None:
            K, dist, _, _ = loaded
            print(f"[apriltag] K from aria_cam_rgb.json")
    if K is None:
        K, dist = parse_K_cli(args, img_w, img_h)
        print(f"[apriltag] K from CLI / vfov (vfov={args.vfov_deg})")

    dictionary = get_dictionary(args.dictionary)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    obj_pts = marker_object_points(args.tag_size_m)

    # VGGT optional
    vggt = None
    if args.vggt_npz:
        vggt_path = as_abs(args.vggt_npz)
        vggt = np.load(vggt_path, allow_pickle=True)
        print(f"[apriltag] loaded VGGT: {vggt_path}  keys={vggt.files}")

    detections = []
    Ts_tag_cam: list[np.ndarray] = []
    scales_vggt: list[float] = []
    target_ids = None if args.tag_id < 0 else {int(args.tag_id)}

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            detections.append({"idx": frame_idx, "path": str(path), "valid": False, "tags": []})
            continue

        tags_out = []
        vis = img.copy()
        aruco.drawDetectedMarkers(vis, corners, ids)

        for i, tid in enumerate(ids.ravel().tolist()):
            if target_ids is not None and tid not in target_ids:
                continue
            c2d = corners[i]
            ok, rvec, tvec = solve_marker_pose(c2d, args.tag_size_m, K, dist)
            if not ok:
                continue
            T_tag_cam = rt_to_T(rvec, tvec)
            # reprojection error
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            err = float(np.linalg.norm(proj - c2d.reshape(4, 2), axis=1).mean())

            tag_rec = {
                "id": int(tid),
                "corners_px": c2d.reshape(4, 2).tolist(),
                "tvec_m": tvec.tolist(),
                "rvec": rvec.tolist(),
                "T_tag_in_cam": T_to_record(T_tag_cam),
                "reproj_err_px": err,
                "distance_m": float(np.linalg.norm(tvec)),
            }

            # VGGT scale on this frame if depth available
            if vggt is not None and "depth" in vggt.files:
                depth_all = vggt["depth"]
                # map frame_idx → depth row: assume same order as video frames
                d_i = None
                if depth_all.shape[0] > frame_idx:
                    d_i = depth_all[frame_idx]
                elif depth_all.shape[0] == len(frames):
                    # positional match in subsampled list
                    pos = [f[0] for f in frames].index(frame_idx)
                    if pos < depth_all.shape[0]:
                        d_i = depth_all[pos]
                if d_i is not None:
                    # Prefer session K for unprojection in original image; optional VGGT K
                    K_depth = K
                    if "intrinsic" in vggt.files and args.use_vggt_K_for_depth:
                        Ki = vggt["intrinsic"]
                        if Ki.ndim == 3:
                            Ki = Ki[min(frame_idx, Ki.shape[0] - 1)]
                        # scale principal point to full image size
                        dh, dw = (d_i.shape[0], d_i.shape[1])
                        K_depth = Ki.astype(np.float64).copy()
                        # Ki is for depth resolution; unproject in depth pixel space
                        c_depth = c2d.reshape(4, 2).copy()
                        c_depth[:, 0] *= (dw - 1) / max(img_w - 1, 1)
                        c_depth[:, 1] *= (dh - 1) / max(img_h - 1, 1)
                        zs = sample_depth_at_pixels(d_i, c_depth[:, 0], c_depth[:, 1], dh, dw)
                        if np.all(np.isfinite(zs)) and np.all(zs > 1e-4):
                            p3 = np.stack(
                                [unproject(c_depth[j, 0], c_depth[j, 1], zs[j], K_depth) for j in range(4)],
                                axis=0,
                            )
                            edges = [
                                np.linalg.norm(p3[0] - p3[1]),
                                np.linalg.norm(p3[1] - p3[2]),
                                np.linalg.norm(p3[2] - p3[3]),
                                np.linalg.norm(p3[3] - p3[0]),
                            ]
                            L_pred = float(np.median(edges))
                            s_i = float(args.tag_size_m / L_pred) if L_pred > 1e-6 else None
                        else:
                            s_i = None
                    else:
                        s_i = scale_from_vggt_depth(
                            c2d, d_i, K, img_w, img_h, args.tag_size_m
                        )
                    if s_i is not None:
                        tag_rec["vggt_edge_scale"] = s_i
                        scales_vggt.append(s_i)

            tags_out.append(tag_rec)
            Ts_tag_cam.append(T_tag_cam)
            cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), args.tag_size_m * 0.5)

        detections.append(
            {
                "idx": frame_idx,
                "path": str(path),
                "valid": len(tags_out) > 0,
                "tags": tags_out,
            }
        )
        if args.save_vis and tags_out:
            cv2.imwrite(str(vis_dir / f"{frame_idx:05d}_apriltag.jpg"), vis)

    n_hit = sum(1 for d in detections if d["valid"])
    print(f"[apriltag] detections: {n_hit}/{len(detections)} frames  dict={args.dictionary}")

    if not Ts_tag_cam:
        raise RuntimeError(
            "No AprilTags detected. Check --dictionary / --tag-id / lighting. "
            f"Tried {args.dictionary}."
        )

    T_tag_cam_avg = average_SE3(Ts_tag_cam)
    T_cam_tag_avg = invert_T(T_tag_cam_avg)

    T_tag_base = load_T_tag_in_base(args)
    T_cam_base = None
    if T_tag_base is not None:
        # p_base = T_tag_base @ p_tag; p_cam = T_tag_cam @ p_tag
        # T_cam_in_base maps cam → base: p_base = T_cam_base @ p_cam
        # T_cam_base = T_tag_base @ inv(T_tag_cam)
        T_cam_base = T_tag_base @ invert_T(T_tag_cam_avg)
        print("[apriltag] computed T_cam_in_base from T_tag_in_base + PnP")

    s_vggt = None
    if scales_vggt:
        s_vggt = float(np.median(scales_vggt))
        print(
            f"[apriltag] VGGT depth scale median s={s_vggt:.4f} "
            f"(n={len(scales_vggt)}, mean={float(np.mean(scales_vggt)):.4f}, "
            f"std={float(np.std(scales_vggt)):.4f})"
        )

    # Optional: scale of PnP tag distance vs export soft-camera (not applied automatically)
    distances = [float(np.linalg.norm(T[:3, 3])) for T in Ts_tag_cam]

    summary = {
        "dictionary": args.dictionary,
        "tag_size_m": float(args.tag_size_m),
        "tag_id_filter": None if args.tag_id < 0 else int(args.tag_id),
        "n_frames": len(detections),
        "n_frames_with_tag": n_hit,
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": [img_w, img_h],
        "T_tag_in_cam_mean": T_to_record(T_tag_cam_avg),
        "T_cam_in_tag_mean": T_to_record(T_cam_tag_avg),
        "T_tag_in_base": None if T_tag_base is None else T_to_record(T_tag_base),
        "T_cam_in_base": None if T_cam_base is None else T_to_record(T_cam_base),
        "tag_distance_m": {
            "median": float(np.median(distances)),
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
        },
        "vggt_depth_scale": {
            "s_median": s_vggt,
            "n": len(scales_vggt),
            "samples": scales_vggt[:50],
        }
        if scales_vggt
        else None,
        "reproj_err_px_median": float(
            np.median(
                [
                    t["reproj_err_px"]
                    for d in detections
                    if d["valid"]
                    for t in d["tags"]
                ]
            )
        ),
        "usage": {
            "hand_eef": "Set export/solve camera from T_cam_in_base when available; "
            "multiply cam-space EE translations by vggt s if WiLoR scale matches VGGT depth, "
            "or re-run export with calibrated extrinsics.",
            "note": "PnP poses are already metric given true tag_size_m and accurate K. "
            "vggt_depth_scale corrects VGGT (and depth-aligned) geometry to tag metric.",
        },
    }

    (out_dir / "apriltag_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "detections.jsonl").write_text(
        "\n".join(json.dumps(d) for d in detections) + "\n",
        encoding="utf-8",
    )

    # Save numpy friendly
    np.savez(
        out_dir / "apriltag_calibration.npz",
        K=K,
        dist=dist,
        T_tag_in_cam_mean=T_tag_cam_avg,
        T_cam_in_tag_mean=T_cam_tag_avg,
        T_cam_in_base=np.asarray(T_cam_base) if T_cam_base is not None else np.array([]),
        vggt_scale=np.asarray(s_vggt if s_vggt is not None else np.nan),
        tag_size_m=np.asarray(args.tag_size_m),
    )

    eef_report = None
    if args.eef_in:
        eef_out = as_abs(args.eef_out) if args.eef_out else out_dir / "eef_corrected"
        # Scale for EEF: prefer explicit --eef-scale, else vggt median if --apply-scale
        scale_eef = args.eef_scale
        if scale_eef is None and args.apply_scale:
            scale_eef = s_vggt
        eef_report = apply_to_eef(
            as_abs(args.eef_in),
            eef_out,
            T_cam_base,
            scale_eef,
            apply_scale=bool(args.apply_scale or args.eef_scale is not None),
            args=args,
        )
        summary["eef_correction"] = eef_report
        (out_dir / "apriltag_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[apriltag] EEF written: {eef_report['out_json']}  n={eef_report['n_corrected']}")

    print("--- AprilTag calibration OK ---")
    print(f"summary: {out_dir / 'apriltag_calibration.json'}")
    print(f"T_tag_in_cam t = {T_tag_cam_avg[:3, 3]}")
    if T_cam_base is not None:
        print(f"T_cam_in_base t = {T_cam_base[:3, 3]}")
    if s_vggt is not None:
        print(f"s_vggt (tag_edge / depth_edge) = {s_vggt:.4f}")
    print(
        "Note: set a real --tag-size-m (printed edge length in meters). "
        "PnP metric accuracy depends on tag size + K quality."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=None, help="HumanEgo session dir containing preprocess/all_data")
    p.add_argument("--images", default=None, help="Image directory (or all_data-like */rgb.png)")
    p.add_argument("--video", default=None)
    p.add_argument("--tag-size-m", type=float, required=True, help="Physical AprilTag black-square edge length [m]")
    p.add_argument("--dictionary", default=DEFAULT_DICT)
    p.add_argument("--tag-id", type=int, default=1, help="Only use this tag id (−1 = all)")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument(
        "--k-fallback",
        choices=["assets", "vfov"],
        default="assets",
        help="When no --fx/--K-json is given and the session has no aria_cam_rgb.json: "
        "'assets' (default) uses the calibrated scene_rgb intrinsics from patches/assets.py "
        "(DEFAULT_ASSETS), scaled to the frame resolution; 'vfov' estimates K from --vfov-deg.",
    )
    p.add_argument("--vfov-deg", type=float, default=70.0, help="Used only when --k-fallback vfov.")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--K-json", default=None, help='JSON with "K" 3x3 and optional "d"')
    p.add_argument(
        "--tag-in-base-json",
        default=None,
        help='Known T_tag_in_base as JSON {"T": [[...],[...],...]}',
    )
    p.add_argument(
        "--tag-origin-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Tag center in robot base [m] (rotation from --tag-rpy-deg)",
    )
    p.add_argument(
        "--tag-rpy-deg",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Tag orientation in base, xyz Euler degrees",
    )
    p.add_argument("--vggt-npz", default=None, help="VGGT predictions.npz for depth-scale estimate")
    p.add_argument(
        "--use-vggt-K-for-depth",
        action="store_true",
        help="Unproject VGGT depth with VGGT intrinsic (mapped to depth res)",
    )
    p.add_argument("--eef-in", default=None, help="EEF trajectory JSON to correct")
    p.add_argument("--eef-out", default=None, help="Output dir for corrected EEF")
    p.add_argument(
        "--apply-scale",
        action="store_true",
        help="When writing EEF, apply median VGGT depth scale to cam-space translations",
    )
    p.add_argument("--eef-scale", type=float, default=None, help="Override scale for EEF rewrite")
    p.add_argument("--save-vis", action="store_true", default=True)
    p.add_argument("--no-save-vis", action="store_false", dest="save_vis")
    p.add_argument("--out", default="outputs/apriltag_calib")
    args = p.parse_args()

    if not any([args.session, args.images, args.video]):
        # default session if present
        default = REPO_ROOT / "outputs" / "nero_pick_place_human"
        if (default / "preprocess" / "all_data").is_dir():
            args.session = str(default.relative_to(REPO_ROOT))
            print(f"[apriltag] default --session {args.session}")
        else:
            p.error("Provide --session, --images, or --video")

    run(args)


if __name__ == "__main__":
    main()
