#!/usr/bin/env python3
"""Calibrate a fixed RealSense camera's extrinsics relative to the NERO robot base frame.

Method: PnP against desktop AprilTags whose top_left corner has a known 3D
position in the robot base frame (measured by touching the robot TCP to that
one corner -- see collect_apriltag_corners.py). Each tag's other 3 corners
are NOT measured; the tag is known to be a flat square of fixed side length
lying in a horizontal plane (same z as the measured corner), but its yaw
(rotation about +Z) on the table is unknown. So this script jointly solves
for the camera pose AND every tag's yaw at once, by minimizing reprojection
error against the tags' auto-detected pixel corners (cv2.aruco) -- a coarse
grid search over yaw candidates picks a good starting point (avoiding local
minima), then scipy.optimize.least_squares polishes camera pose + yaws
together. The square's chirality (which rotation direction is "top_left ->
top_right" vs "top_left -> bottom_left") is physically the same for every
tag lying face-up on the table under the same camera, so it is searched once
globally (2 options) rather than per tag.

Input JSON for --tag-corners-base (units: meters; the single point per tag
is top_left as seen by the camera when the tag is viewed right-side up --
matches collect_apriltag_corners.py's output):

    {
      "units": "meters",
      "tag_size_m": 0.05,
      "tags": {
        "1": [x, y, z],
        "2": [x, y, z]
      }
    }

Input JSON for --intrinsics-json (RealSense pyrealsense2.intrinsics-style
fields also accepted: ppx/ppy in place of cx/cy, coeffs in place of
dist_coeffs; a top-level "by_resolution" map with one entry per resolution,
as produced by read_camera_intrinsics.py, is also accepted directly):

    {
      "width": 1280, "height": 720,
      "fx": 905.86, "fy": 905.42, "cx": 638.85, "cy": 361.16,
      "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
    }

fx/fy/cx/cy/dist-coeffs can also be passed directly as CLI flags instead of
(or to override individual fields of) --intrinsics-json.

Output: T_cam_in_base -- the camera pose in the robot base frame -- which is
the requested camera extrinsic, plus its inverse T_base_in_cam, the solved
per-tag yaw angles, reprojection error diagnostics, and annotated debug
images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS

REPO_ROOT = Path(__file__).resolve().parents[1]

PNP_METHOD_ATTRS = {
    "sqpnp": "SOLVEPNP_SQPNP",
    "epnp": "SOLVEPNP_EPNP",
    "iterative": "SOLVEPNP_ITERATIVE",
}


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def matrix_record(T: np.ndarray) -> dict:
    rot = R.from_matrix(T[:3, :3])
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": rot.as_quat().tolist(),
        "rotvec": rot.as_rotvec().tolist(),
    }


# ---------------------------------------------------------------------------
# Intrinsics / tag geometry loading
# ---------------------------------------------------------------------------


def load_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    data: dict = {}
    if args.intrinsics_json:
        data = json.loads(as_abs(args.intrinsics_json).read_text(encoding="utf-8"))
    elif args.fx is None and args.fy is None:
        # No --intrinsics-json and no CLI overrides: fall back to the scene_rgb
        # camera intrinsics already calibrated in patches/assets.py.
        cam_intr = DEFAULT_ASSETS.camera().intrinsics
        if cam_intr.is_filled():
            data = {
                "fx": cam_intr.fx,
                "fy": cam_intr.fy,
                "cx": cam_intr.cx,
                "cy": cam_intr.cy,
                "dist_coeffs": None if cam_intr.dist_coeffs is None else cam_intr.dist_coeffs.tolist(),
                "width": cam_intr.width,
                "height": cam_intr.height,
            }

    by_res = data.get("by_resolution")
    if by_res:
        if args.resolution:
            key = args.resolution
            if key not in by_res:
                raise ValueError(f"resolution '{key}' not in {args.intrinsics_json}'s by_resolution: {list(by_res)}")
        elif len(by_res) == 1:
            key = next(iter(by_res))
        else:
            raise ValueError(
                f"{args.intrinsics_json} has multiple resolutions {list(by_res)}; pick one with --resolution WxH"
            )
        data = by_res[key]

    fx = args.fx if args.fx is not None else data.get("fx")
    fy = args.fy if args.fy is not None else data.get("fy")
    cx = args.cx if args.cx is not None else data.get("cx", data.get("ppx"))
    cy = args.cy if args.cy is not None else data.get("cy", data.get("ppy"))
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(
            "Missing camera intrinsics: need fx, fy, cx (or ppx), cy (or ppy) "
            "via --intrinsics-json and/or --fx/--fy/--cx/--cy."
        )

    dist = args.dist_coeffs
    if dist is None:
        dist = data.get("dist_coeffs", data.get("coeffs"))
    if dist is None:
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_arr = np.array(dist, dtype=np.float64).reshape(-1, 1)
    meta = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "dist_coeffs": dist_arr.flatten().tolist(),
        "width": data.get("width"),
        "height": data.get("height"),
        "source_json": args.intrinsics_json or "patches/assets.py (DEFAULT_ASSETS scene_rgb)",
    }
    return K, dist_arr, meta


def load_tag_top_left_base(path: str, tag_size_override: float | None) -> tuple[dict[int, np.ndarray], float, dict]:
    data = json.loads(as_abs(path).read_text(encoding="utf-8"))
    tags_raw = data.get("tags")
    if not tags_raw:
        raise ValueError(f"{path}: missing top-level 'tags' object")

    tag_size = tag_size_override if tag_size_override is not None else data.get("tag_size_m")
    if tag_size is None:
        raise ValueError(f"{path}: missing 'tag_size_m' (or pass --tag-size-m explicitly)")

    out: dict[int, np.ndarray] = {}
    for key, value in tags_raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"tag '{key}': expected a single [x,y,z] top_left point (got shape {arr.shape})")
        out[int(key)] = arr

    meta = {
        "units": data.get("units", "meters"),
        "corner_captured": data.get("corner_captured", "top_left"),
        "tag_size_m": float(tag_size),
        "tag_ids": sorted(out.keys()),
    }
    return out, float(tag_size), meta


def corners_from_top_left(top_left: np.ndarray, size: float, yaw: float, handedness: float) -> np.ndarray:
    """Build [top_left, top_right, bottom_right, bottom_left] (base frame, planar in z)
    from the measured top_left corner, known side length, an unknown yaw (rotation of
    the "top_left -> top_right" edge about +Z), and a shared chirality sign."""
    right = size * np.array([np.cos(yaw), np.sin(yaw), 0.0])
    down = size * handedness * np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    top_right = top_left + right
    bottom_left = top_left + down
    bottom_right = top_right + down
    return np.stack([top_left, top_right, bottom_right, bottom_left], axis=0)


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------


def get_aruco_dictionary(family: str):
    candidates = [family, family.upper(), family.replace("h", "H"), family.replace("H", "h")]
    for name in candidates:
        if hasattr(cv2.aruco, name):
            return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    available = sorted(n for n in dir(cv2.aruco) if "APRILTAG" in n.upper())
    raise ValueError(f"Unknown tag family '{family}'. Available: {available}")


def make_aruco_detector(dictionary):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def detect_apriltags(gray: np.ndarray, detect_fn) -> dict[int, np.ndarray]:
    corners, ids, _ = detect_fn(gray)
    result: dict[int, np.ndarray] = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            result[int(i)] = c.reshape(4, 2).astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# PnP solve + diagnostics
# ---------------------------------------------------------------------------


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    method: str,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    attr = PNP_METHOD_ATTRS.get(method, PNP_METHOD_ATTRS["epnp"])
    flag = getattr(cv2, attr, None)
    used = method
    if flag is None:
        flag = cv2.SOLVEPNP_EPNP
        used = "epnp (fallback)"
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=flag)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed to converge")
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, dist, rvec, tvec)
    return rvec, tvec, used


def solve_joint_pose_and_yaws(
    top_lefts: dict[int, np.ndarray],
    size: float,
    image_points_by_tag: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    pnp_method: str,
    yaw_grid_step_deg: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], float]:
    """Jointly solve camera pose (rvec, tvec) and each tag's yaw about +Z, given
    only each tag's top_left corner (base frame) and its fixed side length.

    Strategy: coarse grid search over yaw candidates x {+1,-1} chirality to find
    a good starting point (avoids the local minima that a single continuous
    optimizer could get stuck in), then scipy.optimize.least_squares polishes
    camera pose + yaws jointly (chirality is discrete and stays fixed after the
    grid search picks it).

    All tags lie in the same horizontal plane (same z), so the point set as a
    whole is exactly planar -- this makes the general 6-DOF pose ambiguous:
    reflecting the camera through the tags' plane (and correspondingly
    flipping yaw/chirality) reproduces an almost identical image with an
    equally low reprojection error, just with the camera on the wrong side of
    the table. We break that tie with the physical fact that this camera
    looks down at the table from above: candidates whose recovered camera
    z (base frame) is not above the tags' plane are rejected outright.
    """
    import itertools

    from scipy.optimize import least_squares

    tag_ids = sorted(top_lefts.keys())
    image_points = np.concatenate([image_points_by_tag[t] for t in tag_ids], axis=0)
    tag_plane_z = float(np.mean([top_lefts[t][2] for t in tag_ids]))

    def build_object_points(yaws: list[float] | np.ndarray, handedness: float) -> np.ndarray:
        return np.concatenate(
            [corners_from_top_left(top_lefts[t], size, y, handedness) for t, y in zip(tag_ids, yaws)], axis=0
        )

    def camera_position_base(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R_mat, _ = cv2.Rodrigues(rvec)
        return (-R_mat.T @ tvec.reshape(3, 1)).flatten()

    yaw_grid = np.deg2rad(np.arange(0.0, 360.0, yaw_grid_step_deg))
    best = None  # (rmse, yaws, handedness, rvec, tvec)
    for handedness in (1.0, -1.0):
        for yaws in itertools.product(yaw_grid, repeat=len(tag_ids)):
            obj = build_object_points(yaws, handedness)
            try:
                rvec, tvec, _ = solve_pnp(obj, image_points, K, dist, pnp_method, refine=False)
            except RuntimeError:
                continue
            if camera_position_base(rvec, tvec)[2] <= tag_plane_z:
                continue  # camera below/at the table plane: the reflected-ambiguity twin, reject
            _, _, rmse = reprojection_error(obj, image_points, K, dist, rvec, tvec)
            if best is None or rmse < best[0]:
                best = (rmse, yaws, handedness, rvec, tvec)

    if best is None:
        raise RuntimeError(
            "joint pose+yaw grid search found no solution with the camera above the tags' plane "
            f"(z > {tag_plane_z:.4f} m) -- check --tag-corners-base and the detected tag ids."
        )

    _, yaws0, handedness, rvec0, tvec0 = best

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = params[0:3]
        tvec = params[3:6]
        yaws = params[6:]
        obj = build_object_points(yaws, handedness)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        return (proj.reshape(-1, 2) - image_points).ravel()

    x0 = np.concatenate([rvec0.flatten(), tvec0.flatten(), np.array(yaws0, dtype=np.float64)])
    result = least_squares(residuals, x0, method="lm")

    rvec = result.x[0:3].reshape(3, 1)
    tvec = result.x[3:6].reshape(3, 1)
    if camera_position_base(rvec, tvec)[2] <= tag_plane_z:
        raise RuntimeError(
            "local refinement drifted to the camera-below-the-table twin solution; "
            "try a finer --yaw-grid-step-deg"
        )
    yaws_final = result.x[6:]
    yaw_by_tag = {t: float(np.rad2deg(y) % 360.0) for t, y in zip(tag_ids, yaws_final)}
    return rvec, tvec, yaw_by_tag, handedness


def reprojection_error(
    object_points: np.ndarray, image_points: np.ndarray, K: np.ndarray, dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    per_point = np.linalg.norm(proj - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(per_point**2)))
    return proj, per_point, rmse


def draw_debug_image(img: np.ndarray, detections: dict[int, np.ndarray], proj_by_tag: dict[int, np.ndarray]) -> np.ndarray:
    vis = img.copy()
    for tag_id, corners in detections.items():
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vis, f"id{tag_id} detected", (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for tag_id, proj in proj_by_tag.items():
        pts = proj.astype(int)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 1)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.putText(vis, "green=detected  red=reprojected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def calibrate(args: argparse.Namespace) -> None:
    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if not args.no_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    K, dist, intrinsics_meta = load_intrinsics(args)
    top_lefts, tag_size, tags_meta = load_tag_top_left_base(args.tag_corners_base, args.tag_size_m)
    dictionary = get_aruco_dictionary(args.tag_family)
    detect_fn = make_aruco_detector(dictionary)

    image_paths = [as_abs(p) for p in args.images]
    per_tag_pixels: dict[int, list[tuple[str, np.ndarray]]] = {}
    per_image_data: list[dict] = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detect_apriltags(gray, detect_fn)
        for tag_id, corners in detections.items():
            if tag_id in top_lefts:
                per_tag_pixels.setdefault(tag_id, []).append((str(path), corners))
        per_image_data.append({"path": path, "img": img, "detections": detections})

    tags_used = sorted(per_tag_pixels.keys())
    tags_missing = sorted(set(top_lefts.keys()) - set(tags_used))
    if not tags_used:
        raise RuntimeError(f"No known tags (ids {sorted(top_lefts.keys())}) were detected in any input image.")
    if len(tags_used) < 2:
        print(
            f"[warn] only tag(s) {tags_used} detected; using a single planar tag (4 points) is "
            "more prone to pose ambiguity than using both tags (8 points)."
        )

    image_points_by_tag: dict[int, np.ndarray] = {}
    per_tag_pixel_std: dict[int, list[float]] = {}
    for tag_id in tags_used:
        entries = per_tag_pixels[tag_id]
        stacked = np.stack([c for _, c in entries], axis=0)  # (n, 4, 2)
        image_points_by_tag[tag_id] = stacked.mean(axis=0)
        per_tag_pixel_std[tag_id] = np.linalg.norm(stacked.std(axis=0), axis=1).tolist()

    used_top_lefts = {t: top_lefts[t] for t in tags_used}
    rvec, tvec, yaw_by_tag, handedness = solve_joint_pose_and_yaws(
        used_top_lefts, tag_size, image_points_by_tag, K, dist, args.pnp_method, args.yaw_grid_step_deg
    )
    method_used = args.pnp_method

    object_points = np.concatenate(
        [corners_from_top_left(used_top_lefts[t], tag_size, np.deg2rad(yaw_by_tag[t]), handedness) for t in tags_used],
        axis=0,
    )
    image_points = np.concatenate([image_points_by_tag[t] for t in tags_used], axis=0)
    _, per_point_err, rmse_px = reprojection_error(object_points, image_points, K, dist, rvec, tvec)

    R_mat, _ = cv2.Rodrigues(rvec)
    T_base_in_cam = np.eye(4, dtype=np.float64)
    T_base_in_cam[:3, :3] = R_mat
    T_base_in_cam[:3, 3] = tvec.flatten()
    T_cam_in_base = np.linalg.inv(T_base_in_cam)

    # Per-image diagnostics: reproject the pooled solution into each raw frame,
    # and (when a frame alone has >=4 points) solve PnP standalone to check
    # consistency against the pooled result (large deltas usually mean the
    # camera moved between shots or a frame's detection is bad).
    r_global = R.from_matrix(R_mat)
    per_image_diag = []
    for entry in per_image_data:
        path = entry["path"]
        detections = entry["detections"]
        known = {tid: c for tid, c in detections.items() if tid in used_top_lefts}
        if not known:
            per_image_diag.append({"image": str(path), "tags_detected": [], "reprojection_rmse_px": None})
            continue

        obj_here = np.concatenate(
            [
                corners_from_top_left(used_top_lefts[tid], tag_size, np.deg2rad(yaw_by_tag[tid]), handedness)
                for tid in sorted(known)
            ],
            axis=0,
        )
        img_here = np.concatenate([known[tid] for tid in sorted(known)], axis=0)
        proj_here, _, rmse_here = reprojection_error(obj_here, img_here, K, dist, rvec, tvec)

        diag = {
            "image": str(path),
            "tags_detected": sorted(known.keys()),
            "reprojection_rmse_px": rmse_here,
            "standalone_rotation_delta_deg": None,
            "standalone_translation_delta_mm": None,
        }
        if len(known) * 4 >= 4 and img_here.shape[0] >= 4:
            try:
                rvec_i, tvec_i, _ = solve_pnp(obj_here, img_here, K, dist, args.pnp_method)
                r_i = R.from_matrix(cv2.Rodrigues(rvec_i)[0])
                rot_delta_deg = float(np.degrees((r_global.inv() * r_i).magnitude()))
                trans_delta_mm = float(np.linalg.norm(tvec_i.flatten() - tvec.flatten()) * 1000.0)
                diag["standalone_rotation_delta_deg"] = rot_delta_deg
                diag["standalone_translation_delta_mm"] = trans_delta_mm
            except RuntimeError:
                pass
        per_image_diag.append(diag)

        if not args.no_debug_images:
            proj_by_tag = {}
            offset = 0
            for tid in sorted(known):
                proj_by_tag[tid] = proj_here[offset : offset + 4]
                offset += 4
            vis = draw_debug_image(entry["img"], detections, proj_by_tag)
            out_name = f"{path.stem}_reprojection.png"
            cv2.imwrite(str(debug_dir / out_name), vis)

    metadata = {
        "images": [str(p) for p in image_paths],
        "tag_family": args.tag_family,
        "tag_corners_base_source": args.tag_corners_base,
        "tags_meta": tags_meta,
        "tag_size_m": tag_size,
        "solved_yaw_deg_by_tag": yaw_by_tag,
        "solved_chirality": handedness,
        "intrinsics": intrinsics_meta,
        "tags_used": tags_used,
        "tags_missing": tags_missing,
        "num_correspondences": int(object_points.shape[0]),
        "pnp_method_requested": args.pnp_method,
        "pnp_method_used": method_used,
        "reprojection_rmse_px": rmse_px,
        "per_point_reprojection_error_px": per_point_err.tolist(),
        "per_tag_pixel_detection_std_px": per_tag_pixel_std,
        "per_image_diagnostics": per_image_diag,
    }

    result = {
        "metadata": metadata,
        "T_cam_in_base": matrix_record(T_cam_in_base),
        "T_base_in_cam": matrix_record(T_base_in_cam),
        "camera_position_in_base_m": T_cam_in_base[:3, 3].tolist(),
    }

    out_path = out_dir / "camera_extrinsics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    if not args.no_debug_images:
        print(f"Debug images in {debug_dir}")
    print(f"Tags used: {tags_used}  (missing/not detected: {tags_missing})")
    print(f"Solved yaw per tag (deg): {yaw_by_tag}  chirality: {handedness:+.0f}")
    print(f"PnP method: {method_used}  correspondences: {object_points.shape[0]}")
    print(f"Reprojection RMSE: {rmse_px:.3f} px")
    print(f"Camera position in base frame (m): {T_cam_in_base[:3, 3].tolist()}")
    for diag in per_image_diag:
        if diag["standalone_rotation_delta_deg"] is not None:
            print(
                f"  {Path(diag['image']).name}: rmse={diag['reprojection_rmse_px']:.3f}px "
                f"standalone_delta=({diag['standalone_rotation_delta_deg']:.3f}deg, "
                f"{diag['standalone_translation_delta_mm']:.2f}mm)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate RealSense camera extrinsics (in robot base frame) from two AprilTags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", nargs="+", required=True, help="One or more RGB frames from the static camera.")
    parser.add_argument(
        "--tag-corners-base",
        required=True,
        help="JSON with each tag's top_left corner position in the robot base frame (meters) "
        "and tag_size_m -- see collect_apriltag_corners.py.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=None,
        help="Override the tag side length (meters) instead of reading tag_size_m from --tag-corners-base.",
    )
    parser.add_argument(
        "--yaw-grid-step-deg",
        type=float,
        default=15.0,
        help="Coarse grid resolution (degrees) for the initial per-tag yaw search before local refinement.",
    )
    parser.add_argument("--tag-family", default="DICT_APRILTAG_36h11")
    parser.add_argument("--intrinsics-json", default=None, help="JSON with fx/fy/cx(ppx)/cy(ppy)/dist_coeffs(coeffs).")
    parser.add_argument(
        "--resolution",
        default=None,
        help="Resolution key (e.g. '640x480') to select when --intrinsics-json has a top-level "
        "'by_resolution' map with multiple entries. Not needed if it has only one.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--dist-coeffs", nargs="*", type=float, default=None, help="OpenCV order: k1 k2 p1 p2 k3 ...")
    parser.add_argument("--pnp-method", choices=list(PNP_METHOD_ATTRS.keys()), default="sqpnp")
    parser.add_argument("--out", default="outputs/camera_extrinsics")
    parser.add_argument("--no-debug-images", action="store_true")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
