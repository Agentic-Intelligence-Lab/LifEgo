#!/usr/bin/env python3
"""Smoke-test VGGT-Ω inference on local frames / video.

Loads a local checkpoint under thirdparty/vggt-omega/weights, runs multi-view
depth + camera prediction, and writes a compact NPZ + JSON summary.

This script only exercises the official model API (no EEF correction yet).

Example:

    cd LifEgo
    PY=/home/ymq/miniconda3/envs/lifego/bin/python
    $PY patches/run_vggt_omega_infer.py \
      --images outputs/nero_pick_place_human/preprocess/all_data \
      --max-frames 8 \
      --frame-stride 15 \
      --out outputs/vggt_omega/smoke_nero_pick_place
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "thirdparty" / "vggt-omega"
DEFAULT_CKPT = VGGT_ROOT / "weights" / "VGGT-Omega" / "vggt_omega_1b_512.pt"


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_vggt_path() -> None:
    root = str(VGGT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def collect_images(images_dir: Path | None, video: Path | None, max_frames: int, frame_stride: int) -> list[Path]:
    paths: list[Path] = []
    if images_dir is not None:
        images_dir = as_abs(images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"--images not a directory: {images_dir}")
        # Prefer HumanEgo per-frame layout: .../all_data/00000/rgb.png
        per_frame = sorted(images_dir.glob("*/rgb.png"))
        flat = sorted(
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            and p.name not in {"depth.png"}
        )
        paths = per_frame if per_frame else flat
    if video is not None:
        video = as_abs(video)
        if not video.is_file():
            raise FileNotFoundError(f"--video not found: {video}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video}")
        tmp_dir = as_abs("outputs/vggt_omega/_tmp_frames") / video.stem
        tmp_dir.mkdir(parents=True, exist_ok=True)
        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(frame_stride, 1) == 0:
                out = tmp_dir / f"{saved:06d}.png"
                cv2.imwrite(str(out), frame)
                paths.append(out)
                saved += 1
                if max_frames > 0 and saved >= max_frames:
                    break
            idx += 1
        cap.release()
        # When both images and video: use only the video samples for this branch path list rebuild
        if images_dir is None:
            pass
        else:
            # Prefer video dumps when only --video was intended; if both, concat unique.
            pass

    if images_dir is not None and video is None:
        if frame_stride > 1:
            paths = paths[::frame_stride]
        if max_frames > 0:
            paths = paths[:max_frames]

    if not paths:
        raise RuntimeError("No images found. Pass --images and/or --video.")
    return [Path(p) for p in paths]


def pick_device(prefer: str) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensors_to_numpy(predictions: dict) -> dict:
    out = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().float().cpu().numpy()
            if arr.shape and arr.shape[0] == 1:
                arr = arr[0]
            out[key] = arr
        else:
            out[key] = value
    return out


def _chunk_indices(n: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def run_inference(args: argparse.Namespace) -> None:
    ensure_vggt_path()
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    ckpt = as_abs(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Expected something like thirdparty/vggt-omega/weights/VGGT-Omega/vggt_omega_1b_512.pt"
        )

    image_paths = collect_images(
        as_abs(args.images) if args.images else None,
        as_abs(args.video) if args.video else None,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    n = len(image_paths)
    print(f"[vggt-omega] frames: {n}")
    for p in image_paths[:5]:
        print(f"  - {p}")
    if n > 5:
        print(f"  ... (+{n - 5} more)")

    device = pick_device(args.device)
    print(f"[vggt-omega] device={device}  checkpoint={ckpt.name}")
    print(f"[vggt-omega] resolution={args.image_resolution}  mode={args.preprocess_mode}")
    print(f"[vggt-omega] chunk_size={args.chunk_size} (0 = single forward)")

    enable_alignment = bool(args.enable_alignment)
    model = VGGTOmega(enable_alignment=enable_alignment).to(device).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and all(
        not str(k).startswith("aggregator") for k in state if k != "state_dict"
    ):
        state = state["state_dict"]
    model.load_state_dict(state)
    del state

    chunks = _chunk_indices(n, args.chunk_size)
    merged: dict[str, list[np.ndarray]] = {
        "depth": [],
        "depth_conf": [],
        "pose_enc": [],
        "extrinsic": [],
        "intrinsic": [],
    }
    if args.save_images:
        merged["images"] = []
    if args.save_tokens:
        merged["camera_and_register_tokens"] = []

    preprocess_shape = None
    t_load0 = time.time()
    infer_s = 0.0

    for ci, (a, b) in enumerate(chunks):
        paths_i = image_paths[a:b]
        print(f"[vggt-omega] chunk {ci + 1}/{len(chunks)}  frames [{a}:{b}] ({b - a})")
        images = load_and_preprocess_images(
            [str(p) for p in paths_i],
            mode=args.preprocess_mode,
            image_resolution=args.image_resolution,
        ).to(device)
        preprocess_shape = list(images.shape)
        print(f"  preprocessed: {tuple(images.shape)}")

        t0 = time.time()
        with torch.inference_mode():
            if device.type != "cuda":
                print("[vggt-omega] WARNING: official code expects CUDA; CPU may fail/autocast.")
            predictions = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_s += time.time() - t0

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
        predictions["extrinsic"] = extrinsics
        predictions["intrinsic"] = intrinsics
        pred_np = tensors_to_numpy(predictions)

        for k in merged:
            if k not in pred_np:
                continue
            arr = pred_np[k]
            # After batch squeeze, shape is (S, ...)
            merged[k].append(np.asarray(arr))

        del predictions, images, pred_np, extrinsics, intrinsics
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preload_s = time.time() - t_load0 - infer_s

    # Concatenate along view axis (axis 0)
    pred_np = {k: np.concatenate(v, axis=0) for k, v in merged.items() if v}
    for k, arr in pred_np.items():
        print(f"  merged {k}: {arr.shape}")

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "predictions.npz"

    save_dict = dict(pred_np)
    save_dict["image_paths"] = np.asarray([str(p) for p in image_paths])
    save_dict["chunk_size"] = np.asarray(int(args.chunk_size))
    np.savez_compressed(npz_path, **save_dict)

    depth = pred_np.get("depth")
    depth_conf = pred_np.get("depth_conf")
    depth_stats = None
    conf_stats = None
    if depth is not None:
        d = depth[..., 0] if depth.ndim >= 4 else depth
        depth_stats = {
            "shape": list(d.shape),
            "min": float(np.nanmin(d)),
            "max": float(np.nanmax(d)),
            "mean": float(np.nanmean(d)),
            "median": float(np.nanmedian(d)),
        }
    if depth_conf is not None:
        c = depth_conf[..., 0] if depth_conf.ndim >= 4 else depth_conf
        conf_stats = {
            "shape": list(c.shape),
            "min": float(np.nanmin(c)),
            "max": float(np.nanmax(c)),
            "mean": float(np.nanmean(c)),
        }

    ext = pred_np.get("extrinsic")
    cam_summary = None
    if ext is not None and ext.ndim == 3 and ext.shape[-2:] == (3, 4):
        t = ext[:, :3, 3]
        cam_summary = {
            "translation_mean": t.mean(axis=0).tolist(),
            "translation_std": t.std(axis=0).tolist(),
            "n_views": int(t.shape[0]),
        }

    meta = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "n_frames": n,
        "chunk_size": int(args.chunk_size),
        "n_chunks": len(chunks),
        "image_paths": [str(p) for p in image_paths],
        "source_video": str(as_abs(args.video)) if args.video else None,
        "source_images": str(as_abs(args.images)) if args.images else None,
        "image_resolution": int(args.image_resolution),
        "preprocess_mode": args.preprocess_mode,
        "enable_alignment": enable_alignment,
        "preprocessed_shape_last_chunk": preprocess_shape,
        "infer_seconds": float(infer_s),
        "other_overhead_seconds": float(preload_s),
        "depth": depth_stats,
        "depth_conf": conf_stats,
        "camera_translation": cam_summary,
        "output_npz": str(npz_path),
        "keys_saved": sorted(save_dict.keys()),
        "note": "Chunked multi-view inference for long sequences; each chunk is jointly "
        "processed. Absolute scale still uncalibrated.",
    }
    meta_path = out_dir / "predictions_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if depth is not None:
        d0 = depth[0, ..., 0] if depth.ndim >= 4 else depth[0]
        d_norm = d0 - np.nanmin(d0)
        d_norm = d_norm / max(float(np.nanmax(d_norm)), 1e-6)
        vis = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        vis_path = out_dir / "depth_frame0.png"
        cv2.imwrite(str(vis_path), vis_color)
        # Also save last frame and a mid frame for QA
        for tag, fi in [("mid", n // 2), ("last", n - 1)]:
            di = depth[fi, ..., 0] if depth.ndim >= 4 else depth[fi]
            dn = di - np.nanmin(di)
            dn = dn / max(float(np.nanmax(dn)), 1e-6)
            cv2.imwrite(
                str(out_dir / f"depth_frame_{tag}.png"),
                cv2.applyColorMap((np.clip(dn, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            )
        meta["depth_vis"] = str(vis_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("--- VGGT-Ω inference OK ---")
    print(f"saved: {npz_path}")
    print(f"meta:  {meta_path}")
    if depth_stats:
        print(
            f"depth: shape={depth_stats['shape']}  "
            f"mean={depth_stats['mean']:.3f}  median={depth_stats['median']:.3f}  "
            f"[{depth_stats['min']:.3f}, {depth_stats['max']:.3f}]"
        )
    print(f"infer: {infer_s:.2f}s for {n} frames in {len(chunks)} chunk(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT.relative_to(REPO_ROOT)),
        help="Path to vggt_omega_1b_512.pt (or 256-text)",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Directory of images or HumanEgo all_data/ folder (uses */rgb.png)",
    )
    parser.add_argument("--video", default=None, help="Optional video; frames extracted to outputs/vggt_omega/_tmp_frames")
    parser.add_argument("--max-frames", type=int, default=8, help="Cap number of views (0 = all)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Subsample stride for all_data/rgb or video")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Joint multi-view batch size (0 = all frames at once; laptop 16GB recommend 24–40)",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument(
        "--enable-alignment",
        action="store_true",
        help="Use TextAlignment head (needs vggt_omega_1b_256_text.pt + res 256)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--save-tokens", action="store_true", help="Also dump camera/register tokens (large)")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save model-preprocessed RGB tensors into npz (large for full sequences)",
    )
    parser.add_argument(
        "--out",
        default="outputs/vggt_omega/smoke",
        help="Output directory for predictions.npz + summary",
    )
    args = parser.parse_args()

    if args.images is None and args.video is None:
        # Sensible default: sample frames from pick-place preprocess if present
        default_images = REPO_ROOT / "outputs" / "nero_pick_place_human" / "preprocess" / "all_data"
        default_video = REPO_ROOT / "examples" / "nero_pick_place_human.mp4"
        if default_images.is_dir():
            args.images = str(default_images.relative_to(REPO_ROOT))
            args.frame_stride = max(args.frame_stride, 20)
            print(f"[vggt-omega] defaulting --images to {args.images} stride={args.frame_stride}")
        elif default_video.is_file():
            args.video = str(default_video.relative_to(REPO_ROOT))
            print(f"[vggt-omega] defaulting --video to {args.video}")
        else:
            parser.error("Provide --images or --video (no default session found)")

    if args.enable_alignment and args.image_resolution != 256:
        print("[vggt-omega] NOTE: text-alignment checkpoint usually expects --image-resolution 256")

    run_inference(args)


if __name__ == "__main__":
    main()
