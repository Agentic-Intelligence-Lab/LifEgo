#!/usr/bin/env python3
"""Build one LeRobot EEF dataset from reconstructed ego data and real Nero teleop data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO_EEF_REL = Path("robot_eef_scene_camera_axis_corrected_flat_x0") / "robot_eef_trajectory.json"
DEFAULT_EGO_FRAMES_REL = Path("preprocess") / "all_data"
DEFAULT_REAL_ROOT = REPO_ROOT / "DATA" / "20260816_nero_stack_object_horizontal"
DEFAULT_REPO_ID = "local/nero_ego_real_stack_object_horizontal_eef"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lerobot"
DEFAULT_TASK = "Place the black pillar in the plate."


@dataclass(frozen=True)
class EpisodeFrame:
    image_path: Path | None
    video_path: Path | None
    video_frame_index: int | None
    state: np.ndarray
    source_index: int


@dataclass(frozen=True)
class Episode:
    name: str
    source: str
    frames: list[EpisodeFrame]


def as_abs(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def import_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not available. Run from the OpenPI environment, e.g. "
            "`cd thirdparty/openpi && uv run python ../../ego2exe/build_lerobot_eef_cotrain_dataset.py ...`."
        ) from exc
    return LeRobotDataset, HF_LEROBOT_HOME


def continuous_quat(prev: np.ndarray | None, quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"expected quat shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("invalid zero quaternion")
    quat = quat / norm
    if prev is not None and float(np.dot(prev, quat)) < 0.0:
        quat = -quat
    return quat


def rpy_to_quat_xyzw(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return continuous_quat(
        None,
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
    )


def eef_state(pos: Sequence[float], quat: Sequence[float], grasp: float, prev_quat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pos_arr = np.asarray(pos, dtype=np.float64)
    quat_arr = continuous_quat(prev_quat, quat)
    state = np.concatenate([pos_arr, quat_arr, np.asarray([float(grasp)], dtype=np.float64)]).astype(np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid EEF state: {state}")
    return state, quat_arr


def discover_ego_sessions(paths: Sequence[str], eef_rel: Path) -> list[Path]:
    sessions: list[Path] = []
    for raw in paths:
        path = as_abs(raw)
        if (path / eef_rel).is_file():
            sessions.append(path)
        elif path.is_dir():
            sessions.extend(candidate for candidate in sorted(path.iterdir()) if candidate.is_dir() and (candidate / eef_rel).is_file())
        else:
            raise FileNotFoundError(f"ego path does not exist: {path}")
    return sessions


def load_ego_episode(session: Path, *, eef_rel: Path, frames_rel: Path, min_valid_frames: int) -> Episode:
    eef_path = session / eef_rel
    frames_dir = session / frames_rel
    if not eef_path.is_file():
        raise FileNotFoundError(eef_path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)

    payload = json.loads(eef_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{eef_path} must contain a list field named records")

    frames: list[EpisodeFrame] = []
    prev_quat: np.ndarray | None = None
    for record in records:
        if not record.get("valid", False):
            continue
        idx = int(record["idx"])
        image_path = frames_dir / f"{idx:05d}" / "rgb.png"
        if not image_path.is_file():
            image_path = frames_dir / f"{idx}" / "rgb.png"
        if not image_path.is_file():
            continue
        eef = record.get("T_ee_in_base")
        if not eef:
            continue
        state, prev_quat = eef_state(eef["translation_m"], eef["quat_xyzw"], float(record.get("grasp", 0.0)), prev_quat)
        frames.append(EpisodeFrame(image_path=image_path, video_path=None, video_frame_index=None, state=state, source_index=idx))

    if len(frames) < min_valid_frames:
        raise ValueError(f"{session} has only {len(frames)} usable ego frames")
    return Episode(name=session.name, source="ego", frames=frames)


def load_real_jsonl(path: Path, *, require_valid: bool, min_valid_frames: int) -> Episode:
    metadata: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "metadata" and metadata is None:
                metadata = record
            elif record.get("kind") == "sample":
                samples.append(record)
    if metadata is None:
        raise ValueError(f"missing metadata row: {path}")

    video_path = path.with_suffix(".rgb.mp4")
    if not video_path.is_file():
        raw_video = (metadata.get("camera_rgb") or {}).get("video_path")
        if raw_video:
            video_path = as_abs(raw_video)
    if not video_path.is_file():
        raise FileNotFoundError(f"missing RGB video for {path}")

    frames: list[EpisodeFrame] = []
    prev_quat: np.ndarray | None = None
    for sample in samples:
        if require_valid and not (sample.get("alignment") or {}).get("valid", False):
            continue
        cam = sample.get("camera_rgb") or {}
        video_idx = cam.get("video_frame_index")
        if video_idx is None:
            continue
        poses = sample.get("poses") or {}
        tcp = poses.get("tcp_pose") or ((sample.get("training") or {}).get("state") or {}).get("tcp_pose")
        if tcp is None or len(tcp) < 6:
            continue
        gripper = sample.get("gripper") or {}
        grasp = gripper.get("state_grasp")
        if grasp is None:
            grasp = ((sample.get("training") or {}).get("state") or {}).get("gripper_grasp", 0.0)
        quat = rpy_to_quat_xyzw(tcp[3:6])
        state, prev_quat = eef_state(tcp[:3], quat, float(grasp), prev_quat)
        frames.append(
            EpisodeFrame(
                image_path=None,
                video_path=video_path,
                video_frame_index=int(video_idx),
                state=state,
                source_index=int(sample.get("seq", len(frames))),
            )
        )

    if len(frames) < min_valid_frames:
        raise ValueError(f"{path} has only {len(frames)} usable realbot frames")
    return Episode(name=path.stem, source="realbot", frames=frames)


def load_rgb_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, resample=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


class VideoFrameReader:
    def __init__(self, image_size: tuple[int, int]):
        import cv2

        self.cv2 = cv2
        self.image_size = image_size
        self.path: Path | None = None
        self.cap = None

    def read(self, path: Path, frame_index: int) -> np.ndarray:
        if self.path != path:
            self.close()
            self.path = path
            self.cap = self.cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"failed to open video: {path}")
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"failed to read frame {frame_index} from {path}")
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.image_size[0] or frame_rgb.shape[0] != self.image_size[1]:
            frame_rgb = self.cv2.resize(frame_rgb, self.image_size, interpolation=self.cv2.INTER_AREA)
        return np.asarray(frame_rgb, dtype=np.uint8)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.path = None


def load_frame_image(frame: EpisodeFrame, image_size: tuple[int, int], video_reader: VideoFrameReader) -> np.ndarray:
    if frame.image_path is not None:
        return load_rgb_image(frame.image_path, image_size)
    if frame.video_path is None or frame.video_frame_index is None:
        raise ValueError("frame has neither image path nor video index")
    return video_reader.read(frame.video_path, frame.video_frame_index)


def output_dataset_path(repo_id: str, hf_lerobot_home: Path, output_dir: Path | None) -> Path:
    parent = hf_lerobot_home if output_dir is None else output_dir
    return parent / repo_id


def build_dataset(args: argparse.Namespace) -> None:
    LeRobotDataset, HF_LEROBOT_HOME = import_lerobot()
    image_size = (int(args.image_width), int(args.image_height))
    output_dir = None if args.output_dir is None else as_abs(args.output_dir)
    dataset_path = output_dataset_path(args.repo_id, HF_LEROBOT_HOME, output_dir)
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset already exists: {dataset_path}; pass --overwrite to replace it")
        shutil.rmtree(dataset_path)

    episodes: list[Episode] = []
    failed: list[tuple[Path, str]] = []

    for session in discover_ego_sessions(args.ego_roots, Path(args.ego_eef_rel)):
        try:
            episodes.append(
                load_ego_episode(
                    session,
                    eef_rel=Path(args.ego_eef_rel),
                    frames_rel=Path(args.ego_frames_rel),
                    min_valid_frames=int(args.min_valid_frames),
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((session, str(exc)))
            if not args.skip_bad:
                raise
            print(f"[skip ego] {session}: {exc}")

    real_root = as_abs(args.real_root)
    for jsonl in sorted(real_root.glob(args.real_glob)):
        try:
            episodes.append(
                load_real_jsonl(
                    jsonl,
                    require_valid=not args.keep_invalid_real,
                    min_valid_frames=int(args.min_valid_frames),
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((jsonl, str(exc)))
            if not args.skip_bad:
                raise
            print(f"[skip realbot] {jsonl}: {exc}")

    if not episodes:
        raise ValueError("no episodes loaded")

    print(f"Creating LeRobot co-train dataset: {args.repo_id}")
    print(f"Output path: {dataset_path}")
    print(f"Episodes: {len(episodes)}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_path,
        robot_type="nero_eef",
        fps=int(args.fps),
        features={
            "image": {
                "dtype": "image",
                "shape": (image_size[1], image_size[0], 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["eef_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["eef_action"],
            },
        },
        image_writer_threads=int(args.image_writer_threads),
        image_writer_processes=int(args.image_writer_processes),
    )

    total_frames = 0
    source_counts: dict[str, int] = {}
    video_reader = VideoFrameReader(image_size)
    try:
        for episode in episodes:
            for i, frame in enumerate(episode.frames):
                action = episode.frames[i + 1].state if i + 1 < len(episode.frames) else frame.state
                dataset.add_frame(
                    {
                        "image": load_frame_image(frame, image_size, video_reader),
                        "state": frame.state,
                        "actions": action.astype(np.float32),
                        "task": args.task,
                    }
                )
            try:
                dataset.save_episode(task=args.task)
            except TypeError:
                dataset.save_episode()
            total_frames += len(episode.frames)
            source_counts[episode.source] = source_counts.get(episode.source, 0) + 1
            print(f"[episode] {episode.source:7s} {episode.name}: {len(episode.frames)} frames")
    finally:
        video_reader.close()

    print("Done.")
    print(f"Episodes: {len(episodes)} {source_counts}")
    print(f"Frames: {total_frames}")
    if failed:
        print(f"Skipped episodes: {len(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ego-roots",
        nargs="+",
        default=[
            str(REPO_ROOT / "outputs" / "20260816_ego_ymq"),
            str(REPO_ROOT / "outputs" / "20260816_ego_xule"),
            str(REPO_ROOT / "outputs" / "20260816_ego_hyj"),
        ],
        help="Ego output roots or session directories.",
    )
    parser.add_argument("--real-root", default=str(DEFAULT_REAL_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ego-eef-rel", default=str(DEFAULT_EGO_EEF_REL))
    parser.add_argument("--ego-frames-rel", default=str(DEFAULT_EGO_FRAMES_REL))
    parser.add_argument("--real-glob", default="*.jsonl")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--min-valid-frames", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--keep-invalid-real", action="store_true", help="Keep realbot samples even when alignment.valid is false.")
    parser.add_argument("--skip-bad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise
