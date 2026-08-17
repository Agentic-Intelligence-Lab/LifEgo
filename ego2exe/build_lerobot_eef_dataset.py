#!/usr/bin/env python3
"""Build a LeRobot dataset from ego RGB frames and exported EEF trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EEF_REL = Path("robot_eef_scene_camera_axis_corrected") / "robot_eef_trajectory.json"
DEFAULT_FRAMES_REL = Path("preprocess") / "all_data"
DEFAULT_REPO_ID = "ymq/nero_ego_eef"
DEFAULT_TASK = "follow the demonstrated end-effector motion"


@dataclass(frozen=True)
class EefFrame:
    idx: int
    image_path: Path
    state: np.ndarray


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def import_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not available in this Python environment. Run this script "
            "from the OpenPI/LeRobot environment, for example with "
            "`cd thirdparty/openpi && uv run ../../ego2exe/build_lerobot_eef_dataset.py ...`."
        ) from exc
    return LeRobotDataset, HF_LEROBOT_HOME


def infer_sessions(paths: Sequence[str], eef_rel: Path) -> list[Path]:
    sessions: list[Path] = []
    for raw in paths:
        path = as_abs(raw)
        if (path / eef_rel).is_file():
            sessions.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.iterdir()):
                if candidate.is_dir() and (candidate / eef_rel).is_file():
                    sessions.append(candidate)
            continue
        raise FileNotFoundError(f"cannot find EEF trajectory for session path: {path}")
    if not sessions:
        raise ValueError("no sessions found")
    return sessions


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


def eef_state_from_record(record: dict, prev_quat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    eef = record.get("T_ee_in_base")
    if eef is None:
        raise ValueError("valid EEF record has no T_ee_in_base")
    pos = np.asarray(eef["translation_m"], dtype=np.float64)
    quat = continuous_quat(prev_quat, eef["quat_xyzw"])
    grasp = float(record.get("grasp", 0.0))
    state = np.concatenate([pos, quat, np.asarray([grasp], dtype=np.float64)]).astype(np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid EEF state for frame {record.get('idx')}: {state}")
    return state, quat


def load_episode_frames(session: Path, eef_rel: Path, frames_rel: Path, min_valid_frames: int) -> list[EefFrame]:
    eef_path = session / eef_rel
    frames_dir = session / frames_rel
    if not eef_path.is_file():
        raise FileNotFoundError(f"missing EEF trajectory: {eef_path}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"missing RGB frame directory: {frames_dir}")

    payload = json.loads(eef_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{eef_path} must contain a list field named records")

    frames: list[EefFrame] = []
    prev_quat: np.ndarray | None = None
    missing_images = 0
    for record in records:
        if not record.get("valid", False):
            continue
        idx = int(record["idx"])
        image_path = frames_dir / f"{idx:05d}" / "rgb.png"
        if not image_path.is_file():
            image_path = frames_dir / f"{idx}" / "rgb.png"
        if not image_path.is_file():
            missing_images += 1
            continue
        state, prev_quat = eef_state_from_record(record, prev_quat)
        frames.append(EefFrame(idx=idx, image_path=image_path, state=state))

    if len(frames) < min_valid_frames:
        raise ValueError(
            f"{session} has only {len(frames)} usable EEF/RGB frames "
            f"({missing_images} valid EEF frames had no RGB image)"
        )
    return frames


def load_rgb(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, resample=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def output_dataset_path(repo_id: str, hf_lerobot_home: Path, output_dir: Path | None) -> Path:
    parent = hf_lerobot_home if output_dir is None else output_dir
    return parent / repo_id


def create_dataset(args: argparse.Namespace) -> None:
    LeRobotDataset, HF_LEROBOT_HOME = import_lerobot()

    sessions = infer_sessions(args.sessions, Path(args.eef_rel))
    image_size = (int(args.image_width), int(args.image_height))
    output_dir = None if args.output_dir is None else as_abs(args.output_dir)
    dataset_path = output_dataset_path(args.repo_id, HF_LEROBOT_HOME, output_dir)
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset already exists: {dataset_path}; pass --overwrite to replace it")
        shutil.rmtree(dataset_path)

    print(f"Creating LeRobot dataset: {args.repo_id}")
    print(f"Output path: {dataset_path}")
    print(f"Sessions: {len(sessions)}")

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
    total_episodes = 0
    failed: list[tuple[Path, str]] = []
    for session in sessions:
        try:
            frames = load_episode_frames(
                session,
                Path(args.eef_rel),
                Path(args.frames_rel),
                min_valid_frames=int(args.min_valid_frames),
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((session, str(exc)))
            if not args.skip_bad_sessions:
                raise
            print(f"[skip] {session}: {exc}")
            continue

        task = args.task or DEFAULT_TASK
        for i, frame in enumerate(frames):
            action = frames[i + 1].state if i + 1 < len(frames) else frame.state
            dataset.add_frame(
                {
                    "image": load_rgb(frame.image_path, image_size),
                    "state": frame.state,
                    "actions": action.astype(np.float32),
                    "task": task,
                }
            )
        try:
            dataset.save_episode(task=task)
        except TypeError:
            dataset.save_episode()
        total_episodes += 1
        total_frames += len(frames)
        print(f"[episode] {session.name}: {len(frames)} frames")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["lifego", "nero", "ego", "eef", "lerobot"],
            private=bool(args.private),
            push_videos=True,
            license="apache-2.0",
        )

    print("Done.")
    print(f"Episodes: {total_episodes}")
    print(f"Frames: {total_frames}")
    if failed:
        print(f"Skipped sessions: {len(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sessions",
        nargs="+",
        help="Session directories, or parent directories containing sessions.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional parent directory for LeRobot datasets. Default uses HF_LEROBOT_HOME.",
    )
    parser.add_argument("--eef-rel", default=str(DEFAULT_EEF_REL))
    parser.add_argument("--frames-rel", default=str(DEFAULT_FRAMES_REL))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--min-valid-frames", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bad-sessions", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    create_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise
