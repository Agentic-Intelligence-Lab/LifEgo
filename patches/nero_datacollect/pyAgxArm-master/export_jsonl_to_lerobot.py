#!/usr/bin/env python3
"""Export Nero teleop JSONL(+MP4) recordings to a LeRobot-ready dataset.

Raw recordings remain JSONL + RGB MP4 (not per-frame image folders).
This script selects which vectors become ``observation.state`` / ``action``
for OpenPI pi0.5 / LeRobot LoRA training.

Example:
  python export_jsonl_to_lerobot.py \\
    --input-dir recordings \\
    --output-dir exports/lerobot_tcp \\
    --state-mode tcp+gripper \\
    --action-mode tcp+gripper

State/action modes:
  joints+gripper  -> 8D  [q1..q7, grasp]
  tcp+gripper     -> 7D  [x,y,z,roll,pitch,yaw, grasp]
  flange+gripper  -> 7D
  fk+gripper      -> 7D
  all             -> keeps rich columns; also writes default observation.state=tcp+gripper
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


STATE_MODES = (
    "joints+gripper",
    "tcp+gripper",
    "flange+gripper",
    "fk+gripper",
    "all",
)


@dataclass
class EpisodeBundle:
    jsonl: Path
    video: Optional[Path]
    meta: dict[str, Any]
    samples: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Directory of *.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--glob", default="*.jsonl", help="Episode jsonl glob")
    p.add_argument(
        "--state-mode",
        choices=STATE_MODES,
        default="tcp+gripper",
        help="What becomes observation.state for training",
    )
    p.add_argument(
        "--action-mode",
        choices=STATE_MODES,
        default="tcp+gripper",
        help="What becomes action (default: same family as state; uses next-state target)",
    )
    p.add_argument(
        "--action-policy",
        choices=["next_state", "action_target"],
        default="next_state",
        help="next_state: action[t]=state[t+1]; action_target: use recorded command vector",
    )
    p.add_argument("--fps", type=float, default=None, help="Override fps (else metadata sample_hz)")
    p.add_argument("--repo-id", default="local/nero_teleop")
    p.add_argument(
        "--prefer-lerobot-api",
        action="store_true",
        help="If lerobot is installed, create a real LeRobotDataset (recommended)",
    )
    return p.parse_args()


def load_episode(path: Path) -> EpisodeBundle:
    meta: Optional[dict[str, Any]] = None
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "metadata" and meta is None:
                meta = obj
            elif obj.get("kind") == "sample":
                samples.append(obj)
    if meta is None:
        raise RuntimeError(f"No metadata line in {path}")
    video = path.with_suffix(".rgb.mp4")
    if not video.exists():
        cam = (meta.get("camera_rgb") or {}).get("video_path")
        video = Path(cam) if cam else None
        if video is not None and not video.exists():
            video = None
    return EpisodeBundle(jsonl=path, video=video, meta=meta, samples=samples)


def discover_episodes(input_dir: Path, pattern: str) -> list[EpisodeBundle]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No jsonl matched {input_dir / pattern}")
    episodes = [load_episode(p) for p in paths]
    episodes = [e for e in episodes if e.samples]
    if not episodes:
        raise RuntimeError("No samples found in matched jsonl files")
    return episodes


def _pose(sample: dict[str, Any], key: str) -> Optional[list[float]]:
    poses = sample.get("poses") or {}
    training = ((sample.get("training") or {}).get("state") or {})
    value = poses.get(key) or training.get(key)
    if value is None:
        return None
    out = [float(v) for v in value]
    if len(out) != 6 or any(math.isnan(v) for v in out):
        return None
    return out


def _grasp(sample: dict[str, Any], which: str = "state") -> Optional[float]:
    gripper = sample.get("gripper") or {}
    key = "state_grasp" if which == "state" else "action_grasp"
    if gripper.get(key) is not None:
        return float(gripper[key])
    training = (sample.get("training") or {}).get(
        "state" if which == "state" else "action_target"
    ) or {}
    for name in ("gripper_grasp", "hand_grasp"):
        if training.get(name) is not None:
            return float(training[name])
    return None


def _joints(sample: dict[str, Any], which: str = "state") -> Optional[list[float]]:
    if which == "state":
        follower = sample.get("follower")
        if follower and follower.get("position_rad") is not None:
            return [float(v) for v in follower["position_rad"]]
        training = ((sample.get("training") or {}).get("state") or {})
        if training.get("arm_joint_position_rad") is not None:
            return [float(v) for v in training["arm_joint_position_rad"]]
    else:
        leader = sample.get("leader")
        if leader and leader.get("position_rad") is not None:
            return [float(v) for v in leader["position_rad"]]
        training = ((sample.get("training") or {}).get("action_target") or {})
        if training.get("arm_joint_position_rad") is not None:
            return [float(v) for v in training["arm_joint_position_rad"]]
    return None


def vector_for_mode(sample: dict[str, Any], mode: str, which: str) -> Optional[np.ndarray]:
    grasp = _grasp(sample, "state" if which == "state" else "action")
    if mode == "joints+gripper":
        joints = _joints(sample, "state" if which == "state" else "action")
        if joints is None or grasp is None:
            return None
        return np.asarray(joints + [grasp], dtype=np.float32)
    pose_key = {
        "tcp+gripper": "tcp_pose",
        "flange+gripper": "flange_pose",
        "fk+gripper": "fk_pose",
    }.get(mode)
    if pose_key is None and mode != "all":
        raise ValueError(mode)
    # For action with pose modes, default to current pose (+grasp command);
    # next_state policy replaces this later.
    pose = _pose(sample, pose_key or "tcp_pose")
    if pose is None or grasp is None:
        return None
    return np.asarray(pose + [grasp], dtype=np.float32)


def build_episode_arrays(
    episode: EpisodeBundle,
    state_mode: str,
    action_mode: str,
    action_policy: str,
) -> dict[str, Any]:
    states = []
    actions = []
    timestamps = []
    kept = []
    for sample in episode.samples:
        state = vector_for_mode(
            sample,
            "tcp+gripper" if state_mode == "all" else state_mode,
            "state",
        )
        if state is None:
            continue
        if action_policy == "action_target":
            action = vector_for_mode(
                sample,
                "tcp+gripper" if action_mode == "all" else action_mode,
                "action",
            )
        else:
            action = None  # fill with next state below
        if action_policy == "action_target" and action is None:
            continue
        states.append(state)
        actions.append(action)
        timestamps.append(float(sample.get("elapsed_s", 0.0)))
        kept.append(sample)

    if not states:
        raise RuntimeError(f"No valid frames in {episode.jsonl}")

    if action_policy == "next_state":
        actions = []
        for i, state in enumerate(states):
            if i + 1 < len(states):
                actions.append(states[i + 1].copy())
            else:
                actions.append(states[i].copy())

    rich = {
        "joints": [],
        "tcp": [],
        "flange": [],
        "fk": [],
        "gripper": [],
    }
    for sample in kept:
        j = _joints(sample, "state")
        rich["joints"].append(j if j is not None else [math.nan] * 7)
        for key, bucket in (
            ("tcp_pose", "tcp"),
            ("flange_pose", "flange"),
            ("fk_pose", "fk"),
        ):
            pose = _pose(sample, key)
            rich[bucket].append(pose if pose is not None else [math.nan] * 6)
        g = _grasp(sample, "state")
        rich["gripper"].append(float(g) if g is not None else math.nan)

    return {
        "observation.state": np.stack(states).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "timestamp": np.asarray(timestamps, dtype=np.float32),
        "rich.joints": np.asarray(rich["joints"], dtype=np.float32),
        "rich.tcp": np.asarray(rich["tcp"], dtype=np.float32),
        "rich.flange": np.asarray(rich["flange"], dtype=np.float32),
        "rich.fk": np.asarray(rich["fk"], dtype=np.float32),
        "rich.gripper": np.asarray(rich["gripper"], dtype=np.float32),
        "samples": kept,
        "task": episode.meta.get("task") or "",
    }


def export_numpy_staging(
    episodes: list[EpisodeBundle],
    output_dir: Path,
    state_mode: str,
    action_mode: str,
    action_policy: str,
    fps: float,
    repo_id: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    episode_rows = []
    total_frames = 0
    state_dim = None
    action_dim = None

    for ep_index, episode in enumerate(episodes):
        arrays = build_episode_arrays(episode, state_mode, action_mode, action_policy)
        if state_dim is None:
            state_dim = int(arrays["observation.state"].shape[1])
            action_dim = int(arrays["action"].shape[1])
        np.savez_compressed(
            data_dir / f"episode_{ep_index:06d}.npz",
            observation_state=arrays["observation.state"],
            action=arrays["action"],
            timestamp=arrays["timestamp"],
            rich_joints=arrays["rich.joints"],
            rich_tcp=arrays["rich.tcp"],
            rich_flange=arrays["rich.flange"],
            rich_fk=arrays["rich.fk"],
            rich_gripper=arrays["rich.gripper"],
        )
        video_rel = None
        if episode.video is not None and episode.video.exists():
            dest = video_dir / f"episode_{ep_index:06d}.mp4"
            shutil.copy2(episode.video, dest)
            video_rel = str(dest.relative_to(output_dir)).replace("\\", "/")
        n = int(arrays["observation.state"].shape[0])
        total_frames += n
        episode_rows.append(
            {
                "episode_index": ep_index,
                "length": n,
                "task": arrays["task"],
                "source_jsonl": str(episode.jsonl),
                "video": video_rel,
            }
        )

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": None,
            "mode": state_mode if state_mode != "all" else "tcp+gripper",
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": None,
            "mode": action_mode if action_mode != "all" else "tcp+gripper",
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": [3, 720, 1280],
            "names": ["channels", "height", "width"],
            "info": "Copied from recording RGB MP4; decode by frame_index when training.",
        },
        "rich.joints": {"dtype": "float32", "shape": [7]},
        "rich.tcp": {"dtype": "float32", "shape": [6]},
        "rich.flange": {"dtype": "float32", "shape": [6]},
        "rich.fk": {"dtype": "float32", "shape": [6]},
        "rich.gripper": {"dtype": "float32", "shape": [1]},
    }
    info = {
        "codebase_version": "v2.1-staging",
        "robot_type": "nero_agx_gripper",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "repo_id": repo_id,
        "state_mode": state_mode,
        "action_mode": action_mode,
        "action_policy": action_policy,
        "features": features,
        "notes": (
            "Staging export from nero_teleop_jsonl. "
            "observation.state/action are selected views; rich.* keeps all pose/joint/gripper channels. "
            "Install lerobot and re-run with --prefer-lerobot-api for a Hub-native LeRobot dataset, "
            "or convert this staging set with your OpenPI data pipeline."
        ),
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tasks = sorted({row["task"] or "default" for row in episode_rows})
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}, ensure_ascii=False) + "\n")
    print(f"Wrote staging LeRobot-ready dataset to {output_dir}")
    print(f"  episodes={len(episodes)} frames={total_frames} state_dim={state_dim} action_dim={action_dim}")


def export_with_lerobot_api(
    episodes: list[EpisodeBundle],
    output_dir: Path,
    state_mode: str,
    action_mode: str,
    action_policy: str,
    fps: float,
    repo_id: str,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    # Probe dims from first episode
    first = build_episode_arrays(episodes[0], state_mode, action_mode, action_policy)
    state_dim = int(first["observation.state"].shape[1])
    action_dim = int(first["action"].shape[1])
    height, width = 720, 1280
    meta0 = episodes[0].meta.get("camera_rgb") or {}
    height = int(meta0.get("height") or height)
    width = int(meta0.get("width") or width)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="nero_agx_gripper",
        use_videos=True,
    )

    import cv2  # type: ignore

    for episode in episodes:
        arrays = build_episode_arrays(episode, state_mode, action_mode, action_policy)
        task = arrays["task"] or "nero_teleop"
        cap = None
        if episode.video is not None and episode.video.exists():
            cap = cv2.VideoCapture(str(episode.video))
        try:
            for i in range(len(arrays["observation.state"])):
                frame = None
                if cap is not None:
                    # Prefer alignment index from raw sample when available.
                    sample = arrays["samples"][i]
                    cam = sample.get("camera_rgb") or {}
                    idx = cam.get("video_frame_index")
                    if idx is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame is None:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                dataset.add_frame(
                    {
                        "observation.state": arrays["observation.state"][i],
                        "action": arrays["action"][i],
                        "observation.images.head": frame,
                        "task": task,
                    }
                )
            dataset.save_episode()
        finally:
            if cap is not None:
                cap.release()

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"Wrote LeRobot dataset via API to {output_dir}")


def main() -> int:
    args = parse_args()
    episodes = discover_episodes(args.input_dir, args.glob)
    fps = args.fps
    if fps is None:
        fps = float(episodes[0].meta.get("sample_hz") or 20.0)

    if args.prefer_lerobot_api:
        try:
            export_with_lerobot_api(
                episodes,
                args.output_dir,
                args.state_mode,
                args.action_mode,
                args.action_policy,
                fps,
                args.repo_id,
            )
            return 0
        except ImportError:
            print(
                "lerobot not installed; falling back to staging export. "
                "pip install lerobot  then re-run with --prefer-lerobot-api",
                file=sys.stderr,
            )

    export_numpy_staging(
        episodes,
        args.output_dir,
        args.state_mode,
        args.action_mode,
        args.action_policy,
        fps,
        args.repo_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
