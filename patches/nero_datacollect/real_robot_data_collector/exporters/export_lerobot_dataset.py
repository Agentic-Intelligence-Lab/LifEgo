from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from real_robot_data_collector.exporters.common import (
    choose_action,
    image_paths_for_episode,
    load_arrays,
    qpos_from_arrays,
    resolve_episode_dirs,
)
from real_robot_data_collector.utils.json_utils import append_jsonl_line, load_json, write_json


def export_lerobot_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    episodes: list[str] | None = None,
    action_policy: str = "recorded",
    task_index: int = 0,
) -> None:
    episode_dirs = resolve_episode_dirs(input_dir, episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {input_dir}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "meta").mkdir(parents=True)

    frames_path = output_dir / "frames.jsonl"
    global_frame_index = 0
    summary_episodes = []
    with frames_path.open("w", encoding="utf-8") as frames_file:
        for episode_index, ep_dir in enumerate(episode_dirs):
            arrays = load_arrays(ep_dir)
            metadata = load_json(ep_dir / "metadata.json", default={})
            state = qpos_from_arrays(arrays)
            action, action_source = choose_action(arrays, state, action_policy)
            timestamps = arrays["timestamps_unix"].astype(np.float64)
            frame_indices = np.arange(len(state), dtype=np.int64)
            language_instruction = metadata.get("language_instruction", "")

            np.savez_compressed(
                output_dir / "data" / f"{ep_dir.name}.npz",
                observation_state=state.astype(np.float32),
                action=action.astype(np.float32),
                timestamp=timestamps,
                episode_index=np.full((len(state),), episode_index, dtype=np.int64),
                frame_index=frame_indices,
                task_index=np.full((len(state),), task_index, dtype=np.int64),
                language_instruction=np.asarray([language_instruction] * len(state), dtype=object),
            )

            image_out_dir = output_dir / "images" / ep_dir.name / "head"
            image_out_dir.mkdir(parents=True, exist_ok=True)
            for local_frame_index, src in enumerate(image_paths_for_episode(ep_dir, arrays)):
                dst = image_out_dir / src.name
                shutil.copy2(src, dst)
                append_jsonl_line(
                    frames_file,
                    {
                        "observation.images.head": str(dst.relative_to(output_dir)),
                        "observation.state": state[local_frame_index].astype(float).tolist(),
                        "action": action[local_frame_index].astype(float).tolist(),
                        "timestamp": float(timestamps[local_frame_index]),
                        "episode_index": int(episode_index),
                        "frame_index": int(local_frame_index),
                        "global_frame_index": int(global_frame_index),
                        "task_index": int(task_index),
                        "language_instruction": language_instruction,
                    },
                )
                global_frame_index += 1
            summary_episodes.append(
                {
                    "episode_name": ep_dir.name,
                    "episode_index": episode_index,
                    "num_frames": int(len(state)),
                    "language_instruction": language_instruction,
                    "action_source": action_source,
                }
            )

    write_json(
        output_dir / "meta" / "info.json",
        {
            "format": "LeRobot/OpenPI mappable normalized dataset",
            "source_input_dir": str(input_dir),
            "num_episodes": len(summary_episodes),
            "num_frames": global_frame_index,
            "state_dim": 13,
            "action_dim": 13,
            "image_key": "observation.images.head",
            "state_key": "observation.state",
            "action_key": "action",
            "note": "OpenPI/pi0.5 usually still needs a task-specific data mapping in its training config.",
            "episodes": summary_episodes,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized data for LeRobot/OpenPI mapping.")
    parser.add_argument("--input-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--action-policy", choices=["recorded", "next_qpos", "auto"], default="recorded")
    parser.add_argument("--task-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_lerobot_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        action_policy=args.action_policy,
        task_index=args.task_index,
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
