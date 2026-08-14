from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
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
from real_robot_data_collector.utils.image_utils import read_image_rgb
from real_robot_data_collector.utils.json_utils import load_json


def export_act_hdf5(
    input_dir: str | Path,
    output_path: str | Path,
    episodes: list[str] | None = None,
    image_size: tuple[int, int] | None = None,
    action_policy: str = "next_qpos",
    compress: bool = True,
) -> None:
    episode_dirs = resolve_episode_dirs(input_dir, episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {input_dir}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "ACT-compatible HDF5"
        h5.attrs["action_policy_default"] = action_policy
        for ep_dir in episode_dirs:
            target = h5 if len(episode_dirs) == 1 else h5.create_group(ep_dir.name)
            _write_episode(target, ep_dir, image_size=image_size, action_policy=action_policy, compress=compress)


def _write_episode(group: h5py.Group, episode_dir: Path, image_size: tuple[int, int] | None, action_policy: str, compress: bool) -> None:
    arrays = load_arrays(episode_dir)
    qpos = qpos_from_arrays(arrays)
    action, action_source = choose_action(arrays, qpos, action_policy)
    image_paths = image_paths_for_episode(episode_dir, arrays)
    images = np.stack([read_image_rgb(path, image_size=image_size) for path in image_paths], axis=0).astype(np.uint8)

    compression = "gzip" if compress else None
    observations = group.create_group("observations")
    images_group = observations.create_group("images")
    images_group.create_dataset("head", data=images, compression=compression)
    observations.create_dataset("qpos", data=qpos.astype(np.float32), compression=compression)
    group.create_dataset("action", data=action.astype(np.float32), compression=compression)
    group.attrs["action_source"] = action_source
    metadata = load_json(episode_dir / "metadata.json", default={})
    group.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False, allow_nan=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw robot episodes to ACT-compatible HDF5.")
    parser.add_argument("--input-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--action-policy", choices=["recorded", "next_qpos", "auto"], default="next_qpos")
    parser.add_argument("--no-compress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_size = tuple(args.image_size) if args.image_size else None
    export_act_hdf5(
        input_dir=args.input_dir,
        output_path=args.output,
        episodes=args.episodes,
        image_size=image_size,
        action_policy=args.action_policy,
        compress=not args.no_compress,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
