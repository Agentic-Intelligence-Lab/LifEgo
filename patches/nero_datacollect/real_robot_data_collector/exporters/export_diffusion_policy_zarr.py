from __future__ import annotations

import argparse
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
from real_robot_data_collector.utils.image_utils import read_image_rgb


def export_diffusion_policy_zarr(
    input_dir: str | Path,
    output_path: str | Path,
    episodes: list[str] | None = None,
    image_size: tuple[int, int] = (224, 224),
    compress: bool = False,
    action_policy: str = "recorded",
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required. Install requirements.txt first.") from exc

    episode_dirs = resolve_episode_dirs(input_dir, episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {input_dir}")

    all_img = []
    all_state = []
    all_action = []
    episode_ends = []
    cursor = 0
    for ep_dir in episode_dirs:
        arrays = load_arrays(ep_dir)
        state = qpos_from_arrays(arrays)
        action, _ = choose_action(arrays, state, action_policy)
        img = np.stack(
            [read_image_rgb(path, image_size=image_size) for path in image_paths_for_episode(ep_dir, arrays)],
            axis=0,
        ).astype(np.uint8)
        all_img.append(img)
        all_state.append(state.astype(np.float32))
        all_action.append(action.astype(np.float32))
        cursor += len(state)
        episode_ends.append(cursor)

    img_array = np.concatenate(all_img, axis=0)
    state_array = np.concatenate(all_state, axis=0)
    action_array = np.concatenate(all_action, axis=0)
    episode_ends_array = np.asarray(episode_ends, dtype=np.int64)

    output_path = Path(output_path)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    try:
        root = zarr.open_group(str(output_path), mode="w", zarr_format=2)
    except TypeError:
        root = zarr.open_group(str(output_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = _make_compressor() if compress else None
    _create_dataset(data_group, "img", img_array, compressor)
    _create_dataset(data_group, "state", state_array, compressor)
    _create_dataset(data_group, "action", action_array, compressor)
    _create_dataset(meta_group, "episode_ends", episode_ends_array, None)
    root.attrs["format"] = "Diffusion Policy replay buffer"
    root.attrs["image_size"] = list(image_size)
    root.attrs["action_policy"] = action_policy


def _make_compressor():
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        return None


def _create_dataset(group, name: str, data: np.ndarray, compressor) -> None:
    kwargs = {"data": data, "shape": data.shape, "dtype": data.dtype}
    if compressor is not None:
        kwargs["compressor"] = compressor
    try:
        group.create_dataset(name, **kwargs)
    except TypeError:
        kwargs.pop("compressor", None)
        group.create_dataset(name, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw robot episodes to Diffusion Policy Zarr.")
    parser.add_argument("--input-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=[224, 224])
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--action-policy", choices=["recorded", "next_qpos", "auto"], default="recorded")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_diffusion_policy_zarr(
        input_dir=args.input_dir,
        output_path=args.output,
        episodes=args.episodes,
        image_size=tuple(args.image_size),
        compress=args.compress,
        action_policy=args.action_policy,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
