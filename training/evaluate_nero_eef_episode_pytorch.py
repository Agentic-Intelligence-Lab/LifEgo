#!/usr/bin/env python3
"""Evaluate a PyTorch Nero EEF checkpoint on complete LeRobot episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.models import model as _model
from openpi.training import data_loader
from training.evaluate_nero_eef_pytorch import (
    _build_model,
    _move_to_device,
    _quat_angle_deg,
    _resolve_checkpoint_dir,
    _rmse,
    _summarize,
    _unnormalize_actions,
)
from training.nero_eef_config import DEFAULT_DATASET_ROOT, DEFAULT_REPO_ID, build_config, dataset_home_from_root
from training.nero_eef_policy import ACTION_DIM


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_ranges(dataset_root: Path) -> dict[int, tuple[int, int]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {episodes_path}")

    ranges = {}
    start = 0
    for rec in _read_jsonl(episodes_path):
        episode_index = int(rec["episode_index"])
        length = int(rec["length"])
        ranges[episode_index] = (start, start + length)
        start += length
    return ranges


def _dataset_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return float(info.get("fps", 30.0))


def _collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def _trajectory_records(traj: np.ndarray, fps: float, *, source: str) -> dict[str, Any]:
    traj = np.asarray(traj, dtype=np.float64)
    quat = _normalize_quat(traj[:, 3:7])
    records = []
    for i, action in enumerate(traj):
        records.append(
            {
                "frame_index": i,
                "ts": int(round(i / fps * 1e9)),
                "valid": True,
                "gripper": float(action[7]),
                "T_ee_in_base": {
                    "translation_m": action[:3].tolist(),
                    "quat_xyzw": quat[i].tolist(),
                },
            }
        )
    return {
        "metadata": {
            "source": source,
            "units": "meters",
            "fps": fps,
            "action_layout": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "records": records,
    }


def _write_trajectory_json(path: Path, traj: np.ndarray, fps: float, *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_trajectory_records(traj, fps, source=source), indent=2), encoding="utf-8")


def _episode_indices(args: argparse.Namespace, ranges: dict[int, tuple[int, int]]) -> list[int]:
    if args.all_episodes:
        return sorted(ranges)
    if args.episode is not None:
        if args.episode not in ranges:
            raise ValueError(f"episode {args.episode} not found. Available: {sorted(ranges)}")
        return [args.episode]
    return [min(ranges)]


def _evaluate_one_episode(
    *,
    episode_index: int,
    index_range: tuple[int, int],
    dataset,
    model,
    config,
    data_config,
    device: torch.device,
    batch_size: int,
    sample_steps: int,
    seed: int,
    zero_noise: bool,
    fps: float,
    include_padded_last: bool,
) -> dict[str, Any]:
    start, end = index_range
    eval_end = end if include_padded_last else max(start, end - 1)
    pred_batches = []
    label_batches = []
    frame_indices = []
    timestamps = []
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + episode_index)

    with torch.inference_mode():
        for batch_start in range(start, eval_end, batch_size):
            batch_end = min(batch_start + batch_size, eval_end)
            items = [dataset[i] for i in range(batch_start, batch_end)]
            batch = jax.tree.map(torch.as_tensor, _collate(items))
            observation = _model.Observation.from_dict(batch)
            observation = _move_to_device(observation, device)
            labels = batch["actions"].to(device=device, dtype=torch.float32)

            if zero_noise:
                noise = torch.zeros(
                    labels.shape[0],
                    config.model.action_horizon,
                    config.model.action_dim,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                noise = torch.randn(
                    labels.shape[0],
                    config.model.action_horizon,
                    config.model.action_dim,
                    dtype=torch.float32,
                    device=device,
                    generator=rng,
                )
            preds = model.sample_actions(device, observation, noise=noise, num_steps=sample_steps)
            pred_batches.append(preds[:, 0, :ACTION_DIM].detach().cpu().numpy())
            label_batches.append(labels[:, 0, :ACTION_DIM].detach().cpu().numpy())

            for global_index in range(batch_start, batch_end):
                frame_index = global_index - start
                frame_indices.append(frame_index)
                timestamps.append(frame_index / fps)

    pred_norm = np.concatenate(pred_batches, axis=0)
    label_norm = np.concatenate(label_batches, axis=0)
    pred_eef = _unnormalize_actions(pred_norm, data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    label_eef = _unnormalize_actions(label_norm, data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    error = pred_eef - label_eef
    quat_angle = _quat_angle_deg(pred_eef[:, 3:7], label_eef[:, 3:7])

    return {
        "episode_index": episode_index,
        "start_index": start,
        "end_index": eval_end,
        "episode_num_frames": end - start,
        "num_eval_frames": eval_end - start,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "pred_normalized": pred_norm,
        "label_normalized": label_norm,
        "pred_eef": pred_eef,
        "label_eef": label_eef,
        "overall_rmse": _rmse(error),
        "position_rmse_m": _rmse(error[:, :3]),
        "quaternion_rmse": _rmse(error[:, 3:7]),
        "gripper_rmse": _rmse(error[:, 7:8]),
        "quat_angle_mean_deg": float(np.mean(quat_angle)),
        "quat_angle_p95_deg": float(np.percentile(quat_angle, 95)),
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sample_steps <= 0:
        raise ValueError("--sample-steps must be positive")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(dataset_root, args.repo_id))
    fps = _dataset_fps(dataset_root)
    ranges = _episode_ranges(dataset_root)
    episodes = _episode_indices(args, ranges)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=1,
        pytorch_weight_path=None,
        wandb_enabled=False,
    )
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else config.checkpoint_dir
    step_dir = _resolve_checkpoint_dir(checkpoint_root, args.step)

    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)

    model = _build_model(config, device)
    safetensors.torch.load_model(model, step_dir / "model.safetensors", device=str(device))

    results = []
    for episode_index in episodes:
        result = _evaluate_one_episode(
            episode_index=episode_index,
            index_range=ranges[episode_index],
            dataset=transformed_dataset,
            model=model,
            config=config,
            data_config=data_config,
            device=device,
            batch_size=args.batch_size,
            sample_steps=args.sample_steps,
            seed=args.seed,
            zero_noise=args.zero_noise,
            fps=fps,
            include_padded_last=args.include_padded_last,
        )
        results.append(result)

    print(f"checkpoint: {step_dir}")
    print(f"episodes: {episodes}")
    print(f"fps: {fps:g}")
    print(f"sample_steps: {args.sample_steps}")
    print(f"noise: {'zero' if args.zero_noise else f'fixed_random_seed_{args.seed}'}")

    for result in results:
        print()
        print(
            f"episode {result['episode_index']} eval_frames: {result['num_eval_frames']} "
            f"(episode_frames={result['episode_num_frames']})"
        )
        _summarize(
            f"episode{result['episode_index']}/horizon0_normalized",
            result["pred_normalized"][:, None, :],
            result["label_normalized"][:, None, :],
        )
        print()
        _summarize(
            f"episode{result['episode_index']}/horizon0_eef",
            result["pred_eef"][:, None, :],
            result["label_eef"][:, None, :],
        )

    if len(results) > 1:
        pred_all = np.concatenate([item["pred_eef"] for item in results], axis=0)
        label_all = np.concatenate([item["label_eef"] for item in results], axis=0)
        print()
        print(f"all selected episodes frames: {pred_all.shape[0]}")
        _summarize("all_selected/horizon0_eef", pred_all[:, None, :], label_all[:, None, :])

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(step_dir),
            "episodes": episodes,
            "fps": fps,
            "sample_steps": args.sample_steps,
            "noise": "zero" if args.zero_noise else f"fixed_random_seed_{args.seed}",
            "metrics": [
                {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "episode_index",
                        "start_index",
                        "end_index",
                        "episode_num_frames",
                        "num_eval_frames",
                        "overall_rmse",
                        "position_rmse_m",
                        "quaternion_rmse",
                        "gripper_rmse",
                        "quat_angle_mean_deg",
                        "quat_angle_p95_deg",
                    }
                }
                for result in results
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        for result in results:
            episode_dir = out_dir / f"episode_{result['episode_index']:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                episode_dir / "pred_vs_label.npz",
                frame_indices=result["frame_indices"],
                timestamps=result["timestamps"],
                pred_normalized=result["pred_normalized"],
                label_normalized=result["label_normalized"],
                pred_eef=result["pred_eef"],
                label_eef=result["label_eef"],
            )
            _write_trajectory_json(
                episode_dir / "pred_robot_eef_trajectory.json",
                result["pred_eef"],
                fps,
                source=f"OpenPI prediction, checkpoint={step_dir}, episode={result['episode_index']}",
            )
            _write_trajectory_json(
                episode_dir / "label_robot_eef_trajectory.json",
                result["label_eef"],
                fps,
                source=f"LeRobot label, episode={result['episode_index']}",
            )
        print(f"\nwrote: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--exp-name", default="nero_eef_pi05_pytorch_v1")
    parser.add_argument("--checkpoint-dir", default=None, help="Experiment dir or exact step dir. Defaults from exp name.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load. Defaults to latest.")
    parser.add_argument("--episode", type=int, default=None, help="Episode index. Defaults to the first episode.")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument(
        "--include-padded-last",
        action="store_true",
        help="Include the final padded next-frame label of each episode in RMSE.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
