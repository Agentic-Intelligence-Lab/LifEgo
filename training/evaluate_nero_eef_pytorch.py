#!/usr/bin/env python3
"""Evaluate a PyTorch Nero EEF checkpoint on training samples."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import sys

import numpy as np
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.nero_eef_config import DEFAULT_DATASET_ROOT, DEFAULT_REPO_ID, build_config, dataset_home_from_root
from training.nero_eef_policy import ACTION_DIM


ACTION_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")


def _resolve_checkpoint_dir(checkpoint_dir: Path, step: int | None) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if (checkpoint_dir / "model.safetensors").is_file():
        if step is not None:
            raise ValueError("--step cannot be used when --checkpoint-dir already points to a step directory")
        return checkpoint_dir

    if step is not None:
        step_dir = checkpoint_dir / str(step)
        if not (step_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing model.safetensors under {step_dir}")
        return step_dir

    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoint step directories found under {checkpoint_dir}")
    return checkpoint_dir / str(steps[-1])


def _build_model(config, device: torch.device):
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch

    model_cfg = config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = dataclasses.replace(model_cfg, dtype=config.pytorch_training_precision, pytorch_compile_mode=None)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.eval()
    return model


def _move_to_device(tree, device: torch.device):
    import jax

    return jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, tree)


def _unnormalize_actions(actions: np.ndarray, norm_stats: dict, *, use_quantiles: bool) -> np.ndarray:
    stats = norm_stats["actions"]
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float32)[..., : actions.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[..., : actions.shape[-1]]
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    mean = np.asarray(stats.mean, dtype=np.float32)[..., : actions.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[..., : actions.shape[-1]]
    return actions * (std + 1e-6) + mean


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def _per_dim_rmse(error: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(error), axis=tuple(range(error.ndim - 1))))


def _quat_angle_deg(pred_quat: np.ndarray, label_quat: np.ndarray) -> np.ndarray:
    pred_norm = pred_quat / np.clip(np.linalg.norm(pred_quat, axis=-1, keepdims=True), 1e-8, None)
    label_norm = label_quat / np.clip(np.linalg.norm(label_quat, axis=-1, keepdims=True), 1e-8, None)
    dot = np.abs(np.sum(pred_norm * label_norm, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _summarize(prefix: str, pred: np.ndarray, label: np.ndarray) -> None:
    error = pred - label
    per_dim = _per_dim_rmse(error)
    pos_rmse = _rmse(error[..., :3])
    quat_rmse = _rmse(error[..., 3:7])
    gripper_rmse = _rmse(error[..., 7:8])
    quat_angle = _quat_angle_deg(pred[..., 3:7], label[..., 3:7])

    print(f"{prefix} overall_rmse: { _rmse(error):.6f}")
    print(f"{prefix} position_rmse_m: {pos_rmse:.6f}")
    print(f"{prefix} quaternion_rmse: {quat_rmse:.6f}")
    print(f"{prefix} gripper_rmse: {gripper_rmse:.6f}")
    print(f"{prefix} quat_angle_mean_deg: {float(np.mean(quat_angle)):.3f}")
    print(f"{prefix} quat_angle_p95_deg: {float(np.percentile(quat_angle, 95)):.3f}")
    print(f"{prefix} per_dim_rmse:")
    for name, value in zip(ACTION_NAMES, per_dim, strict=True):
        print(f"  {name}: {float(value):.6f}")


def evaluate(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=1,
        pytorch_weight_path=None,
        wandb_enabled=False,
    )
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else config.checkpoint_dir
    step_dir = _resolve_checkpoint_dir(checkpoint_root, args.step)

    model = _build_model(config, device)
    safetensors.torch.load_model(model, step_dir / "model.safetensors", device=str(device))

    from openpi.training import data_loader

    num_batches = math.ceil(args.num_samples / args.batch_size)
    loader = data_loader.create_data_loader(
        config,
        shuffle=args.shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    data_config = loader.data_config()

    pred_batches = []
    label_batches = []
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    with torch.inference_mode():
        remaining = args.num_samples
        for observation, actions in loader:
            take = min(remaining, actions.shape[0])
            observation = _move_to_device(observation, device)
            labels = actions.to(device=device, dtype=torch.float32)
            if args.zero_noise:
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
            preds = model.sample_actions(device, observation, noise=noise, num_steps=args.sample_steps)
            pred_batches.append(preds[:take, :, :ACTION_DIM].detach().cpu().numpy())
            label_batches.append(labels[:take, :, :ACTION_DIM].detach().cpu().numpy())
            remaining -= take
            if remaining <= 0:
                break

    pred_norm = np.concatenate(pred_batches, axis=0)
    label_norm = np.concatenate(label_batches, axis=0)
    pred_eef = _unnormalize_actions(pred_norm, data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    label_eef = _unnormalize_actions(label_norm, data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)

    print(f"checkpoint: {step_dir}")
    print(f"samples: {pred_norm.shape[0]}")
    print(f"action_horizon: {pred_norm.shape[1]}")
    print(f"action_dim: {ACTION_DIM}")
    print(f"sample_steps: {args.sample_steps}")
    print(f"noise: {'zero' if args.zero_noise else f'fixed_random_seed_{args.seed}'}")
    print()
    _summarize("normalized/all_horizon", pred_norm, label_norm)
    print()
    _summarize("eef/all_horizon", pred_eef, label_eef)
    print()
    _summarize("eef/horizon0", pred_eef[:, :1], label_eef[:, :1])

    if args.output_npz:
        out = Path(args.output_npz).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            pred_normalized=pred_norm,
            label_normalized=label_norm,
            pred_eef=pred_eef,
            label_eef=label_eef,
            checkpoint=str(step_dir),
        )
        print(f"\nwrote: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--exp-name", default="nero_eef_pi05_pytorch_v1")
    parser.add_argument("--checkpoint-dir", default=None, help="Experiment dir or exact step dir. Defaults from exp name.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load. Defaults to latest.")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-npz", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
