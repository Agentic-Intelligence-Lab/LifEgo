#!/usr/bin/env python3
"""PyTorch/DDP training entry point for Nero EEF Pi0/Pi0.5 models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from training.nero_eef_config import DEFAULT_DATASET_ROOT, build_config, dataset_home_from_root
from training.nero_eef_policy import ACTION_DIM


DEFAULT_PI05_WEIGHT_PATH = "/mnt/data/szeluresearch/models/pi05_base"


def _patch_pytorch_action_loss_dim(loss_action_dim: int) -> None:
    """Keep the 32D Pi0.5 head, but supervise only the real Nero EEF dims."""
    if loss_action_dim <= 0:
        return

    from openpi.models_pytorch import pi0_pytorch

    original_forward = pi0_pytorch.PI0Pytorch.forward
    if getattr(original_forward, "_nero_eef_loss_patched", False):
        return

    def forward_with_cropped_loss(self, observation, actions, noise=None, time=None):
        losses = original_forward(self, observation, actions, noise=noise, time=time)
        if loss_action_dim >= losses.shape[-1]:
            return losses
        return losses[..., :loss_action_dim]

    forward_with_cropped_loss._nero_eef_loss_patched = True
    pi0_pytorch.PI0Pytorch.forward = forward_with_cropped_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="local/nero_ego_ymq_eef")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--exp-name", default="nero_eef_pi05_pytorch")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--pytorch-weight-path", default=DEFAULT_PI05_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--loss-action-dim",
        type=int,
        default=ACTION_DIM,
        help="Only the first N action dimensions contribute to the PyTorch loss.",
    )
    args = parser.parse_args()

    weight_path = Path(args.pytorch_weight_path).expanduser()
    if not (weight_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing model.safetensors under {weight_path}")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        pytorch_weight_path=str(weight_path),
        wandb_enabled=args.wandb,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if args.loss_action_dim > config.model.action_dim:
        raise ValueError(f"--loss-action-dim={args.loss_action_dim} exceeds model action_dim={config.model.action_dim}")

    from scripts import train_pytorch as openpi_train_pytorch

    openpi_train_pytorch.init_logging()
    _patch_pytorch_action_loss_dim(args.loss_action_dim)
    openpi_train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
