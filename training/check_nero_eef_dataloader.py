#!/usr/bin/env python3
"""Sanity-check the Nero EEF OpenPI data pipeline without training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.nero_eef_config import DEFAULT_DATASET_ROOT, DEFAULT_REPO_ID, build_config, dataset_home_from_root
from training.nero_eef_policy import ACTION_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    from openpi.training import data_loader

    config = build_config(
        repo_id=args.repo_id,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=10,
        wandb_enabled=False,
    )
    loader = data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=args.skip_norm_stats,
    )
    observation, actions = next(iter(loader))
    print("images:")
    for key, value in observation.images.items():
        print(f"  {key}: {value.shape} {value.dtype}")
    print("image masks:")
    for key, value in observation.image_masks.items():
        print(f"  {key}: {value.shape} {value.dtype} first={value[0]}")
    print(f"state: {observation.state.shape} {observation.state.dtype}")
    print(f"actions: {actions.shape} {actions.dtype}")
    if observation.state.shape[-1] > ACTION_DIM:
        print(f"state padded tail abs max: {np.abs(np.asarray(observation.state[..., ACTION_DIM:])).max():.6g}")
    if actions.shape[-1] > ACTION_DIM:
        print(f"actions padded tail abs max: {np.abs(np.asarray(actions[..., ACTION_DIM:])).max():.6g}")


if __name__ == "__main__":
    main()
