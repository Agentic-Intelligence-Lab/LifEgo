#!/usr/bin/env python3
"""Compute OpenPI normalization stats for the local Nero EEF config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.nero_eef_config import DEFAULT_DATASET_ROOT, build_config, dataset_home_from_root
from openpi import transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute(args: argparse.Namespace) -> None:
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    from openpi.shared import normalize
    from openpi.training import data_loader

    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=not args.full_finetune,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        wandb_enabled=False,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    num_batches = max(len(dataset) // args.batch_size, 1)
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_batches=num_batches,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing Nero EEF norm stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    out = config.assets_dirs / data_config.repo_id
    out.mkdir(parents=True, exist_ok=True)
    normalize.save(out, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Wrote norm stats: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="local/nero_ego_ymq_eef")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--exp-name", default="nero_eef_debug")
    parser.add_argument("--model", choices=["pi0_fast", "pi0", "pi05"], default="pi0_fast")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    args = parser.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
