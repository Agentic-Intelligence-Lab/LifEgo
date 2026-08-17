#!/usr/bin/env python3
"""Train OpenPI on the local Nero EEF LeRobot dataset."""

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
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=not args.full_finetune,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        wandb_enabled=args.wandb,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    from scripts import train as openpi_train

    openpi_train.main(config)


if __name__ == "__main__":
    main()
