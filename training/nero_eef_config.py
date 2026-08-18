"""Local OpenPI training config for Nero EEF datasets."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.training.config as openpi_config
import openpi.training.optimizer as openpi_optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms

from training import nero_eef_policy


DEFAULT_REPO_ID = "local/nero_ego_ymq_eef"
DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs" / "lerobot" / DEFAULT_REPO_ID
DEFAULT_TASK_PROMPT = "Place the black pillar in the plate."


@dataclasses.dataclass(frozen=True)
class NeroEefDataConfig(openpi_config.DataConfigFactory):
    """LeRobot DataConfig for ego RGB to Nero EEF action training."""

    default_prompt: str | None = DEFAULT_TASK_PROMPT

    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> openpi_config.DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = transforms.Group(
            inputs=[nero_eef_policy.NeroEefInputs(model_type=model_config.model_type)],
            outputs=[nero_eef_policy.NeroEefOutputs()],
        )
        model_transforms = openpi_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def build_config(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    exp_name: str = "nero_eef_debug",
    model: str = "pi0_fast",
    low_mem: bool = True,
    batch_size: int = 16,
    num_train_steps: int = 30_000,
    save_interval: int = 1000,
    log_interval: int = 100,
    num_workers: int = 2,
    assets_base_dir: str | None = None,
    checkpoint_base_dir: str | None = None,
    pytorch_weight_path: str | None = None,
    wandb_enabled: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> openpi_config.TrainConfig:
    if model == "pi0_fast":
        model_config = pi0_fast.Pi0FASTConfig(
            action_dim=nero_eef_policy.ACTION_DIM,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora" if low_mem else "gemma_2b",
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    elif model == "pi0":
        model_config = pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora" if low_mem else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if low_mem else "gemma_300m",
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi0_base/params"
    elif model == "pi05":
        model_config = pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if low_mem else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if low_mem else "gemma_300m",
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi05_base/params"
    else:
        raise ValueError(f"unsupported model: {model}")

    freeze_filter = model_config.get_freeze_filter() if low_mem else nnx.Nothing
    return openpi_config.TrainConfig(
        name="nero_eef",
        project_name="lifego",
        exp_name=exp_name,
        model=model_config,
        data=NeroEefDataConfig(
            repo_id=repo_id,
            base_config=openpi_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint),
        pytorch_weight_path=pytorch_weight_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        log_interval=log_interval,
        assets_base_dir=str(REPO_ROOT / "outputs" / "openpi_assets") if assets_base_dir is None else assets_base_dir,
        checkpoint_base_dir=str(REPO_ROOT / "outputs" / "openpi_checkpoints")
        if checkpoint_base_dir is None
        else checkpoint_base_dir,
        lr_schedule=openpi_optimizer.CosineDecaySchedule(
            warmup_steps=min(1_000, max(num_train_steps // 10, 1)),
            peak_lr=5e-5,
            decay_steps=max(num_train_steps, 1),
            decay_lr=5e-5,
        ),
        optimizer=openpi_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=freeze_filter,
        ema_decay=None if low_mem else 0.99,
        wandb_enabled=wandb_enabled,
        overwrite=overwrite,
        resume=resume,
    )


def dataset_home_from_root(dataset_root: str | Path, repo_id: str = DEFAULT_REPO_ID) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    repo_parts = Path(repo_id).parts
    if len(root.parts) >= len(repo_parts) and root.parts[-len(repo_parts) :] == repo_parts:
        return Path(*root.parts[: -len(repo_parts)])
    if (root / repo_id / "meta" / "info.json").is_file():
        return root
    return root.parent.parent if len(root.parts) >= 2 else root
