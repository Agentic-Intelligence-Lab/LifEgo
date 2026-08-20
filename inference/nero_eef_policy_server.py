#!/usr/bin/env python3
"""Serve a trained Nero EEF Pi0.5 PyTorch policy over OpenPI websocket."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys
import time
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_SRC, OPENPI_CLIENT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi import transforms
from openpi.models import model as _model
from openpi.serving import websocket_policy_server
from training.evaluate_nero_eef_pytorch import _build_model, _resolve_checkpoint_dir, _unnormalize_actions
from training.nero_eef_config import DEFAULT_REPO_ID, DEFAULT_TASK_PROMPT, build_config
from training.nero_eef_policy import ACTION_DIM


DEFAULT_DEVICE = "cuda:0"
DEFAULT_SAMPLE_STEPS = 10
DEFAULT_ACTIONS_PER_INFERENCE = 10
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclasses.dataclass
class PolicyRuntime:
    config: Any
    data_config: Any
    model: Any
    input_transform: Any
    device: torch.device
    rng: torch.Generator
    zero_noise: bool
    sample_steps: int
    checkpoint: Path


def build_policy_runtime(args: argparse.Namespace) -> PolicyRuntime:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=1,
        num_workers=0,
        num_train_steps=1,
        pytorch_weight_path=None,
        wandb_enabled=False,
    )
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else config.checkpoint_dir
    step_dir = _resolve_checkpoint_dir(checkpoint_root, args.step)
    data_config = config.data.create(config.assets_dirs, config.model)
    model = _build_model(config, device)
    safetensors.torch.load_model(model, step_dir / "model.safetensors", device=str(device))

    input_transform = transforms.compose(
        [
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    print("===== policy server =====")
    print(f"checkpoint: {step_dir}")
    print(f"device: {device}")
    print(f"sample_steps: {args.sample_steps}")
    print(f"noise: {'zero' if args.zero_noise else f'fixed_random_seed_{args.seed}'}")
    return PolicyRuntime(
        config=config,
        data_config=data_config,
        model=model,
        input_transform=input_transform,
        device=device,
        rng=rng,
        zero_noise=bool(args.zero_noise),
        sample_steps=int(args.sample_steps),
        checkpoint=step_dir,
    )


def infer_action_chunk(policy: PolicyRuntime, image_rgb: np.ndarray, state: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
    sample = {
        "observation/image": np.asarray(image_rgb, dtype=np.uint8),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": np.asarray(prompt),
    }
    transformed = policy.input_transform(sample)
    transformed = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(policy.device)[None, ...], transformed)
    observation = _model.Observation.from_dict(transformed)
    shape = (1, policy.config.model.action_horizon, policy.config.model.action_dim)
    if policy.zero_noise:
        noise = torch.zeros(shape, dtype=torch.float32, device=policy.device)
    else:
        noise = torch.randn(shape, dtype=torch.float32, device=policy.device, generator=policy.rng)

    start = time.perf_counter()
    with torch.inference_mode():
        pred = policy.model.sample_actions(policy.device, observation, noise=noise, num_steps=policy.sample_steps)
    infer_ms = (time.perf_counter() - start) * 1000.0
    pred_norm = pred[0, :, :ACTION_DIM].detach().cpu().numpy()
    actions = _unnormalize_actions(
        pred_norm,
        policy.data_config.norm_stats,
        use_quantiles=policy.data_config.use_quantile_norm,
    )
    return actions.astype(np.float32, copy=False), infer_ms


class NeroEefRemotePolicy:
    def __init__(self, runtime: PolicyRuntime, *, prompt: str, actions_per_inference: int):
        self.runtime = runtime
        self.prompt = prompt
        self.actions_per_inference = actions_per_inference
        self.request_count = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.request_count += 1
        request_id = obs.get("request_id") or f"nero-eef-{self.request_count:06d}"
        image = np.asarray(obs["image"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected image as HWC RGB uint8, got shape {image.shape}")
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"expected state shape ({ACTION_DIM},), got {state.shape}")
        prompt = str(obs.get("prompt") or self.prompt)
        limit = max(1, int(obs.get("actions_per_inference") or self.actions_per_inference))

        started = time.perf_counter()
        actions, infer_ms = infer_action_chunk(self.runtime, image, state, prompt)
        actions = actions[:limit]
        return {
            "request_id": request_id,
            "actions": actions,
            "policy_timing": {
                "infer_ms": infer_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="nero_eef_pi05_pytorch_v1")
    parser.add_argument("--checkpoint-dir", default=None, help="Experiment dir or exact step dir.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load. Defaults to latest.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--validate-only", action="store_true", help="Load model and run one dummy inference.")
    args = parser.parse_args()

    args.repo_id = DEFAULT_REPO_ID
    args.model = "pi05"
    args.sample_steps = DEFAULT_SAMPLE_STEPS
    args.seed = 0
    args.zero_noise = False
    args.prompt = DEFAULT_TASK_PROMPT
    args.actions_per_inference = DEFAULT_ACTIONS_PER_INFERENCE
    return args


def main() -> int:
    args = parse_args()
    runtime = build_policy_runtime(args)
    policy = NeroEefRemotePolicy(runtime, prompt=args.prompt, actions_per_inference=args.actions_per_inference)
    if args.validate_only:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        state = np.array([-0.30, -0.17, 0.12, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = policy.infer({"image": image, "state": state})
        print(f"validation actions: shape={result['actions'].shape}")
        print(f"first action: {np.round(result['actions'][0], 5).tolist()}")
        return 0

    server = websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "policy_name": "nero_eef",
            "checkpoint": str(runtime.checkpoint),
            "prompt": args.prompt,
            "action_dim": ACTION_DIM,
            "action_horizon": runtime.config.model.action_horizon,
            "actions_per_inference": args.actions_per_inference,
            "image_format": "rgb_hwc_uint8",
            "image_size": [224, 224],
            "state_format": "x_y_z_quat_xyzw_gripper",
        },
    )
    print(f"Nero EEF policy server listening on {args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
