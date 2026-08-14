#!/usr/bin/env python3
"""Headless evaluation for the LifEgo Nero mjlab residual IK policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


NEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = NEW_DIR.parent
if str(NEW_DIR) not in sys.path:
  sys.path.insert(0, str(NEW_DIR))

import rl_env_mjlab  # noqa: F401
from rl_env_mjlab import NeroIkCommand
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_error_magnitude


def as_abs(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else REPO_ROOT / path


def make_policy(name: str, env: RslRlVecEnvWrapper, checkpoint: Path | None, device: str):
  if name == "zero":

    def zero_policy(obs) -> torch.Tensor:
      del obs
      return torch.zeros(
        env.num_envs, env.num_actions, dtype=torch.float32, device=device
      )

    return zero_policy

  if checkpoint is None:
    raise ValueError(f"Policy '{name}' requires a checkpoint")
  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def collect_metrics(env: RslRlVecEnvWrapper, actions: torch.Tensor) -> dict[str, float]:
  raw_env = env.unwrapped
  robot = raw_env.scene["robot"]
  site_ids, _ = robot.find_sites(("tcp",), preserve_order=True)
  site_id = site_ids[0]
  command = raw_env.command_manager.get_term("ik_ref")
  if not isinstance(command, NeroIkCommand):
    raise TypeError(f"Expected NeroIkCommand, got {type(command).__name__}")

  current_pos = robot.data.site_pos_w[:, site_id]
  target_pos = command.eef_pos
  pos_err = torch.linalg.norm(current_pos - target_pos, dim=-1)

  current_quat = robot.data.site_quat_w[:, site_id]
  target_quat_xyzw = command.eef_quat_xyzw
  target_quat_wxyz = torch.cat(
    (target_quat_xyzw[:, 3:4], target_quat_xyzw[:, 0:3]), dim=-1
  )
  ang_err = quat_error_magnitude(target_quat_wxyz, current_quat)

  prev_actions = raw_env.action_manager.prev_action
  action_rate = torch.sum(torch.square(actions - prev_actions), dim=-1)
  tracking_reward = torch.exp(-(pos_err**2) / (0.05**2)) * torch.exp(
    -(ang_err**2) / (0.5**2)
  )

  return {
    "pos_err_m_mean": float(pos_err.mean().item()),
    "pos_err_m_p95": float(torch.quantile(pos_err, 0.95).item()),
    "pos_err_m_max": float(pos_err.max().item()),
    "ang_err_deg_mean": float(torch.rad2deg(ang_err).mean().item()),
    "ang_err_deg_p95": float(torch.quantile(torch.rad2deg(ang_err), 0.95).item()),
    "ang_err_deg_max": float(torch.rad2deg(ang_err).max().item()),
    "tracking_reward_mean": float(tracking_reward.mean().item()),
    "action_rate_l2_mean": float(action_rate.mean().item()),
  }


def evaluate_policy(
  *,
  policy_name: str,
  checkpoint: Path | None,
  steps: int,
  num_envs: int,
  seed: int,
  device: str,
) -> dict[str, float | int | str | None]:
  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = num_envs
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=load_rl_cfg(args.task).clip_actions)
  raw_env.seed(seed)
  obs, _ = env.reset()
  policy = make_policy(policy_name, env, checkpoint, device)

  sums: dict[str, float] = {}
  last_obs = obs
  for _ in range(steps):
    with torch.no_grad():
      actions = policy(last_obs)
    metrics = collect_metrics(env, actions)
    for key, value in metrics.items():
      sums[key] = sums.get(key, 0.0) + value
    last_obs, _, _, _ = env.step(actions)

  env.close()
  out = {
    "policy": policy_name,
    "checkpoint": str(checkpoint) if checkpoint else None,
    "steps": steps,
    "num_envs": num_envs,
    "seed": seed,
  }
  out.update({key: value / steps for key, value in sums.items()})
  return out


def main() -> None:
  global args
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", default="Mjlab-Nero-PickPlace-IK-Residual")
  parser.add_argument("--checkpoint", action="append", default=[])
  parser.add_argument("--include-zero", action="store_true")
  parser.add_argument("--steps", type=int, default=240)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  results = []
  if args.include_zero:
    results.append(
      evaluate_policy(
        policy_name="zero",
        checkpoint=None,
        steps=args.steps,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
      )
    )
  for path in args.checkpoint:
    checkpoint = as_abs(path)
    results.append(
      evaluate_policy(
        policy_name=checkpoint.stem,
        checkpoint=checkpoint,
        steps=args.steps,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
      )
    )

  text = json.dumps(results, indent=2)
  print(text)
  if args.out:
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
