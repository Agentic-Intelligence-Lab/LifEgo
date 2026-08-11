#!/usr/bin/env python3
"""mjlab scaffold for Nero IK residual-control experiments.

This file keeps the Nero-specific mjlab integration in patches/ and leaves the
thirdparty/mjlab package untouched.  It currently models only the robot tracking
problem: no interaction object, no object-tracking reward.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if "MUJOCO_GL" not in os.environ:
  if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    os.environ["MUJOCO_GL"] = "glfw"
  else:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np
import torch

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.managers import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS


def as_abs(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else REPO_ROOT / path


def log_stage(message: str) -> None:
  print(f"[nero_mjlab] {message}", flush=True)


def get_nero_scene_spec(scene_path: str | Path) -> mujoco.MjSpec:
  scene_path = as_abs(scene_path)
  root = ET.parse(scene_path).getroot()
  compiler = root.find("compiler")
  if compiler is not None and compiler.get("meshdir"):
    meshdir = Path(compiler.get("meshdir", ""))
    if not meshdir.is_absolute():
      compiler.set("meshdir", str(scene_path.parent / meshdir))
  worldbody = root.find("worldbody")
  if worldbody is not None:
    _remove_mocap_bodies(worldbody)
  return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def _remove_mocap_bodies(parent: ET.Element) -> None:
  for child in list(parent):
    if child.tag == "body" and child.get("mocap") == "true":
      parent.remove(child)
      continue
    _remove_mocap_bodies(child)


def get_nero_robot_cfg(scene_path: str | Path) -> EntityCfg:
  """Wrap the current generated Nero MJCF as one mjlab Entity.

  The existing scene already contains MuJoCo position actuators, so we wrap them
  with XmlActuatorCfg instead of recreating gains/ctrl ranges.
  """

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(joint_pos=None, joint_vel={".*": 0.0}),
    spec_fn=lambda: get_nero_scene_spec(scene_path),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=ALL_JOINTS,
          command_field="position",
        ),
      ),
      soft_joint_pos_limit_factor=1.0,
    ),
  )


class NeroIkReference:
  """Tensorized IK reference trajectory loaded from solve_nero_eef_ik.py output."""

  def __init__(self, ik_path: str | Path, device: str):
    data = np.load(as_abs(ik_path), allow_pickle=True)
    self.arm_joint_pos = torch.tensor(
      np.asarray(data["joint_qpos"], dtype=np.float32), device=device
    )
    self.gripper_width = torch.tensor(
      np.asarray(data["gripper_width_m"], dtype=np.float32), device=device
    )
    if "time_s" in data:
      self.time_s = torch.tensor(
        np.asarray(data["time_s"], dtype=np.float32), device=device
      )
    else:
      self.time_s = torch.arange(
        self.arm_joint_pos.shape[0], dtype=torch.float32, device=device
      )
    self.num_frames = int(self.arm_joint_pos.shape[0])
    if self.arm_joint_pos.ndim != 2 or self.arm_joint_pos.shape[1] != len(ARM_JOINTS):
      raise ValueError(
        f"Expected joint_qpos shape (T, {len(ARM_JOINTS)}), got "
        f"{tuple(self.arm_joint_pos.shape)}"
      )
    if self.gripper_width.shape[0] != self.num_frames:
      raise ValueError("gripper_width_m length must match joint_qpos frames")

  def arm(self, frame_ids: torch.Tensor) -> torch.Tensor:
    return self.arm_joint_pos[frame_ids]

  def width(self, frame_ids: torch.Tensor) -> torch.Tensor:
    return self.gripper_width[frame_ids]

  def full_joint_pos(self, frame_ids: torch.Tensor) -> torch.Tensor:
    arm = self.arm(frame_ids)
    width = self.width(frame_ids).unsqueeze(-1)
    return torch.cat((arm, 0.5 * width, -0.5 * width), dim=-1)


class NeroIkCommand(CommandTerm):
  cfg: "NeroIkCommandCfg"

  def __init__(self, cfg: "NeroIkCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.reference = NeroIkReference(cfg.ik_file, self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.metrics["ref_progress"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    progress = self.progress.unsqueeze(-1)
    return torch.cat((self.arm_joint_pos, self.gripper_width.unsqueeze(-1), progress), dim=-1)

  @property
  def progress(self) -> torch.Tensor:
    denom = max(self.reference.num_frames - 1, 1)
    return self.time_steps.float() / float(denom)

  @property
  def arm_joint_pos(self) -> torch.Tensor:
    return self.reference.arm(self.time_steps)

  @property
  def gripper_width(self) -> torch.Tensor:
    return self.reference.width(self.time_steps)

  @property
  def target_joint_pos(self) -> torch.Tensor:
    return self.reference.full_joint_pos(self.time_steps)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    frame = int(np.clip(frame, 0, self.reference.num_frames - 1))
    self.time_steps[env_ids] = frame
    self._write_reference_state(env_ids)

  def _update_metrics(self) -> None:
    self.metrics["ref_progress"] = self.progress

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self.time_steps[env_ids] = torch.randint(
        0, self.reference.num_frames, (len(env_ids),), device=self.device
      )
    else:
      raise ValueError(f"Unsupported sampling_mode: {self.cfg.sampling_mode}")
    self._write_reference_state(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps[env_ids] += 1
    if self.cfg.loop:
      self.time_steps[env_ids] %= self.reference.num_frames
    else:
      self.time_steps[env_ids] = torch.clamp(
        self.time_steps[env_ids], max=self.reference.num_frames - 1
      )

  def _write_reference_state(self, env_ids: torch.Tensor) -> None:
    q = self.target_joint_pos[env_ids]
    qd = torch.zeros_like(q)
    self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    self.robot.set_joint_position_target(q, env_ids=env_ids)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    del visualizer


@dataclass(kw_only=True)
class NeroIkCommandCfg(CommandTermCfg):
  ik_file: str
  entity_name: str = "robot"
  sampling_mode: Literal["start", "uniform"] = "start"
  loop: bool = False

  def build(self, env: ManagerBasedRlEnv) -> NeroIkCommand:
    return NeroIkCommand(self, env)


class NeroResidualJointAction(ActionTerm):
  cfg: "NeroResidualJointActionCfg"

  def __init__(self, cfg: "NeroResidualJointActionCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    arm_ids, _ = self._entity.find_joints(cfg.arm_joint_names, preserve_order=True)
    gripper_ids, _ = self._entity.find_joints(
      cfg.gripper_joint_names, preserve_order=True
    )
    self._arm_ids = torch.tensor(arm_ids, dtype=torch.long, device=self.device)
    self._gripper_ids = torch.tensor(gripper_ids, dtype=torch.long, device=self.device)
    self._joint_ids = torch.cat((self._arm_ids, self._gripper_ids), dim=0)
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_targets = torch.zeros(
      self.num_envs, len(ALL_JOINTS), device=self.device
    )

  @property
  def action_dim(self) -> int:
    return len(ARM_JOINTS) + 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != (self.num_envs, self.action_dim):
      raise ValueError(
        f"Expected action shape {(self.num_envs, self.action_dim)}, got "
        f"{tuple(actions.shape)}"
      )
    self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
    command = self._env.command_manager.get_term(self.cfg.command_name)
    if not isinstance(command, NeroIkCommand):
      raise TypeError(
        f"Command '{self.cfg.command_name}' must be NeroIkCommand, got "
        f"{type(command).__name__}"
      )

    arm_target = command.arm_joint_pos + self._raw_actions[:, :7] * self.cfg.arm_scale
    width = command.gripper_width + self._raw_actions[:, 7] * self.cfg.gripper_scale
    width = torch.clamp(width, self.cfg.gripper_width_range[0], self.cfg.gripper_width_range[1])
    target = torch.cat((arm_target, 0.5 * width[:, None], -0.5 * width[:, None]), dim=-1)

    limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids]
    self._processed_targets = torch.clamp(target, limits[..., 0], limits[..., 1])

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._processed_targets, joint_ids=self._joint_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class NeroResidualJointActionCfg(ActionTermCfg):
  command_name: str = "ik_ref"
  arm_joint_names: tuple[str, ...] = ARM_JOINTS
  gripper_joint_names: tuple[str, ...] = GRIPPER_JOINTS
  arm_scale: float = 0.08
  gripper_scale: float = 0.015
  gripper_width_range: tuple[float, float] = (0.0, 0.1)

  def build(self, env: ManagerBasedRlEnv) -> NeroResidualJointAction:
    return NeroResidualJointAction(self, env)


def nero_reference_command(env: ManagerBasedRlEnv, command_name: str = "ik_ref") -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, NeroIkCommand):
    raise TypeError(f"Expected NeroIkCommand, got {type(command).__name__}")
  return command.command


def nero_joint_target_error(
  env: ManagerBasedRlEnv,
  command_name: str = "ik_ref",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ALL_JOINTS),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, NeroIkCommand):
    raise TypeError(f"Expected NeroIkCommand, got {type(command).__name__}")
  return robot.data.joint_pos[:, asset_cfg.joint_ids] - command.target_joint_pos


def make_nero_mjlab_env_cfg(
  *,
  scene: str | Path = "outputs/mujoco_nero_scene/scene.xml",
  ik: str | Path = "outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz",
  num_envs: int = 1,
  episode_length_s: float = 2.4,
  sampling_mode: Literal["start", "uniform"] = "start",
  loop_reference: bool = False,
) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot", joint_names=ALL_JOINTS)

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos": ObservationTermCfg(
          func=joint_pos_rel,
          params={"asset_cfg": robot_cfg},
        ),
        "joint_vel": ObservationTermCfg(
          func=joint_vel_rel,
          params={"asset_cfg": robot_cfg},
        ),
        "ik_ref": ObservationTermCfg(func=nero_reference_command),
        "joint_target_error": ObservationTermCfg(func=nero_joint_target_error),
        "actions": ObservationTermCfg(func=last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "joint_pos": ObservationTermCfg(
          func=joint_pos_rel,
          params={"asset_cfg": robot_cfg},
        ),
        "joint_vel": ObservationTermCfg(
          func=joint_vel_rel,
          params={"asset_cfg": robot_cfg},
        ),
        "ik_ref": ObservationTermCfg(func=nero_reference_command),
        "joint_target_error": ObservationTermCfg(func=nero_joint_target_error),
        "actions": ObservationTermCfg(func=last_action),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  return ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=1.2,
      entities={"robot": get_nero_robot_cfg(scene)},
      terrain=None,
      extent=1.2,
    ),
    observations=observations,
    actions={
      "nero_residual": NeroResidualJointActionCfg(entity_name="robot"),
    },
    commands={
      "ik_ref": NeroIkCommandCfg(
        ik_file=str(as_abs(ik)),
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        sampling_mode=sampling_mode,
        loop=loop_reference,
      ),
    },
    rewards={},
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
    },
    events={},
    sim=SimulationCfg(
      nconmax=128,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
        cone="elliptic",
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.4,
      azimuth=145.0,
      elevation=-25.0,
    ),
    episode_length_s=episode_length_s,
    scale_rewards_by_dt=False,
  )


def run_smoke(args: argparse.Namespace) -> None:
  log_stage("building env cfg")
  cfg = make_nero_mjlab_env_cfg(
    scene=args.scene,
    ik=args.ik,
    num_envs=args.num_envs,
    episode_length_s=args.episode_length_s,
    sampling_mode=args.sampling_mode,
    loop_reference=args.loop_reference,
  )
  log_stage(f"creating ManagerBasedRlEnv on device={args.device}")
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  log_stage("env created")
  obs, extras = env.reset(seed=args.seed)
  log_stage("env reset complete")
  del extras
  print(f"reset ok: num_envs={env.num_envs} action_dim={env.action_manager.total_action_dim}")
  for name, value in obs.items():
    if isinstance(value, torch.Tensor):
      print(f"obs[{name}] shape={tuple(value.shape)} device={value.device}")
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=args.device)
  for i in range(args.steps):
    obs, reward, terminated, truncated, extras = env.step(action)
    del obs, extras
    print(
      f"step={i:03d} reward_mean={reward.mean().item():.3f} "
      f"terminated={int(terminated.sum().item())} truncated={int(truncated.sum().item())}"
    )
  env.close()


def run_scene_check(args: argparse.Namespace) -> None:
  from mjlab.scene import Scene

  log_stage("building env cfg")
  cfg = make_nero_mjlab_env_cfg(
    scene=args.scene,
    ik=args.ik,
    num_envs=args.num_envs,
    episode_length_s=args.episode_length_s,
    sampling_mode=args.sampling_mode,
    loop_reference=args.loop_reference,
  )
  log_stage("creating Scene on cpu")
  scene = Scene(cfg.scene, device="cpu")
  log_stage("compiling Scene")
  model = scene.compile()
  print(
    "scene compile ok: "
    f"num_envs={cfg.scene.num_envs} nq={model.nq} nv={model.nv} nu={model.nu} "
    f"nbody={model.nbody} ngeom={model.ngeom}"
  )
  print("actuators:", ", ".join(model.actuator(i).name for i in range(model.nu)))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--check",
    choices=["scene", "env"],
    default="scene",
    help="scene compiles the mjlab MjSpec only; env instantiates MuJoCo Warp and steps.",
  )
  parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
  parser.add_argument("--ik", default="outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--episode-length-s", type=float, default=2.4)
  parser.add_argument("--sampling-mode", choices=["start", "uniform"], default="start")
  parser.add_argument("--loop-reference", action="store_true")
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()
  if args.check == "scene":
    run_scene_check(args)
  else:
    run_smoke(args)


if __name__ == "__main__":
  main()
