"""Register the LifEgo Nero residual IK task with mjlab."""

from __future__ import annotations

from nero_mjlab_env import make_nero_mjlab_env_cfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task


TASK_ID = "Mjlab-Nero-IK-Residual"
PICK_PLACE_TASK_ID = "Mjlab-Nero-PickPlace-IK-Residual"


def nero_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="nero_ik_residual",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
    clip_actions=1.0,
  )


def register_nero_mjlab_task() -> None:
  if TASK_ID in list_tasks():
    pass
  else:
    register_mjlab_task(
      task_id=TASK_ID,
      env_cfg=make_nero_mjlab_env_cfg(
        num_envs=1024,
        sampling_mode="uniform",
        loop_reference=True,
      ),
      play_env_cfg=make_nero_mjlab_env_cfg(
        num_envs=1,
        sampling_mode="start",
        loop_reference=True,
      ),
      rl_cfg=nero_ppo_runner_cfg(),
      runner_cls=None,
    )

  if PICK_PLACE_TASK_ID not in list_tasks():
    register_mjlab_task(
      task_id=PICK_PLACE_TASK_ID,
      env_cfg=make_nero_mjlab_env_cfg(
        scene="outputs/nero_pick_place_human/mujoco_nero_scene_rl_example/scene.xml",
        eef=(
          "outputs/nero_pick_place_human/robot_eef_scene_camera_rl_example/"
          "robot_eef_trajectory.json"
        ),
        ik="outputs/nero_pick_place_human/nero_eef_ik_rl_example/nero_eef_ik.npz",
        num_envs=1024,
        sampling_mode="uniform",
        loop_reference=True,
      ),
      play_env_cfg=make_nero_mjlab_env_cfg(
        scene="outputs/nero_pick_place_human/mujoco_nero_scene_rl_example/scene.xml",
        eef=(
          "outputs/nero_pick_place_human/robot_eef_scene_camera_rl_example/"
          "robot_eef_trajectory.json"
        ),
        ik="outputs/nero_pick_place_human/nero_eef_ik_rl_example/nero_eef_ik.npz",
        num_envs=1,
        sampling_mode="start",
        loop_reference=True,
      ),
      rl_cfg=nero_ppo_runner_cfg(),
      runner_cls=None,
    )


register_nero_mjlab_task()
