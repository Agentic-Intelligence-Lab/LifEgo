## RL example：`nero_pick_place_human`

`nero_pick_place_human` 这段 RGB EEF 已整理出一个可直接用于 RL 的 example。它不是逐帧对齐真机 TCP，而是用真机轨迹估计可执行工作区，把 RGB 估计出的 EEF 放进 Nero 可达区域，并用真机平均 TCP 朝向做整体姿态水平化。

当前最终产物：

```text
outputs/nero_pick_place_human/robot_eef_scene_camera_rl_example/robot_eef_trajectory.json
outputs/nero_pick_place_human/mujoco_nero_scene_rl_example/scene.xml
outputs/nero_pick_place_human/nero_eef_ik_rl_example/nero_eef_ik.npz
outputs/nero_pick_place_human/mujoco_nero_scene_rl_example/replays/nero_pick_place_rl_example_eef.mp4
outputs/nero_pick_place_human/mujoco_nero_scene_rl_example/replays/nero_pick_place_realbot_vs_rl_example_ik.mp4
```

这版 example 的 workspace placement 命令：

```bash
SESSION=outputs/nero_pick_place_human

$PY patches/offset_humanego_eef_to_realbot_workspace.py \
  --eef $SESSION/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/nero_pick_place_realbot.jsonl \
  --out $SESSION/robot_eef_scene_camera_rl_example \
  --anchor shifted-end \
  --shape-scale 0.6 1.1 0.1 \
  --match-real-mean-orientation
```

关键含义：

| 参数 | 含义 |
|--|--|
| `--anchor shifted-end` | 先按 bbox center 放进真机工作区，再保持平移后终点基本不变 |
| `--shape-scale 0.6 1.1 0.1` | 围绕终点对轨迹形状做轻量缩放：起点更远离基座，z 变化更平 |
| `--match-real-mean-orientation` | 用真机 TCP 平均朝向左乘修正 HumanEgo EEF 姿态，使 EEF 更水平 |

当前 RL example 的 IK 误差：

```text
pos mean/max: 11.3/26.1 mm
ang mean/max: 2.01/4.67 deg
```

如需从头重建该 example：

```bash
AGX_ARM_URDF_ROOT=/home/ymq/code/agx_arm_urdf \
$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef $SESSION/robot_eef_scene_camera_rl_example/robot_eef_trajectory.json \
  --realbot examples/nero_pick_place_realbot.jsonl \
  --out $SESSION/mujoco_nero_scene_rl_example

PYOPENGL_PLATFORM=egl MUJOCO_GL=egl \
$PY patches/solve_nero_eef_ik.py \
  --scene $SESSION/mujoco_nero_scene_rl_example/scene.xml \
  --eef $SESSION/robot_eef_scene_camera_rl_example/robot_eef_trajectory.json \
  --target-name tcp \
  --out $SESSION/nero_eef_ik_rl_example/nero_eef_ik.npz \
  --gl-backend egl
```

## mjlab RL residual-control

已注册两个 mjlab task：

| task | 用途 |
|--|--|
| `Mjlab-Nero-IK-Residual` | 通用默认任务，使用 `outputs/ego_nero_easy` 路径 |
| `Mjlab-Nero-PickPlace-IK-Residual` | pick-place RL example，使用上面的 `robot_eef_scene_camera_rl_example` / `nero_eef_ik_rl_example` |

环境定义：

| 项 | 当前配置 |
|--|--|
| action | 8 维 residual：7 个 arm joint offset + 1 个 gripper width offset |
| actor obs | 52 维：joint pos 9 + joint vel 9 + IK reference 17 + joint target error 9 + last action 8 |
| critic obs | 同 actor，52 维 |
| command | `ik_ref`：加载 HumanEgo EEF JSON + mink IK `.npz` |
| reward `tracking` | `site:tcp` 跟踪 HumanEgo EEF 的位置和姿态，weight `1.0` |
| reward `action_smoothness` | action rate L2，weight `-0.01` |
| termination | timeout |

训练默认是单机单 GPU 向量化并行，不是多机分布式。`Mjlab-Nero-PickPlace-IK-Residual` 默认 `num_envs=1024`，每个 PPO iteration 采样：

```text
1024 envs * 24 steps/env = 24576 steps
```

训练：

```bash
$PY patches/train_nero_mjlab.py Mjlab-Nero-PickPlace-IK-Residual \
  --agent.max-iterations 500 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --log-root outputs/nero_pick_place_human/rl_logs
```

从已有 checkpoint 续训：

```bash
$PY patches/train_nero_mjlab.py Mjlab-Nero-PickPlace-IK-Residual \
  --agent.resume True \
  --agent.load-run 2026-08-11_19-30-41 \
  --agent.load-checkpoint model_49.pt \
  --agent.max-iterations 500 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --log-root outputs/nero_pick_place_human/rl_logs
```

本地 viewer。不要在 SSH session 里启动 viewer，由本机图形环境运行：

```bash
# dummy policy，看环境/reference 是否正常
$PY patches/play_nero_mjlab.py Mjlab-Nero-PickPlace-IK-Residual \
  --agent zero \
  --viewer native

# 加载本地 checkpoint
$PY patches/play_nero_mjlab.py Mjlab-Nero-PickPlace-IK-Residual \
  --checkpoint-file outputs/nero_pick_place_human/rl_logs/nero_ik_residual/2026-08-11_19-37-49/model_548.pt \
  --log-root outputs/nero_pick_place_human/rl_logs \
  --viewer native
```

Headless 评估：

```bash
$PY patches/evaluate_nero_mjlab_policy.py Mjlab-Nero-PickPlace-IK-Residual \
  --include-zero \
  --checkpoint outputs/nero_pick_place_human/rl_logs/nero_ik_residual/2026-08-11_19-30-41/model_49.pt \
  --checkpoint outputs/nero_pick_place_human/rl_logs/nero_ik_residual/2026-08-11_19-37-49/model_548.pt \
  --steps 240 \
  --num-envs 1024 \
  --seed 123 \
  --out outputs/nero_pick_place_human/rl_logs/nero_ik_residual/2026-08-11_19-37-49/eval_zero_model_49_model_548.json
```

当前同口径评估结果：

| policy | pos mean | pos p95 | ori mean | ori p95 | tracking reward | action rate |
|--|--:|--:|--:|--:|--:|--:|
| zero / mink IK only | 40.7 mm | 102.1 mm | 7.88 deg | 24.84 deg | 0.613 | 0.000 |
| `model_49` | 19.7 mm | 67.1 mm | 8.26 deg | 16.70 deg | 0.836 | 0.833 |
| `model_548` | 15.6 mm | 48.1 mm | 6.17 deg | 11.75 deg | 0.881 | 2.366 |

`model_548` 的 tracking 更好，但离线评估里的 action rate 更大。后续面向真机执行时，建议调高 `action_smoothness` 权重，例如从 `-0.01` 提到 `-0.03` 或 `-0.05` 后重新训练。