# 安装

```bash
cd LifEgo
PREDOWNLOAD=0 SKIP_HAND=0 SKIP_HARDWARE=1 bash setup.sh
```

# 关键运行命令

以下命令默认在仓库根目录执行。推荐使用当前项目环境：

```bash
PY=/home/ymq/miniconda3/envs/lifego/bin/python
```

## 1. 使用 WiLoR 处理 RGB 视频

export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

默认 WiLoR 权重目录：

```text
.cache/wilor_mini
```

运行单个示例视频：

```bash
$PY patches/process_examples_wilor.py --videos examples/ego_nero_easy.mp4
```

运行多个示例视频：

```bash
$PY patches/process_examples_wilor.py \
  --videos examples/ego_nero_easy.mp4 examples/ego_nero_h.mp4 examples/ego_nero_v.mp4
```

如果权重目录不在默认位置，可以显式指定：

```bash
$PY patches/process_examples_wilor.py \
  --videos examples/ego_nero_easy.mp4 \
  --wilor-pretrained-dir .cache/wilor_mini
```

主要输出：

```text
outputs/ego_nero_easy/preprocess/all_data/<frame>/rgb.png
outputs/ego_nero_easy/preprocess/all_data/<frame>/aria_cam_rgb.json
outputs/ego_nero_easy/preprocess/all_data/<frame>/wilor_hands.json
outputs/ego_nero_easy/preprocess/vis/wilor_hands_vis.mp4
```

## 2. 从 WiLoR 结果导出机器人 EEF 轨迹

```bash
$PY patches/export_robot_eef_from_wilor.py \
  --session outputs/ego_nero_easy \
  --out outputs/ego_nero_easy/robot_eef_scene_camera
```

主要输出：

```text
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.jsonl
outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.csv
```

## 3. 可选：修正 EEF 局部坐标轴

```bash
$PY patches/apply_humanego_eef_axis_correction.py \
  --input outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json \
  --out outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected
```

主要输出：

```text
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.jsonl
outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.csv
```

## 4. 对比真实机器人轨迹和 HumanEgo/WiLoR EEF

```bash
$PY patches/compare_realbot_humanego_eef.py \
  --humanego outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --real-pose-key tcp_pose \
  --out outputs/ego_nero_easy/compare_realbot_humanego
```

主要输出：

```text
outputs/ego_nero_easy/compare_realbot_humanego/comparison_summary.json
outputs/ego_nero_easy/compare_realbot_humanego/comparison_timeseries.csv
outputs/ego_nero_easy/compare_realbot_humanego/positions_absolute.png
outputs/ego_nero_easy/compare_realbot_humanego/positions_relative_to_start.png
outputs/ego_nero_easy/compare_realbot_humanego/orientation_errors.png
```

## 5. 构建 NERO MuJoCo 场景

依赖本机 URDF 资源：

```text
/home/ymq/code/agx_arm_urdf/nero
```

构建场景：

```bash
$PY patches/build_nero_mujoco_scene.py \
  --humanego-eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene
```

主要输出：

```text
outputs/mujoco_nero_scene/scene.xml
outputs/mujoco_nero_scene/README.md
outputs/mujoco_nero_scene/meshes/*.stl
```

## 6. 回放 HumanEgo/WiLoR EEF

生成 MP4：

```bash
$PY patches/replay_humanego_eef_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_humanego_eef_scene_camera_axis_corrected.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_humanego_eef_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --viewer
```

## 7. 回放真实机器人关节/tcp

生成 MP4：

```bash
$PY patches/replay_nero_realbot_joints_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_follower_joints.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_nero_realbot_joints_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --viewer
```

$PY patches/replay_nero_realbot_tcp_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --viewer


## 8. 解 NERO EEF IK

精简 IK 脚本读取 **HumanEgo/WiLoR EEF** 轨迹，求解 `joint1..joint7`，
默认让 MuJoCo `site:tcp` 跟踪该 EEF（目标来自 HumanEgo，不是真机 JSONL）。
其它帧可改 `--target-name`（如 `recorded_flange` / `jaw_parallel_flange`）。

```bash
$PY patches/solve_nero_eef_ik.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --target-name tcp \
  --out outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```

主要输出：

```text
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.json
```

调试少量帧：

```bash
$PY patches/solve_nero_eef_ik.py \
  --end 10 \
  --out /tmp/nero_eef_ik_debug.npz \
  --progress-every 1
```

## 9. 回放 NERO EEF IK

生成 MP4：

```bash
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_nero_eef_ik.mp4
```

打开交互式 MuJoCo viewer：

```bash
$PY patches/replay_nero_eef_ik_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --ik outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz \
  --viewer
```
## 10. 回放真机 TCP 轨迹

用 mocap marker 回放 JSONL 中的 `tcp_pose`（也可用 `--pose-key flange_pose` / `fk_pose`）。

原始数据（未校正姿态）：

```bash
$PY patches/replay_nero_realbot_tcp_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --pose-key tcp_pose \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_tcp.mp4
```

## 11. Nero TCP / 法兰 / 夹爪坐标系

**仿真 `site:tcp`（当前）= tool-centric**，在法兰→尖端线上：

```text
R_tcp = R_flange
p_tcp = p_flange + R_flange @ [0.13, 0, 0]   # link7 +X，朝夹爪
```

**JSONL `tcp_pose`（真机控制器日志，未改正）** 仍是：

```text
p_tcp_log = p_flange + R_flange @ [0, 0, 0.13]   # link7 +Z
```

| 帧 | 说明 |
|--|--|
| 绿 `recorded_flange` | link7 / `flange_pose` |
| 品红 `site:tcp` | 仿真 tool TCP（上式 +X） |
| 橙 `gripper_tip` | 夹爪尖端估计 |
| JSONL `tcp_pose` | 控制器旧定义（+Z），与 `site:tcp` 不同 |

IK / 视觉对齐请用仿真 `site:tcp`（joint FK）。不要直接把未转换的 JSONL `tcp_pose` 当 tip 线上的 TCP。

零位检查：

```bash
$PY patches/view_nero_zero_pose_frames.py --scene outputs/mujoco_nero_scene/scene.xml
```

三色跟随关节：

```bash
$PY patches/replay_nero_frames_legend_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl
```

### 三色对照：法兰 / TCP / 夹爪尖端

```bash
$PY patches/replay_nero_frames_legend_mujoco.py \
  --scene outputs/mujoco_nero_scene/scene.xml \
  --realbot examples/ego_nero_easy_real_bot.jsonl \
  --out outputs/mujoco_nero_scene/replays/ego_nero_easy_frames_legend.mp4
```

| 颜色 | 含义 |
|--|--|
| 绿 | 法兰 `recorded_flange` / `flange_pose` |
| 品红 | 控制器 TCP `site:tcp` / `tcp_pose` |
| 橙 | 夹爪尖端 `gripper_tip`（指中大致位置） |

默认三个 marker 都由 **关节 FK** 贴在臂上；加 `--markers-from-logged` 则绿/品红改用 JSONL 位姿。

### 零位静态检查（推荐先看）

机械臂保持 `joint1..7=0`，用同样的 MuJoCo passive viewer 标出三色坐标系：

```bash
$PY patches/view_nero_zero_pose_frames.py \
  --scene outputs/mujoco_nero_scene/scene.xml
```

绿=法兰，品红=TCP，橙=尖端。关闭窗口退出。

IK：`site:tcp` 跟踪 HumanEgo EEF（不要把真机 JSONL 当目标）：

```bash
$PY patches/solve_nero_eef_ik.py \
  --eef outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json \
  --target-name tcp \
  --out outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz
```
